import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

import copy
import json

import h5py
import os
import numpy as np
from tqdm import tqdm

from interp.dataset import HDF5Dataset, custom_collate, nested_custom_collate
from interp.config import load_config, create_model_from_config

import argparse

def one_hot_encode(upd_pi):
    """
    A more efficient (vectorized) version using scatter.

    Args:
        upd_pi: A torch tensor of shape (N, T, D).

    Returns:
        A torch tensor of shape (N, T, D, D).
    """
    N, T, D = upd_pi.shape
    one_hot = torch.zeros(N, T, D, D, device=upd_pi.device)
    # Important: Use .long() to convert class indices to integers
    return one_hot.scatter(-1, upd_pi.long().unsqueeze(-1), 1.0)

def train_one_epoch(model, dataloader, optimizer, device, epoch, num_epochs):
    model.train()
    running_loss = 0.0
    running_dist_loss = 0.0
    running_class_loss = 0.0
    total_samples = 0
    class_loss = CrossEntropyLoss()

    with tqdm(total=len(dataloader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
        for batch in dataloader:
            batch_i =batch['all_cumsum']
            time_i = batch['all_cumsum_timesteps']
            batch_info = batch['batch']
            no_graphs = batch['num_graphs']
            hidden_states = batch['hidden_states'].to(device) # (T, H, total_D)
            edge_w = batch['edge_weights'].float().to(device) # (total_D, total_D)
            
            upd_pi = batch['upd_pi'].long().to(device) # (T, total_D)
            upd_d = batch['upd_d'].float().to(device) # (T, total_D)

            # Prepare inputs and targets
            inputs = hidden_states

            optimizer.zero_grad()
            class_out, dist_out = model(inputs, edge_w, batch_info, no_graphs, time_i)
            # class_out is a list of (T-1, D, D) shaped vectors. (D may not be constant across vectors) The number of vectors in the list is equal to no_graphs. For each vector
            # dist_out is a list of (T-1, D) shaped vectors. again D may not be constant across vectors.
            loss = 0 # batch loss  
            dist_loss = 0 # batch dist loss
            class_losses = 0 # batch class loss
            samples = 0 # number of samples in the batch
            for i in range(len(class_out)):
                # print(batch_i, i)
                dist_ins = dist_out[i] # (T-1, D)
                class_ins = class_out[i].permute((0,2,1)) # (T-1, D, D) second dimension is the source node

                dist_l = F.mse_loss(dist_ins, upd_d[time_i[i]+1:time_i[i+1],batch_i[i]:batch_i[i+1]])  # Use MSE loss
                class_l = class_loss(class_ins, upd_pi[time_i[i]+1:time_i[i+1],batch_i[i]:batch_i[i+1]])
                loss += (dist_l + class_l) * class_ins.size(1) * class_ins.size(0)
                dist_loss += dist_l * class_ins.size(1) * class_ins.size(0)
                class_losses += class_l * class_ins.size(1) * class_ins.size(0)
                samples += class_ins.size(1) * class_ins.size(0)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_dist_loss += dist_loss.item()
            running_class_loss += class_losses.item()

            total_samples += samples
            pbar.update(1)
            pbar.set_postfix({'dist_loss': f'{running_dist_loss/total_samples:.4f}', 'class_loss': f'{running_class_loss/total_samples:.4f}', 'loss': f'{running_loss/total_samples:.4f}'})

    epoch_loss = running_loss / total_samples
    return epoch_loss


def evaluate(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    total_samples = 0

    class_loss = CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in dataloader:
            batch_i =batch['all_cumsum']
            time_i = batch['all_cumsum_timesteps']
            batch_info = batch['batch']
            no_graphs = batch['num_graphs']
            hidden_states = batch['hidden_states'].to(device) # (T, H, total_D)
            edge_w = batch['edge_weights'].float().to(device) # (total_D, total_D)
            
            upd_pi = batch['upd_pi'].long().to(device) # (T, total_D)
            upd_d = batch['upd_d'].float().to(device) # (T, total_D)

            inputs = hidden_states
            
            class_out, dist_out = model(inputs, edge_w, batch_info, no_graphs, time_i)
            for i in range(len(class_out)):
                dist_ins = dist_out[i] # (T-1, D)
                class_ins = class_out[i].permute((0,2,1)) # (T-1, D, D) second dimension is the source node

                dist_l = F.mse_loss(dist_ins, upd_d[time_i[i]+1:time_i[i+1],batch_i[i]:batch_i[i+1]])  # Use MSE loss
                class_l = class_loss(class_ins, upd_pi[time_i[i]+1:time_i[i+1],batch_i[i]:batch_i[i+1]])

                running_loss += (dist_l + class_l).item() * class_ins.size(0) * class_ins.size(1)
                total_samples += class_ins.size(0) * class_ins.size(1) # the total number of samples is not the number of graphs, but rather the number of nodes in the graphs times the number of timesteps evaluated (T-1)

    avg_loss = running_loss / total_samples

    return avg_loss

def train_one_epoch_joint(model, dataloader, optimizer, device, epoch, num_epochs, train_algos):
    ''' Train algos is the only algorithms to train on (loss is backpropagated for these algorithms only)'''    
    model.train()
    # for the parts of the model that are not train_algos, freeze the parameters
    # first do it for node encoders
    for key in model.node_encoders.keys():
        if key not in train_algos:
            for param in model.node_encoders[key].parameters():
                param.requires_grad = False
        else:
            for param in model.node_encoders[key].parameters():
                param.requires_grad = True
    # then do it for the edge encoders
    for key in model.edge_encoders.keys():
        if key not in train_algos:
            for param in model.edge_encoders[key].parameters():
                param.requires_grad = False
        else:
            for param in model.edge_encoders[key].parameters():
                param.requires_grad = True
    # then do it for update_classifiers
    for key in model.update_classifiers.keys():
        if key not in train_algos:
            for param in model.update_classifiers[key].parameters():
                param.requires_grad = False
        else:
            for param in model.update_classifiers[key].parameters():
                param.requires_grad = True
    # then do it for dist_predictors
    for key in model.dist_predictors.keys():
        if key not in train_algos:
            for param in model.dist_predictors[key].parameters():
                param.requires_grad = False
        else:
            for param in model.dist_predictors[key].parameters():
                param.requires_grad = True

    running_loss = {algo: 0.0 for algo in train_algos}
    running_dist_loss = {algo: 0.0 for algo in train_algos}
    running_class_loss = {algo: 0.0 for algo in train_algos}
    total_samples = {algo: 0 for algo in train_algos}
    class_loss_metric = CrossEntropyLoss()

    with tqdm(total=len(dataloader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
        for batch in dataloader:
            batch_i = {algo: batch[algo]['all_cumsum'] for algo in train_algos}
            time_i = {algo: batch[algo]['all_cumsum_timesteps'] for algo in train_algos}
            batch_info = {algo: batch[algo]['batch'] for algo in train_algos}
            no_graphs = {algo: batch[algo]['num_graphs'] for algo in train_algos}
            hidden_states = {algo: batch[algo]['hidden_states'].to(device) for algo in train_algos} # dictionary of (T, H, total_D)
            edge_w = {algo: batch[algo]['edge_weights'].float().to(device) for algo in train_algos} # dictionary of (total_D, total_D)
            
            upd_pi = {algo: batch[algo]['upd_pi'].long().to(device) for algo in train_algos} # dictionary of (T, total_D)
            upd_d = {algo: batch[algo]['upd_d'].float().to(device) for algo in train_algos} # dictionary of (T, total_D)

            # Prepare inputs and targets
            inputs = hidden_states

            optimizer.zero_grad()
            class_out_dic, dist_out_dic = model(inputs, edge_w, batch_info, no_graphs, time_i)
            # class_out_dic is a dictionary of class_out, dist_out for each algorithm
            # class_out is a list of (T-1, D, D) shaped vectors. (D may not be constant across vectors) The number of vectors in the list is equal to no_graphs. For each vector
            # dist_out is a list of (T-1, D) shaped vectors. again D may not be constant across vectors.
            algo_losses = {algo : 0 for algo in train_algos} # losses for the batch
            dist_losses = {algo : 0 for algo in train_algos} # losses for the batch
            class_losses = {algo : 0 for algo in train_algos} # losses for the batch
            samples = {algo : 0 for algo in train_algos} # number of samples for the batch
            for algo in train_algos:
                class_out = class_out_dic[algo]
                dist_out = dist_out_dic[algo]
                for i in range(len(class_out)):
                    dist_ins = dist_out[i] # (T-1, D)
                    class_ins = class_out[i].permute((0,2,1)) # (T-1, D, D) second dimension is the source node

                    dist_l = F.mse_loss(dist_ins, upd_d[algo][time_i[algo][i]+1:time_i[algo][i+1],batch_i[algo][i]:batch_i[algo][i+1]])  # Use MSE loss
                    class_l = class_loss_metric(class_ins, upd_pi[algo][time_i[algo][i]+1:time_i[algo][i+1],batch_i[algo][i]:batch_i[algo][i+1]])
                    algo_losses[algo] += (dist_l + class_l) * class_ins.size(1) * class_ins.size(0)
                    dist_losses[algo] += dist_l *  class_ins.size(1) * class_ins.size(0)
                    class_losses[algo] += class_l * class_ins.size(1) * class_ins.size(0)
                    samples[algo] += class_ins.size(1) * class_ins.size(0) # the total number of samples is not the number of graphs, but rather the number of nodes in the graphs times the number of timesteps evaluated (T-1)
            loss = sum(algo_losses.values()) # only backpropagate for train_algos
            loss.backward()
            optimizer.step()

            for algo in algo_losses.keys():
                running_loss[algo] += algo_losses[algo].item()
                running_dist_loss[algo] += dist_losses[algo].item()
                running_class_loss[algo] += class_losses[algo].item()
                total_samples[algo] += samples[algo]

            pbar.update(1)
            loss_dict = {algo: f"{running_loss[algo]/total_samples[algo]:.4f}" for algo in algo_losses.keys()}
            loss_dict.update({f'dl_{algo[:3]}': f"{running_dist_loss[algo]/total_samples[algo]:.4f}" for algo in algo_losses.keys()})
            loss_dict.update({f'cl_{algo[:3]}': f"{running_class_loss[algo]/total_samples[algo]:.4f}" for algo in algo_losses.keys()})
            pbar.set_postfix(loss_dict)

    loss_dict = {algo: running_loss[algo]/total_samples[algo] for algo in train_algos if total_samples[algo] > 0}
    epoch_loss = sum(loss_dict.values())/len(loss_dict)
    return epoch_loss, loss_dict

def evaluate_joint(model, dataloader, device):
    ''' Here dataloaders output a dictionary of dictionaries. The outer dictionary has keys as the algorithm names and the inner dictionary has keys as the algorithm names and the values are the dataloader for that algorithm. '''
    model.eval()
    algorithms = list(dataloader.dataset[0].keys())
    running_loss = {algo: 0.0 for algo in algorithms}
    total_samples = {algo: 0 for algo in algorithms}
    class_loss_metric = CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            batch_i = {algo: batch[algo]['all_cumsum'] for algo in algorithms}
            time_i = {algo: batch[algo]['all_cumsum_timesteps'] for algo in algorithms}
            batch_info = {algo: batch[algo]['batch'] for algo in algorithms}
            no_graphs = {algo: batch[algo]['num_graphs'] for algo in algorithms}  
            hidden_states = {algo: batch[algo]['hidden_states'].to(device) for algo in algorithms} # dictionary of (T, H, total_D)
            edge_w = {algo: batch[algo]['edge_weights'].float().to(device) for algo in algorithms} # dictionary of (total_D, total_D)
            
            upd_pi = {algo: batch[algo]['upd_pi'].long().to(device) for algo in algorithms} # dictionary of (T, total_D)
            upd_d = {algo: batch[algo]['upd_d'].float().to(device) for algo in algorithms} # dictionary of (T, total_D)   

            inputs = hidden_states

            class_out_dic, dist_out_dic = model(inputs, edge_w, batch_info, no_graphs, time_i)
            for algo in class_out_dic.keys():
                class_out = class_out_dic[algo]
                dist_out = dist_out_dic[algo]
                for i in range(len(class_out)):
                    dist_ins = dist_out[i] # (T-1, D)
                    class_ins = class_out[i].permute((0,2,1)) # (T-1, D, D) second dimension is the source node

                    dist_l = F.mse_loss(dist_ins, upd_d[algo][time_i[algo][i]+1:time_i[algo][i+1],batch_i[algo][i]:batch_i[algo][i+1]])  # Use MSE loss
                    class_l = class_loss_metric(class_ins, upd_pi[algo][time_i[algo][i]+1:time_i[algo][i+1],batch_i[algo][i]:batch_i[algo][i+1]])
                    running_loss[algo] += (dist_l + class_l).item() * class_ins.size(1) * class_ins.size(0)
                    total_samples[algo] += class_ins.size(1) * class_ins.size(0) # the total number of samples is not the number of graphs, but rather the number of nodes in the graphs times the number of timesteps evaluated (T-1)

    loss_dict = {algo: running_loss[algo]/total_samples[algo] for algo in algorithms if total_samples[algo] > 0}
    avg_loss = sum(loss_dict.values())/len(loss_dict)  
    return avg_loss , loss_dict

def main(args):
    # Load configuration
    config = load_config(args.config)
    
    # Extract training parameters from config
    training_config = config.get('training', {})
    learning_rate = float(training_config.get('learning_rate', 1e-3))
    batch_size = int(training_config.get('batch_size', 16))
    num_epochs = int(training_config.get('num_epochs', 100))
    patience = int(training_config.get('patience', 5))
    dataset_name = training_config.get('dataset', 'all')
    alg = training_config.get('algo', 'bellman_ford')
    model_name = training_config.get('model_name', 'mlp_diff')
    
    # Override with command line arguments if provided
    if args.learning_rate is not None:
        print(f"Overriding learning rate from {learning_rate} to {args.learning_rate}")
        learning_rate = args.learning_rate
        
    if args.batch_size is not None:
        print(f"Overriding batch size from {batch_size} to {args.batch_size}")
        batch_size = args.batch_size
        
    if args.num_epochs is not None:
        print(f"Overriding number of epochs from {num_epochs} to {args.num_epochs}")
        num_epochs = args.num_epochs
        
    if args.dataset is not None:
        print(f"Overriding dataset from {dataset_name} to {args.dataset}")
        dataset_name = args.dataset
        
    if args.algo is not None:
        print(f"Overriding algorithm from {alg} to {args.algo}")
        alg = args.algo
        
    if args.model_name is not None:
        print(f"Overriding model name from {model_name} to {args.model_name}")
        model_name = args.model_name
        
    if alg in ["bellman_ford_bfs", "bellman_ford_dijkstra", "bellman_ford_prims"]:
        joint_training = True
        scheduler = training_config.get('scheduler', {"type": "joint"})
        algorithms = training_config.get('algorithms', ["bellman_ford", "bfs"])
        print("Performing joint training on the following algorithms: ", algorithms)
        print("Using schedule for joint training: ", scheduler)
    else:
        joint_training = False
        print("Performing single algorithm training on the following algorithm: ", alg)

    if args.device == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            raise ValueError("CUDA is not available")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Model name and paths
    checkpoint_dir = os.path.join("interp_checkpoints", alg, f"{model_name}_{dataset_name}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_save_path = os.path.join(checkpoint_dir, f"{model_name}_{dataset_name}.pth")
    config_save_path = os.path.join(checkpoint_dir, f"{model_name}_{dataset_name}_config.json")
    
    # Data loading
    if args.sync:
        data_root = os.path.join("data", alg + "_sync")
    else:
        data_root = os.path.join("data", alg)
    
    if dataset_name == "all":
        train_pth = os.path.join(data_root, "interp_data_all.h5")
    elif dataset_name == "16":
        train_pth = os.path.join(data_root, "interp_data_16.h5")
    elif dataset_name == "8":
        train_pth = os.path.join(data_root, "interp_data_8.h5")
    elif dataset_name == "OOD":
        train_pth = os.path.join(data_root, "interp_data_OOD.h5")
    else:
        raise ValueError(f"Dataset {dataset_name} not found")

    # Validation always on graphs of size 16
    val_pth = os.path.join(data_root, "interp_data_16_eval.h5")

    train_dataset = HDF5Dataset(train_pth, nested=joint_training)
    if joint_training:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=nested_custom_collate)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    val_dataset = HDF5Dataset(val_pth, nested=joint_training)
    if joint_training:
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=nested_custom_collate)
    else:
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

    # Create model from config
    model = create_model_from_config(config).to(device)
    
    # Load pretrained weights if provided
    if args.resume:
        model.load_state_dict(torch.load(model_save_path))
        print(f"Loaded pretrained weights from {model_save_path}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    counter = 0  # Counter for patience
    
    for epoch in range(num_epochs):
        ''' If we are using joint training then we need to use different training and evaluation functions. Furthermore, we need to specify a scheduler for the joint training. The scheduler is not for the optimizer, but rather decides the scehdule for joint training. It is a dictionary with the following keys:
        - 'type': the type of schedule. One of 'joint', 'alternating', 'sequential', 'fine_tune'
                 If 'joint' then all algorithms are trained simultaneously (all losses are backpropagated).
                 If 'alternating' then each algorithm is trained in turn according to some proportional schedule.
                 If 'sequential' then each algorithm is trained sequentially, without cycling back to the first algorithm.
                 If 'fine_tune' then some algorithms are trained first (only their loss is backpropagated), then all are trained jointly (all losses are backpropagated)
        - 'schedule': a list of integers describing the schedule. Only used if 'alternating', 'sequential' or 'fine_tune'.
                If 'alternating' then the schedule is a list of integers which describe how many epochs each algorithm should be trained before the schedule repeats.
                If 'sequential' then the schedule is a list of integers which describe how many epochs each algorithm should be trained before moving on to the next one.
                If 'fine_tune' then the schedule is a list of integers which describe the epoch at which each algorithm starts to train on.
        '''
        if joint_training:            
            if scheduler['type'] == 'joint':
                train_algos = algorithms # all algorithms are trained jointly
                train_loss, loss_dict = train_one_epoch_joint(model, train_dataloader, optimizer, device, epoch, num_epochs, train_algos=train_algos)
            elif scheduler['type'] == 'alternating':
                assert len(scheduler['schedule']) == len(algorithms) # each algo should have a schedule
                repeat_every = sum(scheduler['schedule'])
                cycle_count = epoch % repeat_every
                # find which algo to train on this epoch based on the cycle count
                algo_index = np.where(np.array(scheduler['schedule']).cumsum() > cycle_count)[0].min()
                train_algos = [algorithms[algo_index]]
                train_loss, loss_dict = train_one_epoch_joint(model, train_dataloader, optimizer, device, epoch, num_epochs, train_algos=train_algos)
            elif scheduler['type'] == 'sequential':
                assert len(scheduler['schedule']) == len(algorithms) # each algo should have a schedule
                assert sum(scheduler['schedule']) >= num_epochs # the sum of the schedule should be the total number of epochs
                # find which algo to train on this epoch based on the cycle count
                algo_index = np.where(np.array(scheduler['schedule']).cumsum() > epoch)[0].min()
                train_algos = [algorithms[algo_index]]
                train_loss, loss_dict = train_one_epoch_joint(model, train_dataloader, optimizer, device, epoch, num_epochs, train_algos=train_algos)
            elif scheduler['type'] == 'fine_tune':
                assert len(scheduler['schedule']) == len(algorithms) # each algo should have a schedule
                # this time the scheduler is the epoch at which the corresponding algo starts to train on
                to_train_idx = np.where(np.array(scheduler['schedule']) <= epoch)[0]
                train_algos = [algorithms[i] for i in to_train_idx]
                train_loss, loss_dict = train_one_epoch_joint(model, train_dataloader, optimizer, device, epoch, num_epochs, train_algos=train_algos)
            
            loss_str = " | ".join([f"{algo}: {loss_dict[algo]:.4f}" for algo in loss_dict.keys()])
            print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f} | {loss_str}")
        else:
            train_loss = train_one_epoch(model, train_dataloader, optimizer, device, epoch, num_epochs)
            print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}")
        train_losses.append(train_loss)

        # Validation
        if joint_training:
            val_loss, val_loss_dict = evaluate_joint(model, val_dataloader, device) 
            loss_str = " | ".join([f"{algo}: {val_loss_dict[algo]:.4f}" for algo in val_loss_dict.keys()])
            print(f"Epoch {epoch+1}/{num_epochs}, Total Validation Loss: {val_loss:.4f} | {loss_str}")
        else:
            val_loss = evaluate(model, val_dataloader, device)
            print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}")
        val_losses.append(val_loss)
        
        # Save model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save model weights
            torch.save(model.state_dict(), model_save_path)
            # Save configuration
            with open(config_save_path, 'w') as f:
                # Create a copy of the config with any command-line overrides
                updated_config = copy.deepcopy(config)
                if 'training' not in updated_config:
                    updated_config['training'] = {}
                updated_config['training']['learning_rate'] = learning_rate
                updated_config['training']['batch_size'] = batch_size
                updated_config['training']['num_epochs'] = num_epochs
                updated_config['training']['dataset'] = dataset_name
                updated_config['training']['algo'] = alg
                updated_config['training']['model_name'] = model_name
                json.dump(updated_config, f, indent=4)
            print(f"Saved best model to {model_save_path} and config to {config_save_path}")
            counter = 0  # Reset patience counter
        else:
            counter += 1  # Increment counter if no improvement
            
        # Early stopping check
        if joint_training:
            if scheduler['type'] == 'joint':
                pass
            elif scheduler['type'] == 'alternating':
                if epoch < sum(scheduler['schedule']):
                    counter = 0 # reset counter to make sure all algos are trained one round
            elif scheduler['type'] == 'sequential':
                pass
            elif scheduler['type'] == 'fine_tune':
                if epoch <= max(scheduler['schedule']):
                    counter = 0 # reset counter to make sure all algos are trained one round
        
        if counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs (no improvement for {patience} epochs)")
            break

    # Final evaluation
    if joint_training:
        train_loss, loss_dict = evaluate_joint(model, val_dataloader, device)
        loss_str = " | ".join([f"{algo}: {loss_dict[algo]:.4f}" for algo in loss_dict.keys()])
        print(f"Final Validation Loss: {train_loss:.4f} | {loss_str}")
    else:
        train_loss = evaluate(model, val_dataloader, device)
        print(f"Final Validation Loss: {train_loss:.4f}")

    train_dataset.close()  # Close HDF5 file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the InterpNetwork.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--hidden_dim", type=int, help="Hidden dimension of the model (overrides config)")
    parser.add_argument("--learning_rate", type=float, help="Learning rate (overrides config)")
    parser.add_argument("--batch_size", type=int, help="Batch size (overrides config)")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs (overrides config)")
    parser.add_argument("--dataset", type=str, choices=["all", "16", "8", "OOD"], help="Dataset to use (overrides config)")
    parser.add_argument("--model_name", type=str, help="Model name for saving (defaults to model_type)")
    parser.add_argument("-r", "--resume", action='store_true', help="Resume from previously trained weights")
    parser.add_argument("--algo", type=str, choices=["bellman_ford", "dijkstra", "mst_prim", "bellman_ford_bfs", "bfs"], help="Algorithm to train on (overrides config)")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], help="Device to use.")
    parser.add_argument("--sync", action='store_true', help="Use synchronous datasets.")
    args = parser.parse_args()
    main(args)