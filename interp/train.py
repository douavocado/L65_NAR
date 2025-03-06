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

from interp.dataset import HDF5Dataset, custom_collate
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
            loss = 0
            for i in range(len(class_out)):
                # print(batch_i, i)
                dist_ins = dist_out[i] # (T-1, D)
                class_ins = class_out[i].permute((0,2,1)) # (T-1, D, D) second dimension is the source node

                dist_l = F.mse_loss(dist_ins, upd_d[time_i[i]+1:time_i[i+1],batch_i[i]:batch_i[i+1]])  # Use MSE loss
                class_l = class_loss(class_ins, upd_pi[time_i[i]+1:time_i[i+1],batch_i[i]:batch_i[i+1]])
                loss += dist_l + class_l
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_dist_loss += dist_l.item() * inputs.size(0)
            running_class_loss += class_l.item() * inputs.size(0)

            total_samples += inputs.size(0)
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
            loss = 0
            for i in range(len(class_out)):
                dist_ins = dist_out[i] # (T-1, D)
                class_ins = class_out[i].permute((0,2,1)) # (T-1, D, D) second dimension is the source node

                dist_l = F.mse_loss(dist_ins, upd_d[time_i[i]+1:time_i[i+1],batch_i[i]:batch_i[i+1]])  # Use MSE loss
                class_l = class_loss(class_ins, upd_pi[time_i[i]+1:time_i[i+1],batch_i[i]:batch_i[i+1]])
                loss += dist_l + class_l

            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    avg_loss = running_loss / total_samples

    return avg_loss




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
        

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model name and paths
    checkpoint_dir = os.path.join("interp_checkpoints", alg, f"{model_name}_{dataset_name}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_save_path = os.path.join(checkpoint_dir, f"{model_name}_{dataset_name}.pth")
    config_save_path = os.path.join(checkpoint_dir, f"{model_name}_{dataset_name}_config.json")
    
    # Data loading
    data_root = os.path.join("data", alg)
    
    if dataset_name == "all":
        train_pth = os.path.join(data_root, "interp_data_all.h5")
    elif dataset_name == "16":
        train_pth = os.path.join(data_root, "interp_data_16.h5")
    elif dataset_name == "8":
        train_pth = os.path.join(data_root, "interp_data_8.h5")
    elif dataset_name == "OOD":
        train_pth = os.path.join(data_root, "interp_data_OOD_20_64.h5")
    else:
        raise ValueError(f"Dataset {dataset_name} not found")

    # Validation always on graphs of size 16
    val_pth = os.path.join(data_root, "interp_data_16_eval.h5")

    train_dataset = HDF5Dataset(train_pth)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    val_dataset = HDF5Dataset(val_pth)
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
        train_loss = train_one_epoch(model, train_dataloader, optimizer, device, epoch, num_epochs)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}")
        train_losses.append(train_loss)

        # Validation
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
        if counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs (no improvement for {patience} epochs)")
            break

    # Final evaluation
    train_loss = evaluate(model, val_dataloader, device)
    print(f"Final Training Loss: {train_loss:.4f}")

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
    parser.add_argument("--algo", type=str, choices=["bellman_ford", "dijkstra", "prims"], help="Algorithm to train on (overrides config)")
    args = parser.parse_args()
    main(args)