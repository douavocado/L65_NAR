import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

import h5py
import numpy as np
from tqdm import tqdm

from interp.models import InterpNetwork
from interp.dataset import HDF5Dataset, custom_collate

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
                # print(class_ins.shape, upd_pi[time_i[i]:time_i[i+1]-1,batch_i[i]:batch_i[i+1]].shape)

                dist_l = F.mse_loss(dist_ins, upd_d[time_i[i]:time_i[i+1]-1,batch_i[i]:batch_i[i+1]])  # Use MSE loss
                class_l = class_loss(class_ins, upd_pi[time_i[i]:time_i[i+1]-1,batch_i[i]:batch_i[i+1]])
                loss += dist_l + class_l
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            pbar.update(1)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

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

                dist_l = F.mse_loss(dist_ins, upd_d[time_i[i]:time_i[i+1]-1,batch_i[i]:batch_i[i+1]])  # Use MSE loss
                class_l = class_loss(class_ins, upd_pi[time_i[i]:time_i[i+1]-1,batch_i[i]:batch_i[i+1]])
                loss += dist_l + class_l

            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    avg_loss = running_loss / total_samples

    return avg_loss




def main(args):
    # --- Hyperparameters and Setup ---
    hidden_dim = args.hidden_dim
    approach = args.approach
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model_save_path = args.model_save_path
    pretrained_model_path = args.pretrained_model_path
    resume = args.resume

    # --- Data Loading ---
    train_dataset = HDF5Dataset("data/interp_data_all_lengths.h5")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    val_dataset = HDF5Dataset("data/interp_data_all_lengths_eval.h5")
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

     #get num_nodes from the dataset:
    first_batch = next(iter(train_dataloader))
    # num_nodes = first_batch['hidden_states'].shape[-1] #Get D
    #print("Number of Nodes (D):",num_nodes)


    # --- Model, Optimizer, Loss ---
    model = InterpNetwork(hidden_dim=hidden_dim).to(device)

    # Load pretrained weights if provided
    if resume:
        model.load_state_dict(torch.load(pretrained_model_path))
        print(f"Loaded pretrained weights from {pretrained_model_path}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- Training Loop ---
    train_losses = []
    val_losses = []  # Placeholder - Add a validation dataset if you have one.
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_dataloader, optimizer, device, epoch, num_epochs)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}")
        train_losses.append(train_loss)

        # --- Validation (optional, but highly recommended) ---
        val_loss = evaluate(model, val_dataloader, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}")
        val_losses.append(val_loss)
        if val_loss <= min(val_losses):
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model to {model_save_path}")

    # --- Evaluation (on the training set, for demonstration) ---
    # Ideally, you should have a separate test set.
    train_loss = evaluate(model, val_dataloader, device)
    print(f"Final Training Loss: {train_loss:.4f}")

    # # --- Save Model ---
    # torch.save(model.state_dict(), model_save_path)
    # print(f"Model saved to {model_save_path}")


    train_dataset.close()  # Close HDF5 file

    # --- (Optional) Further Analysis ---
    # You can analyze train_predictions and train_targets here.
    # For example, calculate accuracy, visualize predictions, etc.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the InterpNetwork.")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension of the model.")
    parser.add_argument("--approach", type=str, default="diff_mlp",
                        choices=["simple_diff", "mlp_diff", "transformer", "masked_transformer"],
                        help="Approach for temporal processing.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--num_epochs", type=int, default=1000, help="Number of epochs.")
    parser.add_argument("--model_save_path", type=str, default="interp_checkpoints/trained_model.pth",
                        help="Path to save the trained model.")
    parser.add_argument("--pretrained_model_path", type=str, default="interp_checkpoints/trained_model.pth",
                        help="Path to a pretrained model to load (optional).")
    parser.add_argument("-r", "--resume", action='store_true',
                        help="Resume from previously trained weights taken from --pretrained_model_path")
    

    args = parser.parse_args()
    main(args)