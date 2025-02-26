import h5py
import numpy as np
from torch.utils.data import Dataset
import torch

def custom_collate(batch):
    """
    Creates a single sparse graph as the union of all graphs in the batch.
    Each graph's nodes and edges are preserved as disconnected components.
    """
    batch_size = len(batch)
    
    # Calculate cumulative nodes for creating batch indices
    nodes_per_graph = [item['graph_adj'].shape[0] for item in batch]
    cumsum_nodes = np.cumsum([0] + nodes_per_graph[:-1])
    all_cumsum = np.cumsum([0] + nodes_per_graph)
    total_nodes = sum(nodes_per_graph)
    
    # Create batch indices tensor
    batch_idx = torch.zeros(total_nodes, dtype=torch.long)
    for i, offset in enumerate(cumsum_nodes):
        batch_idx[offset:offset + nodes_per_graph[i]] = i

    batched = {
        'batch': batch_idx,
        'num_graphs': torch.tensor(batch_size),
        'num_nodes_per_graph': torch.tensor(nodes_per_graph),
        'all_cumsum': torch.tensor(all_cumsum),
    }
    
    # Handle each type of feature
    for key in batch[0].keys():
        if key == 'graph_adj':
            # Create block diagonal adjacency matrix
            adj = torch.zeros(total_nodes, total_nodes)
            for i, item in enumerate(batch):
                n = item[key].shape[0]
                start_idx = cumsum_nodes[i]
                adj[start_idx:start_idx + n, start_idx:start_idx + n] = item[key]
            batched[key] = adj
            
        elif key == 'edge_weights':
            # Create block diagonal weight matrix
            weights = torch.zeros(total_nodes, total_nodes)
            for i, item in enumerate(batch):
                n = item[key].shape[0]
                start_idx = cumsum_nodes[i]
                weights[start_idx:start_idx + n, start_idx:start_idx + n] = item[key]
            batched[key] = weights
            
        elif key == 'hidden_states':
            # Calculate total timesteps and create block diagonal hidden states
            timesteps_per_graph = [item[key].shape[0] for item in batch]
            cumsum_timesteps = np.cumsum([0] + timesteps_per_graph[:-1])
            all_cumsum_timesteps = np.cumsum([0] + timesteps_per_graph)
            total_timesteps = sum(timesteps_per_graph)
            h_dim = batch[0][key].shape[1]
            
            hidden = torch.zeros(total_timesteps, h_dim, total_nodes)
            for i, item in enumerate(batch):
                t, _, n = item[key].shape
                start_t = cumsum_timesteps[i]
                start_n = cumsum_nodes[i]
                hidden[start_t:start_t + t, :, start_n:start_n + n] = item[key]
            batched[key] = hidden
            batched['timesteps_per_graph'] = torch.tensor(timesteps_per_graph)
            # batched['cumsum_timesteps'] = torch.tensor(cumsum_timesteps)
            batched['all_cumsum_timesteps'] = torch.tensor(all_cumsum_timesteps)
        elif key in ['upd_pi', 'upd_d']:
            # Create block diagonal updates matching hidden states structure
            timesteps_per_graph = [item[key].shape[0] for item in batch]
            cumsum_timesteps = np.cumsum([0] + timesteps_per_graph[:-1])
            total_timesteps = sum(timesteps_per_graph)
            
            updates = torch.zeros(total_timesteps, total_nodes)
            for i, item in enumerate(batch):
                t, n = item[key].shape
                start_t = cumsum_timesteps[i]
                start_n = cumsum_nodes[i]
                updates[start_t:start_t + t, start_n:start_n + n] = item[key]
            batched[key] = updates
            
        elif key in ['gt_pi', 'start_node']:
            # Concatenate node-level features (N) -> (total_N)
            features = torch.cat([item[key] for item in batch])
            batched[key] = features
    
    return batched

class HDF5Dataset(Dataset):
    def __init__(self, filename):
        self.filename = filename
        self.file = None  # Open the file lazily
        with h5py.File(self.filename, 'r') as f:  # Open temp to get len
            self.length = len(f.keys())

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.filename, 'r')  # Open on first access
        group = self.file[f'datapoint_{idx}']
        datapoint = {}
        for key in group.keys():
            datapoint[key] = torch.from_numpy(np.array(group[key]))  # Convert to tensor

        return datapoint

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None