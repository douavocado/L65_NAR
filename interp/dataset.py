import h5py
import numpy as np
from torch.utils.data import Dataset
import torch

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