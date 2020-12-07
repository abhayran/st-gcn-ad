import torch
from torch_geometric.data import Dataset
import os


class GraphDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.data_list = [torch.load(os.path.join(path, file)) for file in os.listdir(path)]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
