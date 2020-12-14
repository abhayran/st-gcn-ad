import torch
from torch_geometric.data import Dataset
from utils import snet_to_edge_index


class GraphDataset(Dataset):
    def __init__(self, path, names, threshold):
        super().__init__()
        self.data_list = []
        for name in names:
            data = torch.load(f'{path}/{name}')
            data.edge_index = snet_to_edge_index(data.snet, threshold)
            self.data_list.append(data)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
