import torch
from torch_geometric.data import Data, Dataset
from utils import snet_to_edge_index


class GraphDataset(Dataset):
    def __init__(self, path, names, threshold, device):
        """
        :param path: system path to the data
        :param names: file names to be included in the dataset
        :param threshold: edge threshold for graph construction from snets
        :param device: torch.device() instance, refers to CPU or GPU
        """
        super().__init__()
        self.data_list = []
        for name in names:
            data = torch.load(f'{path}/{name}')
            sample = Data(x=data.x, edge_index=snet_to_edge_index(data.snet, threshold), y=data.y)
            sample = sample.to(device)
            self.data_list.append(sample)

    def __len__(self):
        """
        :return: length of the dataset
        """
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        :param idx: index of the sample to get
        :return: sample at the location <idx> in <self.data_list>
        """
        return self.data_list[idx]
