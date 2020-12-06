from torch_geometric.data import Dataset


class GraphDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
