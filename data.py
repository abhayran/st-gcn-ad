from torch_geometric.data import Dataset


class GraphDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
