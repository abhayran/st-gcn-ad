import torch


def snet_to_edge_index(snet, threshold):
    lst = [[i, j] for i in range(snet.shape[0]) for j in range(snet.shape[1]) if snet[i, j] > threshold]
    lst = torch.tensor(lst)
    return torch.transpose(lst, 0, 1)
