import torch


def snet_to_edge_index(snet, threshold):
    """
    :param snet: structural brain network
    :param threshold: edge threshold for graph construction from snets
    :return: edge indices in torch_geometric format
    """
    lst = [[i, j] for i in range(snet.shape[0]) for j in range(snet.shape[1]) if snet[i, j] > threshold and i != j]
    lst = torch.tensor(lst, dtype=torch.long)
    return torch.transpose(lst, 0, 1)
