import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops


class Layer(MessagePassing):
    def __init__(self, kernel_size, stride=1, padding=0, aggr='mean'):
        """
        :param kernel_size: length for the convolution kernel
        :param stride: stride for the convolution kernel
        :param padding: padding for the convolution kernel, applied to both ends
        :param aggr: aggregation scheme for message passing, options: 'max', 'mean', 'sum'
        """
        super(Layer, self).__init__(aggr=aggr)
        self.filter = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x, edge_index):
        """
        :param x: input activations
        :param edge_index: graph edges
        :return: next layer activations
        """
        x = x.unsqueeze(dim=1)
        x = self.filter(x)
        x = x.squeeze(dim=1)
        return self.propagate(edge_index, x=x)


class Model(torch.nn.Module):
    def __init__(self, input_shape, kernel_sizes, strides, paddings, aggr='mean'):
        """
        :param input_shape: tuple: (number of vertices, temporal depth)
        :param kernel_sizes: list of kernel sizes for Conv1d filters for each layer
        :param strides: list of strides for Conv1d filters for each layer
        :param paddings: list of paddings for Conv1d filters for each layer
        :param aggr: neighbor aggregation scheme
        """
        super(Model, self).__init__()
        self.vertices, self.input_dim = input_shape
        self.conv_layers = torch.nn.ModuleList([Layer(kernel_sizes[i], stride=strides[i], padding=paddings[i], aggr=aggr) for i in range(len(kernel_sizes))])
        self.output_layer = torch.nn.Linear(in_features=self.vertices, out_features=3)

    def forward(self, data):
        """
        :param data: input sample
        :return: output activations
        """
        x, edge_index = data.x, data.edge_index
        edge_index, _ = add_self_loops(edge_index)
        for _, layer in enumerate(self.conv_layers):
            x = layer(x, edge_index)
            x = F.relu(x)
        x = x.view(-1, self.vertices)
        x = self.output_layer(x)  # classification layer
        return x
