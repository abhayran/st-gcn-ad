import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops


class Layer(MessagePassing):
    def __init__(self, kernel_size, stride=1, padding=0, aggr='max'):
        """
        :param kernel_size: length for the convolution kernel
        :param stride: stride for the convolution kernel
        :param padding: padding for the convolution kernel, applied to both ends
        :param aggr: aggregation scheme for message passing, options: 'max', 'mean', 'sum'
        """
        super(Layer, self).__init__(aggr=aggr)
        self.message_filter = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.update_filter = torch.nn.Conv1d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x, edge_index):
        """
        :param x: input activations
        :param edge_index: graph edges
        :return: next layer activations
        """
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        """
        :param x_j: source node activations, shape: (x.shape[0], edge_index.shape[1])
        :return: x_j: updated source node activations
        """
        x_j = x_j.unsqueeze(dim=1)
        x_j = self.message_filter(x_j)
        x_j = x_j.squeeze(dim=1)
        return x_j

    def update(self, aggr_out, x):
        """
        :param aggr_out: aggregations for each target node
        :param x: input activations to the layer
        :return: next layer activations
        """
        assert aggr_out.shape[1] == x.shape[1], f'Shape mismatch: aggr_out -> {aggr_out.shape}, x -> {x.shape}'
        aggr_out = aggr_out.unsqueeze(dim=1)
        x = x.unsqueeze(dim=1)
        next_layer_embeddings = torch.cat([x, aggr_out], dim=1)
        next_layer_embeddings = self.update_filter(next_layer_embeddings)
        next_layer_embeddings = next_layer_embeddings.squeeze(dim=1)
        return next_layer_embeddings


class Model(torch.nn.Module):
    def __init__(self, input_shape, kernel_sizes, strides, paddings, aggr='max'):
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
        # edge_index, _ = remove_self_loops(edge_index)
        # edge_index, _ = add_self_loops(edge_index)
        for _, layer in enumerate(self.conv_layers):
            x = layer(x, edge_index)
            x = F.relu(x)
        x, _ = torch.max(x, dim=1)
        x = x.view(-1, self.vertices)
        x = self.output_layer(x)  # classification layer
        return x
