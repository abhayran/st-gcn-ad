import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class SAGE_layer(MessagePassing):
    def __init__(self, kernel_size, stride=1, padding=0, message_channel=10, aggr='max'):
        """
        :param kernel_size: length for convolution kernels
        :param stride: stride for convolution kernels
        :param padding: padding for convolution kernels, applied to both ends
        :param message_channel: number of convolution kernels
        :param aggr: aggregation scheme for message passing, options: 'max', 'mean', 'sum'
        """
        super(SAGE_layer, self).__init__(aggr=aggr)
        self.filter = torch.nn.Conv1d(in_channels=1, out_channels=message_channel, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x, edge_index):
        """
        :param x: input activations
        :param edge_index: graph edges
        :return: next layer activations
        """
        x = x.unsqueeze(dim=1)
        x = self.filter(x)
        x = torch.max(x, dim=1)
        return self.propagate(edge_index, x=x)


class SAGE(torch.nn.Module):
    def __init__(self, input_shape, kernel_sizes, strides, paddings, message_channels, graph_norm=True, aggr='max'):
        """
        :param input_shape: tuple: (number of vertices, temporal depth)
        :param kernel_sizes: list of kernel sizes for Conv1d filters for each layer
        :param strides: list of strides for Conv1d filters for each layer
        :param paddings: list of paddings for Conv1d filters for each layer
        :param message_channels: output channels for the filter that constructs the message for each layer
        :param graph_norm: bool: whether to apply GraphNorm
        :param aggr: neighbor aggregation scheme
        """
        super(SAGE, self).__init__()
        self.vertices, self.input_dim = input_shape
        self.conv_layers = torch.nn.ModuleList([SAGE_layer(kernel_sizes[i], stride=strides[i], padding=paddings[i], message_channel=message_channels[i], aggr=aggr) for i in range(len(kernel_sizes))])
        self.output_layer = torch.nn.Linear(in_features=self.vertices, out_features=3)
        self.graph_norm = graph_norm
        if self.graph_norm:
            dims = [self.input_dim]  # will hold temporal depths for each layer
            for i in range(len(kernel_sizes) - 1):
                dims.append(1 + (dims[-1] + 2*paddings[i] - kernel_sizes[i]) // strides[i])
            self.alpha_hidden = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(dim, dtype=torch.float)) for dim in dims])
            self.scale_hidden = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(dim, dtype=torch.float)) for dim in dims])
            self.shift_hidden = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(dim, dtype=torch.float)) for dim in dims])

    def forward(self, data):
        """
        :param data: input sample
        :return: output activations
        """
        x, edge_index = data.x, data.edge_index
        sqrt_n = torch.sqrt(torch.tensor(self.vertices, dtype=torch.float)).item()
        for idx, layer in enumerate(self.conv_layers):
            if self.graph_norm:
                x = x - self.alpha_hidden[idx] * torch.mean(x, dim=0)
                x = x / (torch.norm(x, dim=0) / sqrt_n)
                x = x * self.scale_hidden[idx]
                x = x + self.shift_hidden[idx]
            x = layer(x, edge_index)
            x = F.relu(x)
        x = torch.sum(x, dim=1)  # sum across the temporal depth
        x = self.output_layer(x)  # classification layer
        return x
