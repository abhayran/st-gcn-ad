import torch
import torch_geometric as tg


class SAGE_layer(tg.nn.MessagePassing):  # custom GraphSAGE layer
    def __init__(self, kernel_size, stride=1, padding=0, aggr='mean'):
        """
        :param kernel_size: kernel size for the 1D convolution
        :param stride: stride for the 1D convolution kernel
        :param padding: padding for the 1d convolution
        :param aggr: neighbor aggregation scheme
        """
        super().__init__(aggr=aggr)
        self.filter = torch.nn.Conv1d(in_channels=1, out_channels=10, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x, edge_index):
        return self.propagate(x=x, edge_index=edge_index)

    def message(self, x_j):
        """
        :param x_j: previous layer feature map for the jth vertex
        :return: message for the aggregator
        """
        x_j = self.filter(x_j)  # pass through the Conv1d layer
        x_j = torch.max(x_j, dim=0)  # channel-wise max-pooling
        return x_j

    def update(self, x, aggr_out):
        """
        :param x: previous layer feature maps
        :param aggr_out: neighbor aggregation
        :return: next layer feature maps
        """
        return 0.5 * (x + aggr_out)


class SAGE(torch.nn.Module):
    def __init__(self, input_shape, kernel_sizes, strides, paddings, graph_norm=True, aggr='mean'):
        """
        :param input_shape: tuple: (number of vertices, temporal depth)
        :param kernel_sizes: list of kernel sizes for Conv1d filters
        :param strides: list of strides for Conv1d filters
        :param paddings: list of paddings for Conv1d filters
        :param graph_norm: bool: whether to apply GraphNorm
        :param aggr: neighbor aggregation scheme
        """
        super().__init__()
        self.vertices, self.input_dim = input_shape
        self.conv_layers = torch.nn.ModuleList([SAGE_layer(kernel_sizes[i], stride=strides[i], padding=paddings[i], aggr=aggr) for i in range(len(kernel_sizes))])
        self.output_layer = torch.nn.Linear(in_features=self.vertices, out_features=3)
        self.graph_norm = graph_norm
        if self.graph_norm:
            dims = [self.input_dim]  # will hold temporal depths for each layer
            for i in range(len(kernel_sizes)):
                dims.append(1 + (dims[-1] + 2*paddings[i] - kernel_sizes[i]) // strides[i])
            self.alpha_hidden = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(dim, dtype=torch.float)) for dim in dims])
            self.scale_hidden = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(dim, dtype=torch.float)) for dim in dims])
            self.shift_hidden = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(dim, dtype=torch.float)) for dim in dims])

    def forward(self, data):
        """
        :param data: input sample
        :return: probabilities for classification
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
            x = torch.nn.ReLU(x)
        x = torch.sum(x, dim=1)  # sum across the temporal depth
        x = self.output_layer(x)  # classification layer
        return x
