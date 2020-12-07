import torch
from torch_geometric.data import DataLoader
from model import SAGE
from data import GraphDataset
from eval import evaluate
import json
import matplotlib.pyplot as plt


def train(config):
    """
    :param config: dict for the training parameters and file paths
    :return: model, train_loss_list, val_loss_list: trained model, lists of training and validation losses
    """
    torch.manual_seed(config['seed'])
    if config['gpu']:
        torch.cuda.manual_seed_all(config['seed'])

    shuffle = config['shuffle']
    epochs = config['epochs']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']

    input_shape = config['input_shape']
    kernel_sizes = config['kernel_sizes']
    strides = config['strides']
    paddings = config['paddings']
    message_channels = config['message_channels']
    graph_norm = config['graph_norm']
    aggr = config['aggr']

    model = SAGE(input_shape, kernel_sizes, strides, paddings, message_channels, graph_norm=graph_norm, aggr=aggr)

    data_train = GraphDataset(config['paths']['train'])
    data_loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=shuffle)
    data_val = GraphDataset(config['paths']['val'])
    data_loader_val = DataLoader(data_val, batch_size=1, shuffle=False)

    optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.CrossEntropyLoss(reduction='sum')

    train_loss_list, val_loss_list = [], []
    for _ in range(epochs):
        model.train()
        train_loss = 0.0
        for _, data in enumerate(data_loader_train):
            optimizer.zero_grad()
            loss = loss_function(model(data), data.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss_list.append(train_loss)
        val_loss_list.append(evaluate(model, data_loader_val))
    return model, train_loss_list, val_loss_list


if __name__ == '__main__':
    with open('config.json') as f:
        configuration = json.load(f)
    network, train_losses, val_losses = train(configuration)
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.title('Model losses')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()
