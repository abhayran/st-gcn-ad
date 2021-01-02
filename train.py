import torch
from torch_geometric.data import DataLoader
from model import Model
from data import GraphDataset
from eval import evaluate
import json
import matplotlib.pyplot as plt
import os


def train(config):
    """
    :param config: dict for the training parameters and file paths
    :return: history: dict containing trained model, lists of training and validation losses & accuracies
    """
    if config['gpu']:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    # torch.set_deterministic(True)

    snet_threshold = config['params']['snet_threshold']
    shuffle = config['params']['shuffle']
    batch_size = config['params']['batch_size']
    epochs = config['params']['epochs']
    learning_rate = config['params']['learning_rate']

    input_shape = config['model']['input_shape']
    kernel_sizes = config['model']['kernel_sizes']
    strides = config['model']['strides']
    paddings = config['model']['paddings']
    aggr = config['model']['aggr']

    model = Model(input_shape, kernel_sizes, strides, paddings, aggr=aggr)
    model = model.float()
    model = model.to(device)

    data_train = GraphDataset('./dataset', config['file_names']['train'], snet_threshold, device)
    data_loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=shuffle)
    data_val = GraphDataset('./dataset', config['file_names']['val'], snet_threshold, device)
    data_loader_val = DataLoader(data_val, batch_size=1, shuffle=False)

    optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.CrossEntropyLoss(reduction='sum')

    train_loss_list, train_acc_list, val_loss_list, val_acc_list = [], [], [], []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        for _, data in enumerate(data_loader_train):
            data = data.to(device)
            optimizer.zero_grad()
            pred = model(data)
            gnd = data.y
            train_acc += float(torch.sum(torch.tensor(torch.max(pred.clone().detach(), 1)[1] == gnd.clone().detach())))
            loss = loss_function(pred, gnd)
            del data
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
        train_loss_list.append(train_loss / len(data_train))
        train_acc_list.append(train_acc / len(data_train))

        val_loss, val_acc = evaluate(model, data_loader_val, loss_function, device)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        print(f'epoch: {epoch + 1} | '
              f'train loss: {train_loss_list[-1]:.4f} | '
              f'train acc: {train_acc_list[-1]:.4f} | '
              f'val loss: {val_loss_list[-1]:.4f} | '
              f'val acc: {val_acc_list[-1]:.4f}')

    return {'model': model,
            'train_loss': train_loss_list,
            'train_acc': train_acc_list,
            'val_loss': val_loss_list,
            'val_acc': val_acc_list}


if __name__ == '__main__':
    with open('config.json') as f:
        configuration = json.load(f)

    path = './dataset'
    sci, mci, ad = [], [], []
    for name in os.listdir(path):
        if name.startswith('sci'):
            sci.append(name)
        elif name.startswith('mci'):
            mci.append(name)
        elif name.startswith('ad'):
            ad.append(name)
        else:
            print(f"Passing for the file '{name}'")
            pass

    configuration['file_names']['train'] = sci[:int((2/3) * len(sci))] + mci[:int((2/3) * len(mci))] + ad[:int((2/3) * len(ad))]
    configuration['file_names']['val'] = sci[int((2/3) * len(sci)):] + mci[int((2/3) * len(mci)):] + ad[int((2/3) * len(ad)):]
    assert set(configuration['file_names']['train']).isdisjoint(configuration['file_names']['val']), 'Training and validation sets have common elements!'

    history = train(configuration)
    plt.plot(history['train_loss'])
    plt.plot(history['val_loss'])
    plt.title('Model losses')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()
