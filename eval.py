import torch


def evaluate(model, data_loader, loss_function, device):
    """
    :param model: model for inference
    :param data_loader: data loader for inference
    :param loss_function: loss function
    :param device: torch.device() instance, refers to CPU or GPU
    :return: mean validation loss, mean validation accuracy
    """
    val_loss, val_acc = 0.0, 0.0
    model.eval()
    with torch.no_grad():
        for _, data in enumerate(data_loader):
            data = data.to(device)
            pred = model(data)
            gnd = data.y
            val_acc += float(torch.max(pred, 1)[1] == data.y)
            del data
            val_loss += float(loss_function(pred, gnd).item())
    return val_loss / len(data_loader), val_acc / len(data_loader)
