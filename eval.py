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
    for _, data in enumerate(data_loader):
        pred = model(data).unsqueeze(dim=0)
        gnd = torch.tensor(torch.where(data.y == 1)[0].item(), dtype=torch.long).unsqueeze(dim=0).to(device)
        val_acc += float(torch.max(pred.squeeze(), 0)[1] == torch.max(data.y, 0)[1])
        del data
        val_loss += float(loss_function(pred, gnd))
    return val_loss / len(data_loader), val_acc / len(data_loader)
