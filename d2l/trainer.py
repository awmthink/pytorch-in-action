import torch
from . import d2l


def cls_accuracy(y_hat, y):
    return (y == torch.argmax(y_hat, dim=1)).sum().item()


def nn_epoch(data_iter, model, loss, optimizer, device):
    if optimizer is None:
        model.eval()
    else:
        model.train()
    epoch_loss = 0
    epoch_accuray = 0
    num_samples = 0
    for X, y in data_iter:
        X, y = X.to(device), y.to(device)
        y_hat = model(X)
        l = loss(y_hat, y)
        epoch_loss += l.item() * X.shape[0]
        epoch_accuray += cls_accuracy(y_hat, y)
        num_samples += X.shape[0]
        if optimizer is None:
            continue
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    return epoch_loss / num_samples, epoch_accuray / num_samples


def nn_train(
    training_iter, testing_iter, model, loss, optimizer, devices, epoch=20, plot=True
):
    animator = d2l.Animator(xlabel="epoch", ylabel="loss", xlim=[1, epoch])
    device = devices
    if isinstance(devices, (tuple, list)):
        device = devices[0]
    for i in range(epoch):
        epoch_train_loss, epoch_train_acc = nn_epoch(
            training_iter, model, loss, optimizer, device
        )
        _, epoch_test_acc = nn_epoch(testing_iter, model, loss, None, device)
        animator.add(
            i + 1,
            (epoch_train_loss, epoch_train_acc, epoch_test_acc),
        )
    print(
        f"epoch {i + 1}: train_loss: {epoch_train_loss:.4f}, train_acc: {epoch_train_acc: .4f}, test_acc: {epoch_test_acc: .4f}"
    )
