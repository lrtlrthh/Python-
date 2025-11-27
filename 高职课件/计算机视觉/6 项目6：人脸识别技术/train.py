import matplotlib.pyplot as plt
import numpy as np
import torch
import model
import dataset
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm



def train(name):
    train_loss = []
    val_loss = []
    val_acc = []
    min_val_acc = np.inf
    best_model = None
    for epoch in range(max_epoch):
        model.train()
        epoch_train_loss = []
        epoch_val_loss = []
        epoch_val_acc = []

        t = tqdm(train_dataloader)
        for x, y in t:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            y_ = model(x)
            loss = loss_func(y_, y)
            loss.backward()
            optimizer.step()
            t.set_postfix(train_loss=loss.item())
            epoch_train_loss.append(loss.item())

        model.eval()
        t = tqdm(val_dataloader)
        for x, y in t:
            x, y = x.to(device), y.to(device)
            y_ = model(x)
            loss = loss_func(y_, y)
            pred = torch.max(y_, dim=1)[1]
            acc = float(torch.eq(y.cpu(), pred.cpu()).sum() / len(x))
            t.set_postfix(val_loss=loss.item(), val_acc=acc)
            epoch_val_loss.append(loss.item())
            epoch_val_acc.append(acc)

        train_loss.append(np.mean(epoch_train_loss))
        val_loss.append(np.mean(epoch_val_loss))
        val_acc.append(np.mean(epoch_val_acc))
        print(f'{epoch + 1}|{max_epoch} train_loss={train_loss[-1]}')
        print(f'{epoch + 1}|{max_epoch} val_loss={val_loss[-1]}')
        print(f'{epoch + 1}|{max_epoch} val_acc={val_acc[-1]}')

        # 保存最好的模型
        if val_loss[-1] < min_val_acc:
            min_val_acc = val_acc[-1]
            best_model = model
            torch.save(best_model, f'/home/jovyan/work/results/best_{name}.pt')

    torch.save(model, f'/home/jovyan/work/results/last_{name}.pt')

    plt.figure()
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.plot(val_acc, label='val_acc')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    max_epoch = 50
    batch_size = 32
    lr = 1e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_path = '/home/jovyan/work/datasets/689457a87fedd2132f427a9e-momodel/data/train_data'
    model_name = 'resnet50'

    train_data = dataset.datasets(data_path=data_path, train=True)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    val_data = dataset.datasets(data_path=data_path, train=False)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    model = model.Resnet50().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    train(name=model_name)
