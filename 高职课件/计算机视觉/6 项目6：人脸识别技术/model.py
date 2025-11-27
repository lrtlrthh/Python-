import torch
from torchvision import models
import torch.nn as nn
import torchsummary


def Resnet101():
    model = model = models.resnet101()
    model.fc = nn.Sequential(nn.Linear(2048, 1000),
                             nn.ReLU(),
                             nn.Linear(1000, 512),
                             nn.ReLU(),
                             nn.Linear(512, 2))
    return model


def Resnet50():
    model = model = models.resnet50()
    model.fc = nn.Sequential(nn.Linear(2048, 1000),
                             nn.ReLU(),
                             nn.Linear(1000, 512),
                             nn.ReLU(),
                             nn.Linear(512, 2))
    return model


def Resnet34():
    model = model = models.resnet34()
    model.fc = nn.Sequential(nn.Linear(512, 2))
    return model


def Resnet18():
    model = model = models.resnet18()
    model.fc = nn.Sequential(nn.Linear(512, 2))
    return model


if __name__ == '__main__':
    x = torch.randn(4, 3, 224, 224)
    model = Resnet18()
    print(model(x).shape)
    torchsummary.summary(model, input_size=(3, 224, 224), batch_size=4, device='cpu')
