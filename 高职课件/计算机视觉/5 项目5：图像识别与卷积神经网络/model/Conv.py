import torch.nn as nn
import torch


class ResFormer(nn.Module):
    def __init__(self):
        super(ResFormer, self).__init__()
        self.cnn1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.cnn2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.cnn3 = nn.Conv2d(128, 256, 3, 1, 1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.dump_sample = nn.MaxPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        self.fcn = nn.Sequential(nn.Flatten(), nn.Linear(256 * 28 * 28, 1024),
                                 nn.ReLU(inplace=True), nn.Linear(1024, 512),
                                 nn.ReLU(inplace=True), nn.Linear(512, 102))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # 使用xavier_uniform进行初始化 :
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dump_sample(x)
        x = self.cnn2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dump_sample(x)
        x = self.cnn3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dump_sample(x)
        x = self.fcn(x)
        return x


if __name__ == '__main__':
    model = ResFormer()
    X = torch.randn(1, 3, 224, 224)
    # summary(model, input_size=(3, 224, 224), batch_size=1)
    print(model(X).shape)
