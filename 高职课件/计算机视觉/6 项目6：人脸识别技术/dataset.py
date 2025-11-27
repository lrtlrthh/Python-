import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


class datasets(Dataset):
    def __init__(self, data_path='', transform=None, train=True):
        super(datasets, self).__init__()
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(224, ),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5), ])
        else:
            self.transform = transform
        class_path = os.listdir(data_path)
        self.data = []
        self.labels = []
        for idx, classes in enumerate(class_path):
            images = os.listdir(data_path + '/' + classes)
            for image in images:
                self.data.append(data_path + '/' + classes + '/' + image)
                self.labels.append(idx)

        # 打乱数据
        state = np.random.get_state()
        np.random.shuffle(self.data)
        np.random.set_state(state)
        np.random.shuffle(self.labels)

        # 划分训练验证集
        if train:
            self.data = self.data[:int(len(self.data)*0.7)]
            self.labels = self.labels[:int(len(self.labels)*0.7)]
        else:
            self.data = self.data[int(len(self.data) * 0.7):]
            self.labels = self.labels[int(len(self.labels) * 0.7):]

    def __getitem__(self, item):
        image = Image.open(self.data[item])
        # print(image)
        image_mat = self.transform(image)
        return image_mat, self.labels[item]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    data = datasets('data/train_data')
    # print(len(data))
    print(data[0])
    # dataloader = DataLoader(data, batch_size=4, shuffle=True)
    # for x, y in dataloader:
    #     print(x.shape, y)
