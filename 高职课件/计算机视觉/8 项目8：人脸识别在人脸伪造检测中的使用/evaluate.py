import os
import matplotlib.pyplot as plt
from mindspore import nn
from mindspore import Tensor
import numpy as np
import mindspore
import mindspore.dataset as ds
import mindspore.dataset.vision as CV
import mindspore.dataset.transforms as C
from mindspore import dtype as mstype

batch_size_test = 1  # 测试集划分为batch的大小
resolution = 224


path_train_set = "/home/jovyan/work/datasets/6892fb249e78856c69427a9c-momodel/faceDataset/train_set"
path_test_set = "/home/jovyan/work/datasets/6892fb249e78856c69427a9c-momodel/faceDataset/test_set"
Reset = False  # 是否重置模型参数
# PATH = '/home/jovyan/work/results/parameter_c.ckpt'  # 参数的目录
PATH = '/home/jovyan/work/datasets/6892fb249e78856c69427a9c-momodel/parameter_c.ckpt'  # 参数的目录

class SE(nn.Cell): # SE注意力模块
    def __init__(self, channel_in, channel_mid):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.liner1 = nn.SequentialCell(
            nn.Dense(channel_in, channel_mid),
            nn.HSwish()
        )
        self.liner2 = nn.SequentialCell(
            nn.Dense(channel_mid, channel_in),
            nn.Sigmoid()

        )

    def construct(self, x):
        result = self.pool(x)
        result = result.squeeze(-1)
        result = result.squeeze(-1)

        result = self.liner1(result)
        result = self.liner2(result)

        result = result.unsqueeze(2)
        result = result.unsqueeze(3)
        result = x * result
        return result


class MBConv(nn.Cell):
    def __init__(self, channel_in, times, channel_out, kernel_size, stride):
        super().__init__()

        self.channel_in = channel_in
        self.channel_out = channel_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.channel_mid = channel_in * times
        self.Conv1 = nn.SequentialCell(
            nn.Conv2d(channel_in, self.channel_mid, 1, 1),
            nn.BatchNorm2d(self.channel_mid),
            nn.HSwish()
        )
        self.DConv = nn.SequentialCell(
            nn.Conv2d(self.channel_mid, self.channel_mid, kernel_size, stride, 'same', 0, 1, self.channel_mid),
            nn.BatchNorm2d(self.channel_mid),
            nn.HSwish()
        )
        self.SE = SE(self.channel_mid, int(self.channel_in / 4))
        self.Conv2 = nn.SequentialCell(
            nn.Conv2d(self.channel_mid, self.channel_out, 1, 1),
            nn.BatchNorm2d(self.channel_out)
        )
        self.Drop = nn.Dropout(p=0.25)

    def construct(self, x):
        output = x
        if self.channel_in != self.channel_mid:
            output = self.Conv1(x)
        output = self.DConv(output)
        output = self.SE(output)
        output = self.Conv2(output)
        output = self.Drop(output)
        if output.shape == x.shape:
            output = output + x
        return output


class Network(nn.Cell):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, 2)

        self.mbc1 = MBConv(32, 1, 16, 3, 1)
        self.mbc2 = MBConv(16, 6, 24, 3, 2)
        self.mbc3 = MBConv(24, 6, 40, 5, 2)
        self.mbc4 = MBConv(40, 6, 80, 3, 2)

        self.pool = nn.AvgPool2d(2)

        self.liner = nn.SequentialCell(
            nn.Dense(13520, 2048),
            nn.LeakyReLU(),
            nn.Dense(2048, 512),
            nn.LeakyReLU(),
            nn.Dense(512, 32),
            nn.LeakyReLU(),
            nn.Dense(32, 2),
        )

    def construct(self, x, batch_size):
        x = self.conv1(x)

        result = self.mbc1(x)
        result = self.mbc2(result)
        result = self.mbc3(result)
        result = self.mbc4(result)
        result = self.pool(result)
        result = result.reshape((batch_size, -1))
        result = self.liner(result)
        return result


def create_dataset(path):  # 创建数据集并返回总图像数
    dataset = ds.ImageFolderDataset(path, shuffle=True)
    num_imgs = dataset.get_dataset_size()
    print("数据集大小为：" + str(num_imgs))
    return dataset


def dataset_transform(dataset, side_length=resolution):
    image_size = [side_length, side_length]
    mean = [127.5, 127.5, 127.5]
    std = [255., 255., 255.]
    trans = [
        CV.Decode(),
        CV.Resize(image_size),
        CV.Normalize(mean=mean, std=std),
        CV.HWC2CHW()
    ]
    type_cast_op = C.TypeCast(mstype.int32)
    # 实现数据的map映射、批量处理和数据重复的操作
    dataset = dataset.map(operations=trans, input_columns="image", num_parallel_workers=8)
    dataset = dataset.map(operations=type_cast_op, input_columns="label", num_parallel_workers=8)
    print("数据集一共分为" + str(dataset.num_classes()) + "列，分别为：" + str(dataset.get_col_names()))
    print("标签与类别的对应关系为：" + str(dataset.get_class_indexing()))
    return dataset


def create_batch(dataset, batch_size, repeat=1):
    batch_set = dataset.batch(batch_size, drop_remainder=True)
    num_bt = batch_set.get_dataset_size()
    batch_set = batch_set.repeat(repeat)
    print("数据集划分为的batch数为：" + str(num_bt))
    return batch_set


test_set = create_dataset(path_test_set)
test_set = dataset_transform(test_set)
test_set = create_batch(test_set, batch_size_test, 1)
network = Network()


if os.path.exists(PATH) and not Reset:
    param_dict = mindspore.load_checkpoint(PATH)
    mindspore.load_param_into_net(network, param_dict)
    print("模型参数加载完毕")


def test_evaluate():  # 评估模型的准确率
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for batch, (data, label) in enumerate(test_set.create_tuple_iterator()):
        predict = network(data, batch_size_test)
        predict = predict.argmax(1)
        for i in range(len(label)):
            if predict[i] == 0: # 0对应fake对应阳性
                if int(label[i]) == 0:
                    TP += 1
                else:
                    FP += 1
            elif predict[i] == 1:
                if int(label[i]) == 1:
                    TN += 1
                else:
                    FN += 1
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)  # TPR
    FPR = FP / (TN + FP)
    print("the accuracy rate is: %.1f %%" % (accuracy * 100))
    print("the precision rate is: %.1f %%" % (precision * 100))
    print("the recall rate(TPR) is: %.1f %%" % (recall * 100))
    print("the FPR is: %.1f %%" % (FPR * 100))


test_evaluate()