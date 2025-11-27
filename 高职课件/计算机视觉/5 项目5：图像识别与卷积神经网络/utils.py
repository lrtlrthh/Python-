import mindspore as ms
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import mindspore.nn as nn
import mindspore.ops as ops

from typing import List, Optional
from mindspore.common.initializer import Normal
from mindspore import dtype as mstype, load_checkpoint, load_param_into_net
from mindspore.dataset import ImageFolderDataset
class util:
    weight_init = Normal(mean=0, sigma=0.02)
    gamma_init = Normal(mean=1, sigma=0.02)
    def __init__(self,num_epochs,network,step_size_train):
        """初始化各参数"""
        self.num_epochs=num_epochs #迭代次数
        self.network=network       
        self.step_size_train=step_size_train
        self.lr=nn.cosine_decay_lr(min_lr=0.00001, max_lr=0.001, total_step=step_size_train * num_epochs,
                            step_per_epoch=step_size_train, decay_epoch=num_epochs)
        self.opt = nn.Momentum(params=network.trainable_params(), learning_rate=self.lr, momentum=0.9)
        self.loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        self.grad_fn = ms.ops.value_and_grad(self.forward_fn, None, self.opt.parameters)

    def forward_fn(self,inputs, targets):
        logits =  self.network(inputs)
        loss =  self.loss_fn(logits, targets)
        return loss
    def train_step(self,inputs, targets):
        loss, grads = self.grad_fn(inputs, targets)
        self.opt(grads)
        return loss
    def train(self,data_loader, epoch):
        """模型训练"""
        losses = []
        self.network.set_train(True)
        for i, (images, labels) in enumerate(data_loader):
            loss = self.train_step(images, labels)
            if i % 100 == 0 or i == self.step_size_train - 1:
                print('Epoch: [%3d/%3d], Steps: [%3d/%3d], Train Loss: [%5.3f]' %
                      (epoch + 1, self.num_epochs, i + 1, self.step_size_train, loss))
            losses.append(loss)
        return sum(losses) / len(losses)

    def evaluate(self,data_loader):
        """模型验证"""
        network=self.network
        network.set_train(False)

        correct_num = 0.0  # 预测正确个数
        total_num = 0.0  # 预测总数

        for images, labels in data_loader:
            logits = network(images)
            pred = logits.argmax(axis=1)  # 预测结果
            correct = ops.equal(pred, labels).reshape((-1,))
            correct_num += correct.sum().asnumpy()
            total_num += correct.shape[0]

        acc = correct_num / total_num  # 准确率

        return acc
    @staticmethod
    def resnet(block,layers: List[int], num_classes: int,input_channel: int):
        model = ResNet(block, layers, num_classes, input_channel)
        return model
    @staticmethod
    def create_dataset(dataset_dir, usage, resize, batch_size, workers, num_samples, shuffle=True, decode=True,
                       class_indexing=None, crop_size=32):
        data_set = ImageFolderDataset(
            dataset_dir=dataset_dir, shuffle=shuffle, decode=decode, num_samples=num_samples,
            num_parallel_workers=workers, class_indexing=class_indexing)
        trans = []
        if usage == "train":
            trans += [
                vision.RandomCrop((crop_size, crop_size), (4, 4, 4, 4)),
                vision.RandomHorizontalFlip(prob=0.5)
            ]
        trans += [
            vision.Resize(resize),
            vision.Rescale(1.0 / 255.0, 0.0),
            vision.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            vision.HWC2CHW()
        ]

        target_trans = transforms.TypeCast(mstype.int32)
        # 数据映射操作
        data_set = data_set.map(operations=trans,
                                input_columns='image',
                                num_parallel_workers=workers)

        data_set = data_set.map(operations=target_trans,
                                input_columns='label',
                                num_parallel_workers=workers)
        # 批量操作
        data_set = data_set.batch(batch_size)
        return data_set

    @staticmethod
    def make_layer(last_out_channel, block,
                   channel: int, block_nums: int, stride: int = 1):
        down_sample = None  # shortcuts分支

        if stride != 1 or last_out_channel != channel * block.expansion:
            down_sample = nn.SequentialCell([
                nn.Conv2d(last_out_channel, channel * block.expansion,
                          kernel_size=1, stride=stride, weight_init=util.weight_init),
                nn.BatchNorm2d(channel * block.expansion, gamma_init=util.gamma_init)
            ])
        layers = []
        layers.append(block(last_out_channel, channel, stride=stride, down_sample=down_sample))
        in_channel = channel * block.expansion
        # 堆叠残差网络
        for _ in range(1, block_nums):
            layers.append(block(in_channel, channel))
        return nn.SequentialCell(layers)

class ResidualBlock(nn.Cell):
    expansion = 4  # 最后一个卷积核的数量是第一个卷积核数量的4倍
    def __init__(self, in_channel: int, out_channel: int,
                 stride: int = 1, down_sample: Optional[nn.Cell] = None) -> None:
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel,
                               kernel_size=1, weight_init=util.weight_init)
        self.norm1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel,
                               kernel_size=3, stride=1,
                               weight_init=util.weight_init)
        self.norm2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(out_channel, out_channel,
                               kernel_size=3, stride=stride,
                               weight_init=util.weight_init)
        self.norm3 = nn.BatchNorm2d(out_channel)
        self.conv4 = nn.Conv2d(out_channel, out_channel * self.expansion,
                               kernel_size=1, weight_init=util.weight_init)
        self.norm4 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU()
        self.down_sample = down_sample
    def construct(self, x):
        identity = x  # shortscuts分支
        out = self.conv1(x)  # 主分支第一层：1*1卷积层
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)  # 主分支第二层：3*3卷积层
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv3(out)  # 主分支第三层：3*3卷积层
        out = self.norm3(out)
        out = self.relu(out)
        out = self.conv4(out)  # 主分支第四层：1*1卷积层
        out = self.norm4(out)
        if self.down_sample is not None:
            identity = self.down_sample(x)
        out += identity  # 输出为主分支与shortcuts之和
        out = self.relu(out)
        return out
class ResNet(nn.Cell):
    def __init__(self, block,
                 layer_nums: List[int], num_classes: int, input_channel: int) -> None:
        super(ResNet, self).__init__()
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, weight_init=util.weight_init)
        self.norm = nn.BatchNorm2d(64)

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')

        self.layer1 = util.make_layer(64, block, 64, layer_nums[0])
        self.layer2 = util.make_layer(64 * block.expansion, block, 128, layer_nums[1], stride=1)
        self.layer3 = util.make_layer(128 * block.expansion, block, 256, layer_nums[2], stride=2)
        self.layer4 = util.make_layer(256 * block.expansion, block, 512, layer_nums[3], stride=2)
        self.layer5 = util.make_layer(512 * block.expansion, block, 1024, layer_nums[4], stride=2)

        self.avg_pool = nn.AvgPool2d()
  
        self.flatten = nn.Flatten()
 
        self.fc = nn.Dense(in_channels=input_channel, out_channels=num_classes)
    def construct(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x