###############################################################################
# 重要: 请务必把任务(jobs)中需要保存的文件存放在 results 文件夹内
# Important : Please make sure your files are saved to the 'results' folder
# in your jobs
###############################################################################
import mindspore
import mindspore.nn as nn
from mindspore.common.initializer import Normal
from mindspore import context, save_checkpoint, ops, Tensor
import mindspore.dataset as ds
import mindspore.dataset.vision as CV
import mindspore.dataset.transforms as C
from mindspore import dtype as mstype
from PIL import Image
import random
import numpy as np
import os
from pathlib import Path

seed = 1234
random.seed(seed)
np.random.seed(seed)
mindspore.set_seed(seed)

class cnn_net(nn.Cell):
    """
    网络结构
    """
    def __init__(self, num_class=4, num_channel=3):
        super(cnn_net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_channel, out_channels=8, kernel_size=3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 64, 3)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(2304, 128, weight_init=Normal(0.02))
        self.fc2 = nn.Dense(128, num_class, weight_init=Normal(0.02))
    def construct(self, x):
        # 使用定义好的运算构建前向网络
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
def create_dataset(data_path, batch_size, repeat_num=2):
    """定义数据集"""
    data_set = ds.ImageFolderDataset(data_path, num_parallel_workers=8, shuffle=True)
    image_size = [100, 100]
    mean = [127.5, 127.5, 127.5]
    std = [255., 255., 255.]
    trans = [
        CV.Decode(),
        CV.Resize(image_size),
        CV.RandomCrop((100, 100)),
        CV.RandomHorizontalFlip(prob=0.5),
        CV.RandomColorAdjust(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        CV.Normalize(mean=mean, std=std),
        CV.HWC2CHW()
    ]
    # 实现数据的map映射、批量处理和数据重复的操作
    type_cast_op = C.TypeCast(mstype.int32)
    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=8)
    data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=8)
    data_set = data_set.batch(batch_size, drop_remainder=True)
    data_set = data_set.repeat(repeat_num)
    return data_set
batch_size = 16
train_data_path = "/home/jovyan/work/datasets/68919f50abbdb34700427ae2-momodel/Chinese medicine_train"
eval_data_path = "/home/jovyan/work/datasets/68919f50abbdb34700427ae2-momodel/Chinese medicine_eval"
train_ds = create_dataset(train_data_path, batch_size)
eval_ds = create_dataset(eval_data_path, 1)
num_classes = 4
input_channel = 3
net = cnn_net(num_classes, input_channel)
# 定义损失函数
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
# 定义优化器函数
net_opt = nn.Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9)
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
# 设置模型保存参数，模型训练保存参数的step为1000
config_ck = CheckpointConfig(save_checkpoint_steps=1000, keep_checkpoint_max=10)
# 应用模型保存参数
ckpoint = ModelCheckpoint(prefix="cnn_net", directory="/home/jovyan/work/results", config=config_ck)
# from mindvision.engine.callback import LossMonitor
from mindspore.train import Model
# 初始化模型参数
model = Model(net, loss_fn=net_loss, optimizer=net_opt, metrics={'accuracy'})
# 训练网络模型，并保存为cnn.ckpt文件
# model.train(10, train_ds, callbacks=[ckpoint, LossMonitor(0.01, 1000)])
model.train(10, train_ds, callbacks=[ckpoint])


# 修改result及其目录下生成的模型参数文件权限及所属用户
best_ckpt_dir = "/home/jovyan/work/results"
paths = Path(best_ckpt_dir)
new_p = 0o766

def change_permission(path):
    os.chmod(path, new_p)
    os.chown(path, 1000, 100)

for root, dirs, files in os.walk(paths):
    root = Path(root)
    change_permission(root)
    for file_name in files:
        change_permission(root.joinpath(file_name))
