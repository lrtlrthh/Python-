###############################################################################
# 重要: 请务必把任务(jobs)中需要保存的文件存放在 results 文件夹内
# Important : Please make sure your files are saved to the 'results' folder
# in your jobs
###############################################################################
import mindspore as ms
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import mindspore.nn as nn
import mindspore.ops as ops
from typing import List, Optional
import random
from pathlib import Path

from mindspore.common.initializer import Normal
from mindspore import dtype as mstype
from mindspore.dataset import ImageFolderDataset
from utils import util,ResidualBlock,ResNet


data_dir = "/home/jovyan/work/datasets/6892bfae9e78856c69427a90-momodel/train"
seed = 1234
random.seed(seed)
np.random.seed(seed)
ms.set_seed(seed)

path_names=os.listdir(data_dir)
classes=path_names
batch_size = 20  
re_image_size =32  
crop_size=100
workers = 4 
num_classes =len(path_names) 
num_samples=8000
dataset_train = util.create_dataset(dataset_dir=data_dir,
                                       usage="train",
                                       resize=re_image_size,
                                       batch_size=batch_size,
                                        crop_size=crop_size,
                                       workers=workers,
                                       num_samples=num_samples)
step_size_train = dataset_train.get_dataset_size()
num_samples=1000
dataset_val = util.create_dataset(dataset_dir=data_dir,
                                     usage="test",
                                     resize=re_image_size,
                                     batch_size=batch_size,
                                     workers=workers,
                                     num_samples=num_samples)
step_size_val = dataset_val.get_dataset_size()


network = util.resnet(ResidualBlock, [3, 4, 5, 3,2], num_classes, 4096)

in_channel = network.fc.in_channels
fc = nn.Dense(in_channels=in_channel, out_channels=num_classes)

network.fc = fc

num_epochs = 8
data_loader_train = dataset_train.create_tuple_iterator(num_epochs=num_epochs)
data_loader_val = dataset_val.create_tuple_iterator(num_epochs=num_epochs)
best_acc = 0
best_ckpt_dir = "/home/jovyan/work/results"
best_ckpt_path = "/home/jovyan/work/results/resnet-bestnew.ckpt"

my_util=util(num_epochs = num_epochs,network=network,step_size_train=step_size_train)
for epoch in range(num_epochs):
    curr_loss = my_util.train(data_loader_train, epoch)
    curr_acc = my_util.evaluate(data_loader_val)
    print("-" * 50)
    print("Epoch: [%3d/%3d], Average Train Loss: [%5.3f], Accuracy: [%5.3f]" % (
        epoch+1, num_epochs, curr_loss, curr_acc
    ))
    print("-" * 50)
ms.save_checkpoint(network, best_ckpt_path)


# 修改result及其目录下生成的模型参数文件权限及所属用户
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
