import os
import random
from pathlib import Path
import matplotlib.pyplot as plt
from mindspore import nn
from mindspore import Tensor
import numpy as np
import mindspore
import mindspore.dataset as ds
import mindspore.dataset.vision as CV
import mindspore.dataset.transforms as C
from mindspore import dtype as mstype

batch_size_train = 32  # 训练集划分为batch的大小
batch_size_test = 1  # 测试集划分为batch的大小
repeat = 1  # 训练集加载时，图像重复的次数
learning_rate = 1e-2  # 模型训练学习率
epoch = 20  # 训练集的训练轮次
resolution = 224

seed = 1234
random.seed(seed)
np.random.seed(seed)
mindspore.set_seed(seed)

path_train_set = "/home/jovyan/work/datasets/6892fb249e78856c69427a9c-momodel/faceDataset/train_set"
path_test_set = "/home/jovyan/work/datasets/6892fb249e78856c69427a9c-momodel/faceDataset/test_set"
Reset = False  # 是否重置模型参数
PATH = '/home/jovyan/work/results/parameter_c.ckpt'  # 参数的目录



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


class Network(nn.Cell):  # 模型结构
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


def dataset_transform(dataset, side_length=resolution):  # 图像的转换处理
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


def create_batch(dataset, batch_size, repeat=1):  # 图像划分批次
    batch_set = dataset.batch(batch_size, drop_remainder=True)
    num_bt = batch_set.get_dataset_size()
    batch_set = batch_set.repeat(repeat)
    print("数据集划分为的batch数为：" + str(num_bt))
    return batch_set


print("训练集：")
train_set = create_dataset(path_train_set)
train_set = dataset_transform(train_set)
train_set = create_batch(train_set, batch_size_train, repeat)

print("测试集：")
test_set = create_dataset(path_test_set)
test_set = dataset_transform(test_set)
test_set = create_batch(test_set, batch_size_test, 1)


network = Network()
# if os.path.exists(PATH) and not Reset:   # 加载模型参数
#     param_dict = mindspore.load_checkpoint(PATH)
#     mindspore.load_param_into_net(network, param_dict)
#     print("模型参数加载完毕")

loss = nn.CrossEntropyLoss()
optimizer = nn.Adam(network.trainable_params(), learning_rate)
network.set_train()


def forward(data, label):  # 前向传播
    predict = network(data, batch_size_train)
    result = loss(predict, label)
    return result, predict


gradient = mindspore.value_and_grad(forward, None, optimizer.parameters, has_aux=True)  #梯度


def train_step(data, label):
    (loss, _), grads = gradient(data, label)
    optimizer(grads)
    return _, loss


def expand_label(label):  # 扩展标签维度
    labels = np.zeros((batch_size_train, 2)).astype(np.float32)
    for i in range(0, batch_size_train):  # 标签为0则第0维的值更大，便于之后测试进行argmax相对应
        if abs(label[i] - 0.0) < 0.01:
            labels[i][0] = 1.0
        elif abs(label[i] - 1.0) < 0.01:
            labels[i][1] = 1.0
    labels = Tensor(labels)
    return labels


def show_loss(loss_list):  # 将损失变化输出为图片
    plt.figure()
    plt.plot(loss_list, 'b', label='Recon_loss')
    plt.ylabel('Recon_loss')
    plt.xlabel('iter_num')
    plt.legend()
    plt.savefig("/home/jovyan/work/results/1_recon_loss.jpg")


def train():
    global learning_rate
    running_loss = 0
    loss_list = []
    store_dis = 0
    store_loss = running_loss
    for i in range(epoch):
        print("epoch:" + str(i))
        for batch, (data, label) in enumerate(train_set.create_tuple_iterator()):
            label = expand_label(label)
            predict, result = train_step(data, label)
            loss_list.append(result)
            running_loss += result

            if batch % 10 == 0:
                print("the train loss is: %.4f" % (running_loss / 10))
                mindspore.save_checkpoint(network, PATH)
                if abs(store_loss - running_loss) < store_dis and learning_rate > 0.00001:
                    learning_rate *= 0.8
                store_dis = abs(store_loss - running_loss)
                store_loss = running_loss
                running_loss = 0
            
            if batch % 40 == 0:
                test_loss()

            if batch % 100 == 0:
                print(learning_rate)
                show_loss(loss_list)
                loss_list = []
                
        test()
    print("训练结束！")


def test_loss():  # 跟踪展示测试集上的损失
    running_loss = 0
    for batch, (data, label) in enumerate(test_set.create_tuple_iterator()):
        predict = network(data, batch_size_test)
        result = loss(predict, label)
        running_loss += result
        if batch == 99:
            print("the test loss is: %.4f" % (running_loss / 100))
            return


def test():
    sum_test = 0
    num = 0
    for batch, (data, label) in enumerate(test_set.create_tuple_iterator()):
        sum_test += batch_size_test
        predict = network(data, batch_size_test)
        predict = predict.argmax(1)
        for i in range(len(label)):
            if int(label[i]) == predict[i]:
                num += 1

    current = num / sum_test
    print("the current rate is: %.2f %%" % (current * 100))
    return current


train()
test()


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

