import mindspore as ms
import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore import nn, ops
import os
import stat

data_dir = "/home/jovyan/work/datasets/68959a7ee65e3f13b7427a7e-momodel/" # 数据集根目录
batch_size = 6 # 批量大小
image_size = [320, 320] # 训练图像空间大小
num_classes = 5

from pathlib import Path
paths = Path('/home/jovyan/work')
new_p = 0o766

def change_permission(path):
    # os.chmod(path, new_p)
    # os.chown(path, 1000, 100)
    pass

for root, dirs, files in os.walk(paths):
    root = Path(root)
    change_permission(root)
    for file_name in files:
        change_permission(root.joinpath(file_name))

def create_dataset(dataset_dir, usage, resize, batch_size):
    data_set = ds.ImageFolderDataset(dataset_dir=dataset_dir+usage,
                                 decode=True,
                                 shuffle=True)

    trans = []#需要做的变化的集合

    if usage == "train":
        trans += [
            vision.Resize(resize),
            vision.RandomColorAdjust(brightness=(0.5, 1),
                                        contrast=(0.4, 1),
                                        saturation=(0.3, 1)),
            vision.RandomHorizontalFlip(0.5)
                             ]

    """
    对数据集进行大小、规模的重组，以及归一化（帮助模型收敛）
    """
    trans += [
        vision.Resize(resize),
        vision.Rescale(1.0 / 255.0, 0.0),
        vision.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        vision.HWC2CHW()
    ]

    #对于label进行的操作
    target_trans = [(lambda x: np.array([x]).astype(np.int32)[0])]

    # 数据映射操作
    data_set = data_set.map(
        operations=trans,
        input_columns='image')

    data_set = data_set.map(
        operations=target_trans,
        input_columns='label')

    # 批量操作
    data_set = data_set.batch(batch_size)


    return data_set

dataset_train = create_dataset(dataset_dir=data_dir,
                                       usage="train",
                                       resize=image_size,
                                       batch_size=batch_size)

step_size_train = dataset_train.get_dataset_size()
index_label_dict = dataset_train.get_class_indexing()

dataset_val = create_dataset(dataset_dir=data_dir,
                                     usage="val",
                                     resize=image_size,
                                     batch_size=batch_size)

step_size_val = dataset_val.get_dataset_size()

print(step_size_val,step_size_train,index_label_dict,dataset_train)


data_iter = next(dataset_train.create_dict_iterator())
images = data_iter["image"].asnumpy()
labels = data_iter["label"].asnumpy()
print(f"Image shape: {images.shape}, Label: {labels}")

ms.set_context(device_target='CPU')

class VGG16(nn.Cell):
    def __init__(self):
        super().__init__()
        numClasses = 5
        self.all_sequential = nn.SequentialCell(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, pad_mode="pad"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, pad_mode="pad"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, pad_mode="pad"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, pad_mode="pad"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, pad_mode="pad"),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, pad_mode="pad"),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, pad_mode="pad"),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1, pad_mode="pad"),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, pad_mode="pad"),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, pad_mode="pad"),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1, pad_mode="pad"),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, pad_mode="pad"),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, pad_mode="pad"),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 原始模型vgg16输入image大小是224*224，这里使用的数据集输入大小为32*32，缩小7倍
            nn.Flatten(),
            nn.Dense(512*10*10, 256),
            # nn.Dropout(),
            nn.ReLU(),
            nn.Dense(256, 256),
            # nn.Dropout(),
            nn.ReLU(),
            nn.Dense(256, numClasses),
        )

    def construct(self, x):
        x = self.all_sequential(x)
        return x

from mindspore import load_checkpoint, load_param_into_net

def _vgg16(pretrained: bool = False):
    model = VGG16()
    "VGG16模型"
    #预训练模型的下载网址
    # model_url = "https://download.mindspore.cn/model_zoo/official/cv/vgg/vgg16_ascend_0.5.0_cifar10_official_classification_20200715/vgg16.ckpt"
    # #存储路径
    # model_ckpt = "./LoadPretrainedModel/vgg16_0715.ckpt"

    # if pretrained:
    #     download(url=model_url, path=model_ckpt)
    #     param_dict = load_checkpoint(model_ckpt)
    #     load_param_into_net(model, param_dict)

    return model


# 定义VGG16网络，此处不采用预训练，即将pretrained设置为False
vgg16 = _vgg16(pretrained=False)

#param.requires_grad = True表示所有参数都需要求梯度进行更新。
for param in vgg16.get_parameters():
    param.requires_grad = True

# 设置训练的轮数和学习率，这里训练的轮数设置为40
num_epochs = 200
#基于余弦衰减函数计算学习率。学习率最小值为0.001，最大值为0.005，具体API见文档https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.cosine_decay_lr.html?highlight=cosine_decay_lr
lr = nn.cosine_decay_lr(min_lr=0.0001, max_lr=0.0005, total_step=step_size_train * num_epochs,
                        step_per_epoch=step_size_train, decay_epoch=num_epochs)
# lr = 0.001
# 定义优化器和损失函数
#Adam优化器
opt = nn.Adam(params=vgg16.trainable_params(), learning_rate=lr)
# 交叉熵损失
loss_fn = nn.CrossEntropyLoss()

#前向传播，计算loss
def forward_fn(inputs, targets):
    logits = vgg16(inputs)
    loss = loss_fn(logits, targets)
    return loss

#计算梯度和loss
grad_fn = ops.value_and_grad(forward_fn, None, opt.parameters)

def train_step(inputs, targets):
    loss, grads = grad_fn(inputs, targets)
    opt(grads)
    return loss

# 实例化模型
model = ms.Model(vgg16, loss_fn, opt, metrics={"Accuracy": nn.Accuracy()})
# 创建迭代器
data_loader_train = dataset_train.create_tuple_iterator(num_epochs=num_epochs)
data_loader_val = dataset_val.create_tuple_iterator(num_epochs=num_epochs)

# 最佳模型存储路径
best_acc = 0
best_ckpt_dir = "./results/"
best_ckpt_path = "./results/vgg16-best_model7.ckpt"
print("Start Training Loop ...")


Iters = list(i for i in range(num_epochs))
total_loss, total_acc =  [], []

for epoch in range(num_epochs):
    losses = []
    vgg16.set_train()

    # 为每轮训练读入数据

    for i, (images, labels) in enumerate(data_loader_train):
        loss = train_step(images, labels)
        if i % 10 == 0 or i == step_size_train -1:
            print('Epoch: [%3d/%3d], Steps: [%3d/%3d], Train Loss: [%5.3f]'%(
                epoch+1, num_epochs, i+1, step_size_train, loss))
        losses.append(loss)

    # 每个epoch结束后，验证准确率

    acc = model.eval(dataset_val)['Accuracy']
    total_acc.append(acc)
    total_loss.append(float(sum(losses)/len(losses)))


    print("-" * 50)
    print("Epoch: [%3d/%3d], Average Train Loss: [%5.3f], Accuracy: [%5.3f]" % (
        epoch+1, num_epochs, sum(losses)/len(losses), acc))
    print("-" * 50)

    if acc > best_acc:
        best_acc = acc
        if not os.path.exists(best_ckpt_dir):
            os.mkdir(best_ckpt_dir)
        if os.path.exists(best_ckpt_path):
            os.chmod(best_ckpt_path, stat.S_IWRITE)#取消文件的只读属性，不然删不了
            os.remove(best_ckpt_path)
        ms.save_checkpoint(vgg16, best_ckpt_path)

print("=" * 80)
print(f"End of validation the best Accuracy is: {best_acc: 5.3f}, "
      f"save the best ckpt file in {best_ckpt_path}", flush=True)

print("Iters:",Iters)
print("total_loss:",total_loss)
print("total_acc:",total_acc)


y_true = []
y_pred = []

# 加载验证集的数据进行验证
for i, (images, labels) in enumerate(data_loader_val):
    for label in labels.asnumpy():
        y_true.append(label)

    # 预测图像类别
    output = model.predict(ms.Tensor(images))
    preds = np.argmax(output.asnumpy(), axis=1)

    for pred in preds:
        y_pred.append(pred)

    print("第{}轮的lable:{},preds:{}".format(i,labels.asnumpy(),preds))

# 打印结果
print("y_true:",y_true)
print("y_pred:",y_pred)
