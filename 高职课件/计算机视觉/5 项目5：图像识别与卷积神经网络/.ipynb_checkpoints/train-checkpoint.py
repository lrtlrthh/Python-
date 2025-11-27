###############################################################################
# 重要: 请务必把任务(jobs)中需要保存的文件存放在 results 文件夹内
# Important : Please make sure your files are saved to the 'results' folder
# in your jobs
###############################################################################
import numpy as np
import random
from PIL import Image
import os
import mindspore
from mindspore import nn
from mindspore.dataset import transforms, vision
from mindspore.dataset import GeneratorDataset
import matplotlib.pyplot as plt
from mindspore import ops
import pandas as pd
import argparse
import warnings
import time
import matplotlib.pyplot as plt
from tqdm import tqdm


class Flower_data():
    def __init__(self, source_root, train_rate=0.6, mode='train', transform=None):
        self.images = []
        self.labels = []
        self.transform = transform
        # 设置预处理
        if self.transform is None:
            self.transform = transforms.Compose([
                vision.Resize((256, 256)),
                vision.Rescale(1.0 / 255.0, 0),
                vision.RandomCrop((224, 224)),
                vision.Normalize(mean=(0.485,), std=(0.229,)),
                vision.HWC2CHW()
            ])
        kind_list = os.listdir(source_root)

        # 加载数据
        for kind in kind_list:
            images_list = os.listdir(os.path.join(source_root, kind))
            for images in images_list:
                self.images.append(os.path.join(source_root, kind, images))
                self.labels.append(int(kind) - 1)  # 因为没有0，所以-1补充

        # 随机打乱顺序
        state = np.random.get_state()
        np.random.shuffle(self.images)
        np.random.set_state(state)
        np.random.shuffle(self.labels)

        #         print(*zip(self.images, self.labels))

        # 划分训练和验证集
        assert mode in ['train', 'valid']
        if mode == 'train':
            self.images = self.images[:int(len(self.images) * train_rate)]
            self.labels = self.labels[:int(len(self.labels) * train_rate)]
        elif mode == 'valid':
            self.images = self.images[int(len(self.images) * train_rate):]
            self.labels = self.labels[int(len(self.labels) * train_rate):]

        # print(self.labels)

    def __getitem__(self, index):
        #         image_data = cv2.imread(self.images[index], cv2.IMREAD_COLOR)
        image_data = Image.open(self.images[index])
        #         image_data = cv2.resize(image_data, (224, 224))
        image_data = np.array(image_data)
        #         image_data = self.transform(image_data)
        return mindspore.Tensor(image_data,mindspore.float32), mindspore.Tensor(self.labels[index], mindspore.int32)


    def __len__(self):
        return len(self.labels)

def fix_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    mindspore.set_seed(SEED)

fix_seed(1234)

def find_name(json_file='/home/jovyan/work/datasets/688c8679b5074ef61818a082-momodel/cat_to_name.json'):
    with open(json_file, 'r') as f:
        dic = eval(''.join(f.readlines()))
        return dic


class ResidualBlock(nn.Cell):
    def __init__(self, input_c, output_c):
        super().__init__()
        self.conv1 = nn.Conv2d(input_c, output_c, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(output_c)
        self.conv2 = nn.Conv2d(output_c, output_c, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(output_c)

        self.relu = nn.ReLU()

    def construct(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x += residual
        return x


class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=(3, 3, 3, 3),
                               pad_mode='pad')
        self.bn1 = nn.BatchNorm2d(64)
        self.resi1 = ResidualBlock(64, 64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='SAME')

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.resi2 = ResidualBlock(128, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='SAME')

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.resi3 = ResidualBlock(256, 256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='SAME')

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.resi4 = ResidualBlock(512, 512)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='SAME')

        self.avg = nn.AvgPool2d(kernel_size=7, stride=1)
        self.relu = nn.ReLU()

        self.flatten = nn.Flatten()
        self.dense_relu_sequential = nn.SequentialCell(
            nn.Dense(512, 1024),
            nn.ReLU(),
            nn.Dense(1024, 512),
            nn.ReLU(),
            nn.Dense(512, 102)
        )

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.resi1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.resi2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.resi3(x)
        x = self.maxpool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.resi4(x)
        x = self.maxpool4(x)

        x = self.avg(x)
        x = self.flatten(x)
        x = self.dense_relu_sequential(x)
        return x



def get_argparse():
    parser = argparse.ArgumentParser(description=" Classification of Flower")
    parser.add_argument('--batch_size', default=8)
    parser.add_argument('--Learning_rate', default=1e-4)
    parser.add_argument('--classes', default=102)
    parser.add_argument('--max_epoch', default=100)
    parser.add_argument('--model1_name', default='UUnet')
    parser.add_argument('--device', default='gpu')
    parser.add_argument('--model_path', default=f'/home/jovyan/work/results/{now_time}/model.ckpt')
    parser.add_argument('--load_model_path', default=f'')
    parser.add_argument('--info', default=dict())
    args = parser.parse_args([])
    args.info['epoch'] = []
    args.info['train_loss'] = []
    args.info['valid_loss'] = []
    args.info['valid_Accuracy'] = []
    args.info['valid_Recall'] = []
    args.info['valid_Precision'] = []
    args.info['valid_F1_score'] = []
    args.best_accuracy = 0
    return args


# 绘制损失函数
def draw_loss(cfg):
    plt.figure()
    plt.plot(cfg.info['epoch'], cfg.info['train_loss'], label='train_loss')
    plt.plot(cfg.info['epoch'], cfg.info['valid_loss'], label='valid_loss')
    plt.legend()
    plt.savefig(f"/home/jovyan/work/results/{now_time}/loss.png")
    plt.show()
    return


def draw_val_acc(cfg):
    plt.figure()
    plt.plot(cfg.info['epoch'], cfg.info['valid_Accuracy'], label='valid_Accuracy')
    plt.legend()
    plt.savefig(f"/home/jovyan/work/results/{now_time}/valid_Accuracy.png")
    plt.show()
    return


def save_info(cfg):
    dataframe = pd.DataFrame(cfg.info)
    dataframe.to_csv(f"/home/jovyan/work/results/{now_time}/combine_result.csv", index=False, sep=',')
    return

def model_train(cfg, model, optimizer, loss_func, train_dataset):
    def forward_fn(data, label):
        out = model(data)
        loss = loss_func(out, label)
        return loss, out

    length = train_dataset.get_dataset_size()
    train_dataloader = train_dataset.create_tuple_iterator()
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
    model.set_train()

    train_loss = []

#     train_tq = tqdm(total=length)
    for ite, (data, label) in enumerate(train_dataloader):
#         train_tq.set_description("train:")
        (loss, _), grads = grad_fn(data, label)
        loss = ops.depend(loss, optimizer(grads))
        train_loss.append(float(loss))
#         train_tq.set_postfix(loss=loss)
#         train_tq.update(1)

    train_loss = np.mean(train_loss)
    cfg.info['train_loss'].append(train_loss)
    # mindspore.save_checkpoint(model, cfg.model_path)


def model_eval(cfg, model, optimizer, loss_func, test_dataset):
    def forward_fn(data, label):
        out = model(data)
        loss = loss_func(out, label)
        return loss, out

    length = test_dataset.get_dataset_size()
    test_dataloader = test_dataset.create_tuple_iterator()

    model.set_train(False)

    total, test_loss, correct = 0, 0, 0
#     test_tq = tqdm(total=length)
    for ite, (data, label) in enumerate(test_dataloader):
#         test_tq.set_description("valid:")
        out = model(data)
        total += len(data)
        loss = loss_func(out, label).asnumpy()
        test_loss += loss
        correct += (out.argmax(1) == label).asnumpy().sum()

#         test_tq.set_postfix(loss=loss)
#         test_tq.update(1)

    test_loss /= length
    correct /= total

    cfg.info['valid_loss'].append(test_loss)
    cfg.info['valid_Accuracy'].append(correct)

    if correct > cfg.best_accuracy:
        cfg.best_accuracy = correct
        mindspore.save_checkpoint(model, cfg.model_path)


warnings.filterwarnings('ignore')
now_time = time.strftime('%Y_%m_%d_%H_%M')

try:
    os.mkdir(f'/home/jovyan/work/results/{now_time}/')
except:
    pass


# 定义超参数
cfg = get_argparse()
print(cfg)

# 数据集的构建
train_data = Flower_data(source_root='/home/jovyan/work/datasets/688c8679b5074ef61818a082-momodel/flower_data/train', train_rate=0.6, mode='train')
train_dataset = GeneratorDataset(source=train_data, column_names=["data", "label"],
                                 shuffle=True).map(operations=train_data.transform,
                                                   num_parallel_workers=1).batch(batch_size=cfg.batch_size)

test_data = Flower_data(source_root='/home/jovyan/work/datasets/688c8679b5074ef61818a082-momodel/flower_data/train', train_rate=0.6, mode='valid')
test_dataset = GeneratorDataset(source=test_data, column_names=["data", "label"],
                                shuffle=False).map(operations=test_data.transform,
                                                   num_parallel_workers=1).batch(batch_size=cfg.batch_size)

# 模型，优化器，损失函数
model = Network()
loss_func = nn.CrossEntropyLoss()
optimizer = nn.Adam(model.trainable_params(), learning_rate=cfg.Learning_rate)

# 开始迭代训练
for epoch in range(cfg.max_epoch):
    cfg.info['epoch'].append(epoch + 1)
    # 可视化输出
    model_train(cfg, model, optimizer, loss_func, train_dataset)
    print('{{"metric": "epoch", "value": {}}}'.format(epoch + 1))
    model_eval(cfg, model, optimizer, loss_func, test_dataset)
    print('{{"metric": "accuracy", "value": {}}}'.format(cfg.info["valid_Accuracy"][-1]))
#     print(epoch + 1, 'valid_Accuracy ---', cfg.info['valid_Accuracy'][-1])
draw_loss(cfg)
draw_val_acc(cfg)


from pathlib import Path
import os
paths = Path('/home/jovyan/work/results')
new_p = 0o766

def change_permission(path):
    os.chmod(path, new_p)
    os.chown(path, 1000, 100)

for root, dirs, files in os.walk(paths):
    root = Path(root)
    change_permission(root)
    for file_name in files:
        change_permission(root.joinpath(file_name))