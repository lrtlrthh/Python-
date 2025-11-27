###############################################################################
# 重要: 请务必把任务(jobs)中需要保存的文件存放在 results 文件夹内
# Important : Please make sure your files are saved to the 'results' folder
# in your jobs
###############################################################################
import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim import Adam
from sklearn.metrics import accuracy_score

seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



class BirdDataset(Dataset):
    def __init__(self, data_path, mode='train', transform=None):
        """
        数据读取器
        :param data_path: 数据集所在路径
        :param mode: train or eval
        """
        super(BirdDataset, self).__init__()
        self.data_path = data_path
        self.img_paths = []
        self.labels = []
        self.transform = transform

        if mode == 'train':
            with open(os.path.join(self.data_path, "train.txt"), "r", encoding="utf-8") as f:
                self.info = f.readlines()
            for img_info in self.info:
                img_path, label = img_info.strip().split('\t')
                self.img_paths.append(img_path)
                self.labels.append(int(label))

        else:
            with open(os.path.join(self.data_path, "eval.txt"), "r", encoding="utf-8") as f:
                self.info = f.readlines()
            for img_info in self.info:
                img_path, label = img_info.strip().split('\t')
                self.img_paths.append(img_path)
                self.labels.append(int(label))

    def __getitem__(self, index):
        """
        获取一组数据
        :param index: 文件索引号
        :return:
        """
        # 第一步打开图像文件并获取label值
        img_path = self.img_paths[index]
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB') 
        img = self.transform(img)
        label = self.labels[index]
        label = np.array([label], dtype="int64")
        return img, label

    def print_sample(self, index: int = 0):
        print("文件名", self.img_paths[index], "\t标签值", self.labels[index])

    def __len__(self):
        return len(self.img_paths)
'''
参数配置
'''
train_parameters = {
    "input_size": [3, 224, 224],                                #输入图片的shape
    "class_dim": 200,                                          #分类数
    "target_path":"/home/jovyan/work/datasets/689320899e78856c69427ad2-momodel/bird",                     #要解压的路径
    "train_list_path": "/home/jovyan/work/train.txt",       #train.txt路径
    "eval_list_path": "/home/jovyan/work/eval.txt",         #eval.txt路径
    "label_dict":{},         #标签字典
    "num_epochs": 60,                                         #训练轮数
    "train_batch_size": 32,                                   #训练时每个批次的大小
    "learning_strategy": {                                    #优化函数相关的配置
        "lr": 0.0001                                           #超参数学习率
    }, 
    'skip_steps': 5,                                         #每N个批次打印一次结果
    'save_epos': 4,                                         #每N个epoch保存一次模型参数
    "checkpoints": "/home/jovyan/work/results"          #保存的路径

}


train_transform = transforms.Compose([
    transforms.Resize(256), 
                        transforms.RandomRotation(degrees=15),  # 图像以-15到15的角度随机旋转
                        transforms.RandomHorizontalFlip(),     # 随机水平旋转图像，默认概率为50%
                        transforms.CenterCrop(224),   # 将图片从中心切剪成3*224*224大小的图片
                        transforms.ToTensor(),         # 把图片进行归一化为0-1，并把数据转换成Tensor类型
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])

eval_transform = transforms.Compose([
    transforms.Resize((224,224)), 
                        transforms.ToTensor(),         # 把图片进行归一化为0-1，并把数据转换成Tensor类型
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])

#训练数据加载
train_dataset = BirdDataset('/home/jovyan/work/',mode='train', transform = train_transform)
train_loader = DataLoader(train_dataset, 
                          batch_size=train_parameters['train_batch_size'], 
                          shuffle=True
                                    )
#测试数据加载
eval_dataset = BirdDataset('/home/jovyan/work/',mode='eval', transform = eval_transform)
eval_loader = DataLoader(eval_dataset,
                                   batch_size=train_parameters['train_batch_size'], 
                                   shuffle=False
                                   )



#自行定义卷积网络
class MyCNN(nn.Module):
    def __init__(self, class_nums):
        super(MyCNN, self).__init__()
        self.hidden1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.hidden2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.hidden3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
        self.hidden4 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.hidden5 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)
        self.hidden6 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.hidden7 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)
        self.hidden8 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.hidden11 = nn.Linear(512 * 14 * 14, 2048)
        self.hidden12 = nn.Linear(2048, 1024)
        self.hidden13 = nn.Linear(1024, 512)
        self.hidden14 = nn.Linear(512, 256)
        self.hidden15 = nn.Linear(256, class_nums)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        
    def forward(self, input):
        x = self.hidden1(input)
        x = self.relu(x)

        x = self.hidden2(x)
    
        x = self.hidden3(x)
        x = self.relu(x)

        x = self.hidden4(x)

        x = self.hidden5(x)
        x = self.relu(x)

        x = self.hidden6(x)

        x = self.hidden7(x)
        x = self.relu(x)

        x = self.hidden8(x)

        x = self.flatten(x)

        x = self.hidden11(x)
        x = self.relu(x)
        
        x = self.hidden12(x)
        x = self.relu(x)
        
        x = self.hidden13(x)
        x = self.relu(x)
        
        x = self.hidden14(x)
        x = self.relu(x)

        out = self.hidden15(x)
        return out
def draw_process(title,color,iters,data,label):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel(label, fontsize=20)
    plt.plot(iters, data,color=color,label=label) 
    plt.legend()
    plt.grid()
    plt.show()

use_gpu = torch.cuda.is_available()
model = MyCNN(train_parameters['class_dim'])

if use_gpu:
    model = model.cuda()
cross_entropy = nn.CrossEntropyLoss()
optimizer = Adam(params=model.parameters(), lr=train_parameters['learning_strategy']['lr']) 
best_eval_acc = 0
save_path = train_parameters["checkpoints"]+"/save_eval_best.ckpt"
                                  
steps = 0
Iters, total_loss, total_acc = [], [], []

for epo in range(train_parameters['num_epochs']):
    model.train()
    for _, data in enumerate(train_loader):
        steps += 1
        x_data = data[0]
        y_data = data[1].view(-1)
        if use_gpu:
            x_data, y_data = x_data.cuda(), y_data.cuda()
        predicts = model(x_data)
        loss = cross_entropy(predicts, y_data)
        pred = predicts.cpu().argmax(axis=1)
        acc = accuracy_score(pred, y_data.cpu())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if steps % train_parameters["skip_steps"] == 0:
            Iters.append(steps)
            total_loss.append(loss.cpu().item())
            total_acc.append(acc)
            #打印中间过程
            print('epo: {}, step: {}, loss is: {}, acc is: {}'\
                  .format(epo, steps, loss.cpu().item(), acc))
        #保存模型参数
    
    model.eval()
    eval_acc = []
    with torch.no_grad():
        for _, data in enumerate(eval_loader):
            x_data = data[0]
            y_data = data[1].view(-1)
            if use_gpu:
                x_data = x_data.cuda()
            predicts = model(x_data)
            pred = predicts.cpu().argmax(axis=1)
            acc = accuracy_score(pred, y_data)
            eval_acc.append(acc)
    current_eval_acc = np.mean(eval_acc)
    if current_eval_acc > best_eval_acc:
        best_eval_acc = current_eval_acc
        print('best_acc is: {}'.format(current_eval_acc))
        print('save model to: ' + save_path)
        torch.save(model.state_dict(),save_path)

torch.save(model.state_dict(),train_parameters["checkpoints"]+"/"+"save_dir_final.ckpt")
draw_process("trainning loss","red",Iters,total_loss,"trainning loss")
draw_process("trainning acc","green",Iters,total_acc,"trainning acc")
# '''
# 模型预测
# '''
# model__state_dict = paddle.load(train_parameters["checkpoints"]+"/"+"save_dir_final.pdparams")
# model_eval = MyCNN()
# model_eval.set_state_dict(model__state_dict) 
# model_eval.eval()
# accs = []

# for _, data in enumerate(eval_loader()):
#     x_data = data[0]
#     y_data = data[1]
#     predicts = model_eval(x_data)
#     acc = paddle.metric.accuracy(predicts, y_data)
#     accs.append(acc.numpy()[0])
# print('模型在验证集上的准确率为：',np.mean(accs))