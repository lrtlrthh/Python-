###############################################################################
# 重要: 请务必把任务(jobs)中需要保存的文件存放在 results 文件夹内
# Important : Please make sure your files are saved to the 'results' folder
# in your jobs
###############################################################################
import pandas as pd
import numpy as np
# 读取 xlsx 文件数据
coffee_data = pd.read_excel('/home/jovyan/work/datasets/675659647fc08c9152b2572a-momodel/coffee_dataset.xlsx')

# kind属性中出现了Arabica 和 Robusta两种取值，用 0 代表 Arabica，1 代表 Robusta
coffee_data['kind'].replace({'Arabica': 0, 'Robusta': 1}, inplace=True)

# color属性中出现了Light Brown、Medium Brown、Dark Brown三种取值，分别用0，1，2代替
coffee_data['color'].replace({'Light Brown': 0, 'Medium Brown': 1, 'Dark Brown':2}, inplace=True)

# aroma属性中出现了Floral、Fruity、Nutty、Chocolaty、Spicy、Caramel六种取值，分别用0、1、2、3、4、5代替
coffee_data['aroma'].replace({'Floral': 0, 'Fruity': 1, 'Nutty':2, 'Chocolaty':3, 'Spicy':4, 'Caramel':5}, inplace=True)

# acidity属性为0-100区间范围内整数型取值，设定范围1-50区间内替换为0，50-80区间内替换为1，80-100区间内替换为2
coffee_data['acidity'].replace(range(1, 51), 0, inplace=True)
coffee_data['acidity'].replace(range(50, 81), 1, inplace=True)
coffee_data['acidity'].replace(range(80, 101), 2, inplace=True)

# mellow属性中出现了Light-bodied、Medium-bodied、Full-bodied三种取值，分别用0、1、2代替
coffee_data['mellow'].replace({'Light-bodied': 0, 'Medium-bodied': 1, 'Full-bodied':2}, inplace=True)

# aftertaste属性中出现了Short、Medium、Long三种取值，分别用0、1、2代替
coffee_data['aftertaste'].replace({'short': 0, 'medium': 1, 'long':2}, inplace=True)

# sweetness属性中出现了Low、Medium、High三种取值，分别用0、1、2代替
coffee_data['sweetness'].replace({'low': 0, 'medium': 1, 'high':2}, inplace=True)

# cleanliness属性中出现了Low、Medium、High三种取值，分别用0、1、2代替
coffee_data['cleanliness'].replace({'low': 0, 'medium': 1, 'high':2}, inplace=True)

# complexity属性中出现了Low、Medium、High三种取值，分别用0、1、2代替
coffee_data['complexity'].replace({'low': 0, 'medium': 1, 'high':2}, inplace=True)

# foam属性中出现了No、Low、Moderate、Abundant四种取值，分别用0、1、2、3代替
coffee_data['foam'].replace({'No': 0, 'low': 1, 'moderate':2, 'abundant':3}, inplace=True)

# caffeine属性为0-120区间范围内整数型取值，设定范围0-30区间内替换为0，30-100区间内替换为1，100-120区间内替换为2
coffee_data['caffeine'].replace(range(0, 31), 0, inplace=True)
coffee_data['caffeine'].replace(range(30, 101), 1, inplace=True)
coffee_data['caffeine'].replace(range(100, 121), 2, inplace=True)

# brew_method属性中出现了Drip Brewing、Pressure Brewing、Instant Brewing、Cold Drip Brewing、Immersion Brewing五种取值，分别用0、1、2、3、4代替
coffee_data['brew_method'].replace({'drip brewing': 0, 'pressure brewing': 1, 'instant brewing':2, 'cold drip brewing':3,  'immersion brewing':4}, inplace=True)

# dry_method属性中出现了Natural、Mechanical、Solar三种取值，分别用0、1、2代替
coffee_data['dry_method'].replace({'natural': 0, 'mechanical': 1, 'solar':2}, inplace=True)

# 复制一份数据集
coffee_copy = coffee_data.copy(deep=True)

coffee_copy['popular'] = coffee_copy.groupby(['kind', 'color', 'aroma', 'acidity', 'mellow', 'aftertaste', 'cleanliness', 'complexity', 'foam', 'caffeine', 'brew_method', 'dry_method'])['kind'].transform('count')
coffee_copy.drop_duplicates(inplace=True)

coffee_copy['popular'].describe().loc[['min', 'max']]
# 定义要统计的属性列表
attributes = ['kind', 'color', 'aroma', 'acidity', 'mellow', 'aftertaste', 'sweetness', 'cleanliness', 'complexity','foam','caffeine','brew_method','dry_method']

value_labels = {
    'kind': {0: '0', 1: '1'},
    'color': {0: '0', 1: '1', 2: '2'},
    'aroma': {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5'},
    'acidity': {0: '0', 1: '1', 2: '2'},
    'mellow': {0: '0', 1: '1', 2: '2'},
    'aftertaste': {0: '0', 1: '1', 2: '2'},
    'sweetness': {0: '0', 1: '1', 2: '2'},
    'cleanliness': {0: '0', 1: '1', 2: '2'},
    'complexity': {0: '0', 1: '1', 2: '2'},
    'foam': {0: '0', 1: '1', 2: '2', 3: '3'},
    'caffeine': {0: '0', 1: '1', 2: '2'},
    'brew_method': {0: '0', 1: '1', 2: '2', 3: '3', 4: '4'},
    'dry_method': {0: '0', 1: '1', 2: '2'},
    
    
}
# 创建空的结果DataFrame
result = pd.DataFrame()

# 打印结果以表格形式展示，包括具体取值
for attribute in attributes:
    print(f"Attribute: {attribute}")
    attribute_counts = coffee_copy[attribute].value_counts(normalize=True).reset_index()
    attribute_counts.columns = ['Value', 'Probability']
    attribute_counts['Value'] = attribute_counts['Value'].map(value_labels[attribute])
    for index, row in attribute_counts.iterrows():
        print(f"Value: {row['Value']}, Probability: {row['Probability']:.2%}")
    print()


# 打印结果以表格形式展示
print(result.to_string(index=False))
# 统计popular属性的取值频次
popular_counts = coffee_copy['popular'].value_counts().reset_index()
popular_counts.columns = ['Value', 'Count']

# 按照取值从小到大排序
popular_counts = popular_counts.sort_values(by='Value')

# 创建新的属性列 popular_level，并根据 popular 列的取值进行分类
coffee_copy['popular_level'] = np.where(coffee_copy['popular'] > 10, 1, 0)

# 转换为整数类型，这样后面的热力图可以展示该属性列
coffee_copy['popular_level'] = coffee_copy['popular_level'].astype(int)

print(coffee_copy['popular_level'].dtype)
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
plt.figure(dpi=100, figsize=(10, 5))# 设置图片大小
coffee_without_popular = coffee_copy.drop("popular", axis=1)  # 去掉属性列"popular"
sns.heatmap(coffee_without_popular.corr(), annot=True, fmt='.2f')
# 筛选相关性大于0.5的属性间关系（不包括对角线）
threshold = 0.5
correlation_matrix = coffee_copy.corr()
high_correlation = correlation_matrix[(correlation_matrix > threshold) & (correlation_matrix < 1.0)]
high_correlation = high_correlation.dropna(how='all', axis=1).dropna(how='all', axis=0)

# 按相关性大小从大到小进行排序
high_correlation_sorted = high_correlation.stack().sort_values(ascending=False)

# 打印排序结果，转换为元组形式并去除重复元组，不打印包含 'popular' 和 'popular_level' 属性的元组
unique_tuples = set()
for idx, value in high_correlation_sorted.items():
    attr1, attr2 = idx
    if ('popular' in (attr1, attr2)) or ('popular_level' in (attr1, attr2)):
        continue
    if (attr2, attr1) not in unique_tuples:
        unique_tuples.add((attr1, attr2))
        print((attr1, attr2, value))
# 设置相关性阈值
threshold = 0.2

# 获取与特定属性 'popular_level' 相关性大于阈值的属性，不包括popular属性和自身
correlation_matrix = coffee_copy.corr()
related_attributes = correlation_matrix['popular_level'][(correlation_matrix['popular_level'] > threshold) & (correlation_matrix.columns != 'popular') 
 & (correlation_matrix.columns != 'popular_level')].index.tolist()

import mindspore as ms
import mindspore.nn as nn
import mindspore.dataset as ds
from mindspore import Model
from mindspore.train.callback import LossMonitor
# 将 coffee_modified 中的 DataFrame对象转换为适合 MindSpore 框架的 Tensor 格式，同时去掉无用的popular列，只保留popular_level
coffee_tensor = coffee_copy.drop('popular', axis=1)
coffee_tensor = ms.Tensor(coffee_tensor.values, ms.float32) 


#print(coffee_tensor)
#selected_columns = ['color','aroma','mellow','aftertaste','sweetness','foam','brew_method','popular_level']
#selected_columns = ['cleanliness', 'complexity', 'brew_method']

# 定义属性权重, 对应位置属性赋予更大的权重
attribute_weights = ms.Tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 1.0, 1.0, 3.0, 1.0], dtype=ms.float32)

# 特征列索引为0到4，标签列索引为5
features = coffee_tensor[:, :13]
labels = coffee_tensor[:, 13].astype(np.int32)
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


#加性注意力
class AttentionLayer(nn.Cell):
    def __init__(self, input_dim, output_dim):  # 修改构造函数，接收输出维度
        super(AttentionLayer, self).__init__()
        self.fc = nn.Dense(input_dim, output_dim)  # 后面训练模型时，创建AttentionLayer实例时，传入参数
        self.softmax = nn.Softmax(axis=1)  #对注意力权重进行归一化，使得注意力权重之和等于1

    def construct(self, x):
        attention_scores = self.fc(x)
        attention_weights = self.softmax(attention_scores)
        attended_features = attention_weights * x * attention_weights
        return attended_features

    
class MyModel(nn.Cell):
    def __init__(self, num_features, num_classes, attribute_weights):
        super(MyModel, self).__init__()
        self.fc1 = nn.Dense(num_features, 64)
        self.relu1 = nn.ReLU()
        self.attention_layer = AttentionLayer(64, 64)  # 修改输出维度为 64
        self.fc2 = nn.Dense(64, num_classes)
        self.attribute_weights = attribute_weights

    def construct(self, x):
        x = self.fc1(x)
        x = self.relu1(x)

        x = self.attention_layer(x)
        x = self.fc2(x)

        return x
# 创建模型实例
model = MyModel(num_features=13, num_classes=2, attribute_weights=attribute_weights)

# 定义损失函数
loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

# 定义优化器
optimizer = nn.Adam(model.trainable_params(), learning_rate=0.001)

# 创建WithLossCell
network_with_loss = nn.WithLossCell(model, loss)

# 创建TrainOneStepCell
train_network = nn.TrainOneStepCell(network_with_loss, optimizer)
# 数据预处理和转换
def preprocess_data(features, labels):
    features = features.astype(np.float32)
    labels = labels.astype(np.int32)
    return ms.Tensor(features), ms.Tensor(labels)

# 定义训练过程
def train(train_network, features, labels, epochs):
    features, labels = preprocess_data(features, labels)

    for epoch in range(epochs):
        epoch_loss = 0.0

        # 执行一次训练
        loss = train_network(features, labels)

        epoch_loss = loss.asnumpy()
        #print("Epoch: {}, Loss: {}".format(epoch, epoch_loss))

# 假设训练10个epochs
train(train_network, features, labels, epochs=100)
from graphviz import Digraph

# 生成模型结构的树形图
dot = Digraph()
dot.attr(rankdir='LR')  # 设置图的方向，从左到右
dot.node_attr.update(shape='box')  # 设置节点的形状为矩形

# 遍历模型的子模块，生成节点和边
for cell in model.cells():
    dot.node(str(id(cell)), str(cell))
    for subcell in cell.cells():
        dot.edge(str(id(cell)), str(id(subcell)))

# 展示树形图
dot
# 读取 xlsx 文件数据
coffee_test = pd.read_excel('/home/jovyan/work/datasets/675659647fc08c9152b2572a-momodel/coffee_test.xlsx')

#print(coffee_test)
## 处理测试集数据
coffee_test['kind'].replace({'Arabica': 0, 'Robusta': 1}, inplace=True)

coffee_test['color'].replace({'Light Brown': 0, 'Medium Brown': 1, 'Dark Brown':2}, inplace=True)

coffee_test['aroma'].replace({'Floral': 0, 'Fruity': 1, 'Nutty':2, 'Chocolaty':3, 'Spicy':4, 'Caramel':5}, inplace=True)

coffee_test['acidity'].replace(range(1, 51), 0, inplace=True)
coffee_test['acidity'].replace(range(50, 81), 1, inplace=True)
coffee_test['acidity'].replace(range(80, 101), 2, inplace=True)

coffee_test['mellow'].replace({'Light-bodied': 0, 'Medium-bodied': 1, 'Full-bodied':2}, inplace=True)

coffee_test['aftertaste'].replace({'short': 0, 'medium': 1, 'long':2}, inplace=True)

coffee_test['sweetness'].replace({'low': 0, 'medium': 1, 'high':2}, inplace=True)

coffee_test['cleanliness'].replace({'low': 0, 'medium': 1, 'high':2}, inplace=True)

coffee_test['complexity'].replace({'low': 0, 'medium': 1, 'high':2}, inplace=True)

coffee_test['foam'].replace({'No': 0, 'low': 1, 'moderate':2, 'abundant':3}, inplace=True)

coffee_test['caffeine'].replace(range(0, 31), 0, inplace=True)
coffee_test['caffeine'].replace(range(30, 101), 1, inplace=True)
coffee_test['caffeine'].replace(range(100, 121), 2, inplace=True)

coffee_test['brew_method'].replace({'drip brewing': 0, 'pressure brewing': 1, 'instant brewing':2, 'cold drip brewing':3,  'immersion brewing':4}, inplace=True)

coffee_test['dry_method'].replace({'natural': 0, 'mechanical': 1, 'solar':2}, inplace=True)

#print(coffee_test)
# 将测试数据转换为Tensor格式
test_data = ms.Tensor(coffee_test.values, ms.float32)

# 提取特征
test_features = test_data[:, :13]

# 使用训练好的模型进行预测
predictions = model(test_features)

# 将预测结果转换为类别标签
predicted_labels = np.argmax(predictions, axis=1)
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

# 比较预测类别与真实标签，计算预测正确的样本数
true_label = coffee_test['popular_level'].values
correct_predictions = np.sum(predicted_labels == true_label)
# 计算准确率
accuracy = correct_predictions / len(true_label)

print("Test Accuracy: {:.4f}".format(accuracy))
# 特征列索引为0到12，标签列索引为13
features = coffee_tensor[:, :13]
labels = coffee_tensor[:, 13].astype(np.int32)
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


#加性注意力
class AttentionLayer(nn.Cell):
    def __init__(self, input_dim, output_dim):  # 修改构造函数，接收输出维度
        super(AttentionLayer, self).__init__()
        self.fc = nn.Dense(input_dim, output_dim)  # 后面训练模型时，创建AttentionLayer实例时，传入参数
        self.softmax = nn.Softmax(axis=1)  #对注意力权重进行归一化，使得注意力权重之和等于1

    def construct(self, x):
        attention_scores = self.fc(x)
        attention_weights = self.softmax(attention_scores)
        attended_features = attention_weights * x
        return attended_features

    
class MyModel(nn.Cell):
    def __init__(self, num_features, num_classes):
        super(MyModel, self).__init__()
        self.fc1 = nn.Dense(num_features, 64)
        self.relu1 = nn.ReLU()
        # self.fc2 = nn.Dense(64, num_classes)
        self.fc2 = nn.Dense(64, num_classes)

        self.attention_layer = AttentionLayer(64, 64)  # 修改输出维度为 64

    def construct(self, x):
        x = self.fc1(x)
        x = self.relu1(x)

        x = self.attention_layer(x)
        x = self.fc2(x)

        return x
# 创建模型实例
model = MyModel(num_features=13, num_classes=2)  # 后续训练好的模型和参数保存在这个对象中

# 定义损失函数
loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
#loss = nn.SoftMarginLoss(reduction='mean')

# 定义优化器
optimizer = nn.Adam(model.trainable_params(), learning_rate=0.001)

# 创建WithLossCell
network_with_loss = nn.WithLossCell(model, loss)

# 创建TrainOneStepCell
train_network = nn.TrainOneStepCell(network_with_loss, optimizer)
# 数据预处理和转换
def preprocess_data(features, labels):
    features = features.astype(np.float32)
    labels = labels.astype(np.int32)
    return ms.Tensor(features), ms.Tensor(labels)

# 定义训练过程
def train(train_network, features, labels, epochs):
    features, labels = preprocess_data(features, labels)

    for epoch in range(epochs):
        epoch_loss = 0.0

        # 执行一次训练
        loss = train_network(features, labels)

        epoch_loss = loss.asnumpy()
        #print("Epoch: {}, Loss: {}".format(epoch, epoch_loss))
        
# 假设训练10个epochs
train(train_network, features, labels, epochs=100)

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

#加性注意力
class AttentionLayer(nn.Cell):
    def __init__(self, input_dim, output_dim):  # 修改构造函数，接收输出维度
        super(AttentionLayer, self).__init__()
        self.fc = nn.Dense(input_dim, output_dim)  # 后面训练模型时，创建AttentionLayer实例时，传入参数
        self.softmax = nn.Softmax(axis=1)  #对注意力权重进行归一化，使得注意力权重之和等于1

    def construct(self, x):
        attention_scores = self.fc(x)
        attention_weights = self.softmax(attention_scores)
        attended_features = attention_weights * x
        return attended_features

    
class MyModel(nn.Cell):
    def __init__(self, num_features, num_classes):
        super(MyModel, self).__init__()
        self.fc1 = nn.Dense(num_features, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Dense(64, num_classes)
        self.attention_layer = AttentionLayer(64, 64)  # 修改输出维度为 64

    def construct(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.attention_layer(x)
        x = self.fc2(x)

        return x


# 创建模型实例
model = MyModel(num_features=13, num_classes=2)

# 将测试数据转换为Tensor格式
test_data = ms.Tensor(coffee_test.values, ms.float32)

# 提取特征
test_features = test_data[:, :13]

# 使用训练好的模型进行预测
predictions = model(test_features)

# 将预测结果转换为类别标签
predicted_labels = np.argmax(predictions, axis=1)

# 比较预测类别与真实标签，计算预测正确的样本数
true_label = coffee_test['popular_level'].values
correct_predictions = np.sum(predicted_labels == true_label)
# 计算准确率
accuracy = correct_predictions / len(true_label)

print("Test Accuracy: {:.4f}".format(accuracy))