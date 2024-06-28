import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from torch.nn.functional import softmax

# 创建一些示例数据，这里使用 PyTorch 张量
x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = torch.tensor([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])

# 将数据分割为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 从原始数据中提取训练集和测试集


print(x_train, y_train)
print(x_test, y_test)


from torch.utils.data import TensorDataset

# 创建一些示例数据
data1 = torch.tensor([[1, 2], [3, 4], [5, 6]])
data2 = torch.tensor([[10, 20], [30, 40], [50, 60]])
labels = torch.tensor([0, 1, 2])

# 创建 TensorDataset 对象
dataset = TensorDataset(data1, labels)

# 打印 TensorDataset 的返回值形式
print(dataset[1])


from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return index, x, y

# 假设已经有了数据（X）和标签（y）
import numpy as np

# 创建一些样本特征（X）和标签（y）
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
y = np.array([2, 5, 8, 11])

print("特征 (X):", X)
print("标签 (y):", y)

dataset = MyDataset(X, y)
data_loader = DataLoader(dataset, batch_size=2, drop_last=True)
print(dataset)
print(data_loader)

for i,(idx, x, y) in enumerate(data_loader):
    x_np = x.numpy()
    y_np = y.numpy()
    print("批次 {}:".format(i+1))
    print("索引:", idx)
    print("特征 (x):", x_np)
    print("标签 (y):", y_np)


# 1. 数据预处理
'''
class GTZANDataset(Dataset):
    def __init__(self, data_path, labels, transform=None):
        self.data_path = data_path
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        data = extract_features(self.data_path[idx])  # 你需要定义extract_features方法
        label = self.labels[idx]

        if self.transform:
            data = self.transform(data)

        return data, label
'''