from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import torch
from torch import nn, optim
import os
import PIL
import PIL.Image
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy.io import loadmat

""" Training dataset"""


class TrainDS(torch.utils.data.Dataset):
    # Dataset 是一个数据集抽象类，它是其他所有数据集类的父类（所有其他数据集类都应该继承它），继承时需要重写方法 __len__ 和 __getitem__
    # __len__ 是提供数据集大小的方法， __getitem__ 是可以通过索引号找到数据的方法
    def __init__(self, Xtrain, ytrain):
        self.len = Xtrain.shape[0]  # 样本数量
        self.x_data = torch.FloatTensor(Xtrain)  # torch.FloatTensor生成32位浮点类型张量
        self.y_data = torch.LongTensor(ytrain)  # torch.LongTensor生成64位整型张量

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len


""" Testing dataset"""


class TestDS(torch.utils.data.Dataset):
    def __init__(self, Xtest, ytest):
        self.len = Xtest.shape[0]  # 样本数量
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len


""" Pre dataset"""


class PreDS(torch.utils.data.Dataset):
    def __init__(self, Xpre, ypre):
        self.len = Xpre.shape[0]  # 样本数量
        self.x_data = torch.FloatTensor(Xpre)
        self.y_data = torch.LongTensor(ypre)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len


# 创建模型
class HybridSN(nn.Module):
    def __init__(self):
        super(HybridSN, self).__init__()
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(7, 3, 3), stride=1, padding=0),  # 输入图像的通道数是1，输出图像的通道数是8
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
        )
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=(5, 3, 3), stride=1, padding=0),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=0),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )

        self.conv2d_4 = nn.Sequential(
            nn.Conv2d(576, 64, kernel_size=(3, 3), stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.fc1 = nn.Linear(18496, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 16)
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        out = self.conv3d_1(x)
        out = self.conv3d_2(out)
        out = self.conv3d_3(out)
        out = self.conv2d_4(out.reshape(out.shape[0], -1, 19, 19))
        out = out.reshape(out.shape[0], -1)
        # out = F.relu(self.dropout(self.fc1(out)))
        out = self.dropout(self.fc1(out))
        out = out.relu()
        # out = F.relu(self.dropout(self.fc2(out)))
        out = self.dropout(self.fc2(out))
        out = out.relu()
        out = self.fc3(out)
        return out
