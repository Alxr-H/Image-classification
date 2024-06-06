import torch
from torch.utils.data import Dataset


""" Training dataset"""


class TrainDS(Dataset):
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


class TestDS(Dataset):
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


class PreDS(Dataset):
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


