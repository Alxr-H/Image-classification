import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy.io as sio
import os


# 加载数据集的函数
def loadData(name):
    data_path = os.path.join(os.getcwd(), './dataset')
    if name == 'IP':
        data = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
    elif name == 'SA':
        data = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
    elif name == 'PU':
        data = sio.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']

    return data, labels


def normalization(X):
    """
    Normalization: make data values within [-1,1]
    """

    img_add = X.astype('float32')
    for i in range(X.shape[2]):  # X.shape[2]=args.n_channels #对每一光谱维的数据进行操作：归一化
        img_add[:, :, i] = img_add[:, :, i] - img_add[:, :, i].min()
        img_add[:, :, i] = img_add[:, :, i] / img_add[:, :, i].max()
        img_add[:, :, i] = img_add[:, :, i] * 2 - 1
    return img_add
    # end of normalization


## padding操作：对单个像素周围提取 patch 时，边缘像素就无法取了，因此给这部分像素进行padding操作
def padWithZeros(X, margin=7):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))  ##填充四个像素，因为此时窗口大小为15
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X  ##填充数据，nexX[：]进行切片
    return newX


## patch操作：
def createImageCubes(X, y, windowSize=15, removeZeroLabels=True):
    # 给X做padding
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)

    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))  ##规定数据样本、标签形状
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0  # 编号
    for r in range(margin, zeroPaddedX.shape[0] - margin):  ##进行数据填充
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:  ##清理数据
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1
    return patchesData, patchesLabels


def createImageCubes2(X, y, windowSize=15, data_len=10249):
    # 给X做padding
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)

    # split patches
    patchesData = np.zeros((data_len, windowSize, windowSize, X.shape[2]))  ##规定数据样本、标签形状
    patchesLabels = np.zeros(data_len)
    patchIndex = 0  # 编号
    for r in range(margin, zeroPaddedX.shape[0] - margin):  ##进行数据填充
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            if y[r - margin, c - margin] != 0:
                patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
                patchesData[patchIndex, :, :, :] = patch
                patchesLabels[patchIndex] = y[r - margin, c - margin]
                patchIndex = patchIndex + 1
    patchesLabels -= 1
    # print(patchIndex)
    return patchesData, patchesLabels


# 统计非零元素个数
def trainall(y, n_classes):
    TruthMap1D = y.reshape(1, -1)
    len_data = 0
    for i in range(n_classes):
        lableoneclass_x, lableoneclass_y = np.where(TruthMap1D == i + 1)
        len_data = len_data + len(lableoneclass_y)

    return len_data


def minibatcher(inputs, targets, batchsize, shuffle=False):
    """
   Generate batches.

   :param inputs: features
   :param targets: labels
   :param batchsize: the size of each batch
   :param shuffle: whether shuffle the batch.
   """

    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
    # end of minibatcher


def Sample_drawn_fixnum(n_classes, n_perclass, n_channels, feature, labels):
    """
       按固定数目生成训练集、测试集.例如每类取100个

       :param n_classes: number of predefined classes
       :param n_perclass: number of training samples per class
       :param n_channels: number of bands
       :param feature: 特征.(10249, 15, 15, 200)
       :param labels: 标签.(10249,)
       """

    samples_per_class = n_perclass  # 每一类取samples_per_class个样本，16类，即共samples_per_class*16 训练样本
    num_class = n_classes  # num_class表示类别数
    train_samples = samples_per_class * num_class  # 训练样本个数
    train_f = np.zeros((train_samples, 15, 15, n_channels))  # train_f=（n*15*15*200）
    train_labels = np.zeros(train_samples)  # train_labels=（n，1）
    test_f = np.zeros((labels.shape[0] - train_samples, 15, 15, n_channels))  # test_f（）
    test_labels = np.zeros(labels.shape[0] - train_samples)  # test_labels（）

    p = 0
    q = 0
    for i in range(num_class):  # num_class=16,i=[0-15]
        index = np.reshape(np.array(np.where(np.reshape(labels, (-1)) == i)),
                           (-1))  # 将矩阵labels变为行数为（labels=10249）行  #index为：标签为i的位置，比如labels中有哪些样本标签为1，返回那些样本的位置
        # print(index.shape)——(46,)(1428,)(830,)(237,)(483,)(730,)(28,)(478,)(20,)(972,)(2455,)(593,)(205,)(1265,)(386,)(93,)
        random_num = np.random.choice(index.__len__(), samples_per_class,
                                      replace=False)  # np.random.choice(a, size, replace)从数组中随机抽取元素：从a(只要是ndarray都可以，但必须是一维的)中随机抽取数字，并组成指定大小(size)的数组 #index.__len__()——Return len(self)，samples_per_class=6 ##replace:True表示可以取相同数字，False表示不可以取相同数字
        # 例如此时样本标签0，有46个样本标签为0，index.__len__()=46，np.random.choice(46，6, replace=False)从[0，46）中输出6个数字并组成一维数组，这个值可以看作【索引】
        # print(random_num)——[ 0 20 34  4  2 23]...
        for j in range(index.__len__()):
            if np.reshape(np.array(np.where(random_num == j)), (-1)).shape[0] == 0:  # a.shape[0]表示矩阵的行数 #此时标签范围为0-15，j为样本i的长度 #np.where若没有符合条件的返回空
                test_f[p] = feature[index[j]]  # 测试集
                test_labels[p] = labels[index[j]]
                p += 1
            else:  # 训练集
                # print(np.where(random_num == j))——(array([3]),)...
                train_f[q] = feature[index[j]]
                train_labels[q] = labels[index[j]]
                q += 1
    return train_f, train_labels, test_f, test_labels


def Sample_drawn_fixnum_list(n_classes, n_perclass, n_channels, feature, labels):
    """
       按固定数目生成训练集、测试集.例如每类取不一样的样本数量

       :param n_classes: number of predefined classes
       :param n_perclass: [array] the fixed numbers of training samples for each class
       :param n_channels: number of bands
       :param feature: 特征.(10249, 15, 15, 200)
       :param labels: 标签.(10249,)
       """

    samples_per_class = n_perclass  # 每一类取samples_per_class个样本，16类，即共samples_per_class*16 训练样本
    num_class = n_classes  # num_class表示类别数

    train_samples = 0
    for k in range(0, len(samples_per_class)):
        train_samples += samples_per_class[k]
    # train_samples = samples_per_class * num_class  # 训练样本个数
    train_f = np.zeros((train_samples, 15, 15, n_channels))  # train_f=（n*15*15*200）
    train_labels = np.zeros(train_samples)  # train_labels=（n，1）
    test_f = np.zeros((labels.shape[0] - train_samples, 15, 15, n_channels))  # test_f（）
    test_labels = np.zeros(labels.shape[0] - train_samples)  # test_labels（）

    p = 0
    q = 0
    for i in range(num_class):
        # fetch all the samples belonging to class i
        index = np.reshape(np.array(np.where(np.reshape(labels, (-1)) == i)),
                           (-1))  # 将矩阵labels变为行数为（labels=10249）行  #index为：标签为i的位置，比如labels中有哪些样本标签为1，返回那些样本的位置
        # print(index.shape)——(46,)(1428,)(830,)(237,)(483,)(730,)(28,)(478,)(20,)(972,)(2455,)(593,)(205,)(1265,)(386,)(93,)
        random_num = np.random.choice(index.__len__(), samples_per_class[i],
                                      replace=False)  # np.random.choice(a, size, replace)从数组中随机抽取元素：从a(只要是ndarray都可以，但必须是一维的)中随机抽取数字，并组成指定大小(size)的数组 #index.__len__()——Return len(self)，samples_per_class=6 ##replace:True表示可以取相同数字，False表示不可以取相同数字
        # 例如此时样本标签0，有46个样本标签为0，index.__len__()=46，np.random.choice(46，6, replace=False)从[0，46）中输出6个数字并组成一维数组，这个值可以看作【索引】
        # print(random_num)——[ 0 20 34  4  2 23]...
        for j in range(index.__len__()):
            if np.reshape(np.array(np.where(random_num == j)), (-1)).shape[
                0] == 0:  # a.shape[0]表示矩阵的行数 #此时标签范围为0-15，j为样本i的长度 #np.where若没有符合条件的返回空
                test_f[p] = feature[index[j]]  # 测试集
                test_labels[p] = labels[index[j]]
                p += 1
            else:  # 训练集
                # print(np.where(random_num == j))——(array([3]),)...
                train_f[q] = feature[index[j]]
                train_labels[q] = labels[index[j]]
                q += 1
    return train_f, train_labels, test_f, test_labels


# PCA降维
def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX, pca


if __name__ == '__main__':
    X, y = loadData('PU')
    len_data = trainall(y, 9)
    print(len_data)
    X, y = createImageCubes2(X, y, 15, len_data)  # padding图像、提取patch块、清洗数据
    Xtrain, ytrain, Xtest, ytest = Sample_drawn_fixnum(9, 100, 103, X, y)  # 每类固定不一样数量抽样
