import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

"""
code reference: https://blog.csdn.net/hahahd3/article/details/112199724
"""


def loadData(path_image='dataset/Indian-pines/Indian_pines_corrected.mat',
             path_label='dataset/Indian-pines/Indian_pines_gt.mat',
             key_image='indian_pines_corrected',
             key_label='indian_pines_gt',
             use_pca=False, pca_components=5, normalize=False):  # h, w, c
    # mat = loadmat(path_image)
    # print(mat.keys())
    features = loadmat(path_image)[key_image]

    # mat_labels = loadmat(path_label)
    # print(mat_labels.keys())
    labels = loadmat(path_label)[key_label]
    if use_pca:
        features = applyPCA(features, numComponents=pca_components)
    if normalize:
        min_val = np.min(features, axis=(0, 1))
        max_val = np.max(features, axis=(0, 1))
        features = (features - min_val) / (max_val - min_val) * 2. - 1.
    return features.astype(np.float32), labels


def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX.astype(np.float32)


def zeroPadding(X: np.ndarray, margin=2):
    """
    channel_last
    :param X:
    :param margin:
    :return:
    """
    shapeX = X.shape
    newX = np.zeros((shapeX[0] + 2 * margin, shapeX[1] + 2 * margin, shapeX[2]), dtype=X.dtype)
    newX[margin: shapeX[0] + margin, margin: shapeX[1] + margin, :] = X
    return newX


def window_slides(features: np.ndarray, labels: np.ndarray, window_size=9, removeZeroLabels=True):
    """
    channel_last
    :param removeZeroLabels:
    :param features:
    :param labels:
    :param window_size:
    :return:
    """
    h, w = labels.shape
    margin = (window_size - 1) // 2
    features = zeroPadding(X=features, margin=margin)
    patch_fea = np.zeros(shape=(h * w, window_size, window_size, features.shape[-1]), dtype=features.dtype)
    for i in range(h):
        for j in range(w):
            patch_fea[i * w + j, :, :, :] = features[i: i + margin * 2 + 1, j: j + margin * 2 + 1, :]
    patch_labels = labels.reshape((h * w,))
    if removeZeroLabels:
        patch_fea = patch_fea[patch_labels > 0, :, :, :]
        patch_labels = patch_labels[patch_labels > 0]
        patch_labels -= 1
    return patch_fea, patch_labels


def randomSplitTrainTestSet(X, y, testRatio=0.25, random_state=256):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=random_state,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test


def splitTrainTestSet(X: np.numarray, y: np.numarray, trainNumList: list = None, normalize=False, return_index=False):
    if trainNumList is None:
        trainNumList = [33, 50, 50, 50, 50, 50, 20, 50, 14, 50, 50, 50, 50, 50, 50, 50]  # 50
        # [33, 100, 100, 100, 100, 100, 20, 100, 14, 100, 100, 100, 100, 100, 100, 75]  # 100
        # [33, 200, 200, 181, 200, 200, 20, 200, 14, 200, 200, 200, 143, 200, 200, 75]  # 200
    trainNum = sum(trainNumList)
    total = X.shape[0]
    print('trainNum:', trainNum, 'testNum:', total - trainNum)
    X_train = np.zeros(shape=(trainNum,) + X.shape[1:], dtype=X.dtype)
    X_test = np.zeros(shape=(total - trainNum,) + X.shape[1:], dtype=X.dtype)
    y_train = np.zeros(shape=(trainNum,) + y.shape[1:], dtype=y.dtype)
    y_test = np.zeros(shape=(total - trainNum,) + y.shape[1:], dtype=y.dtype)
    item_train = 0
    item_test = 0
    tmp_train = []
    tmp_test = []
    for i in range(len(trainNumList)):
        index = np.where(y == i)
        if isinstance(index, tuple):
            index = index[0]
        index_train = np.random.choice(index, size=trainNumList[i], replace=False)
        index_test = np.setdiff1d(index, index_train)
        tmp_train.append(index_train)
        tmp_test.append(index_test)
        X_train[item_train: item_train + trainNumList[i]] = X[index_train]
        X_test[item_test: item_test + index_test.shape[0]] = X[index_test]
        y_train[item_train: item_train + trainNumList[i]] = y[index_train]
        y_test[item_test: item_test + index_test.shape[0]] = y[index_test]
        item_train += trainNumList[i]
        item_test += index_test.shape[0]
    if normalize:
        min_val = np.min(X_train, axis=(0, 1, 2))
        max_val = np.max(X_train, axis=(0, 1, 2))
        X_train = (X_train - min_val) / (max_val - min_val) * 2. - 1.
        X_test = (X_test - min_val) / (max_val - min_val) * 2. - 1.
    if return_index:
        return X_train, X_test, y_train, y_test, tmp_train, tmp_test
    else:
        return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    a, b = loadData(path_image='../dataset/Indian-pines/Indian_pines_corrected.mat',
                    path_label='../dataset/Indian-pines/Indian_pines_gt.mat',
                    key_image='indian_pines_corrected',
                    key_label='indian_pines_gt')
    print(np.unique(a), np.unique(b))
    print(a.dtype, b.dtype)
    print(a.shape, b.shape)
