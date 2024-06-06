import numpy as np
import torch
from torch.utils.data import Dataset
from z_Gabor_Pavia.dataProcess import loadData, window_slides


class SpectrumDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, channel_first=False):
        """
        :param features:
        :param labels:
        :param channel_first: if True, do not need to transform; else transform
        """
        if not channel_first:
            if len(features.shape) == 4:
                features = features.transpose((0, 3, 1, 2))
            elif len(features.shape) == 5:
                features = features.transpose((0, 4, 3, 1, 2))
            else:
                raise ValueError(f'{features.shape} must be 4D or 5D')
        self.features = torch.FloatTensor(features.astype(np.float32))
        self.labels = torch.LongTensor(labels)

    def __getitem__(self, item):
        return self.features[item], self.labels[item]

    def __len__(self):
        return self.features.shape[0]


if __name__ == '__main__':
    a, b = window_slides(*loadData(path_image='../dataset/Indian-pines/Indian_pines_corrected.mat',
                                   path_label='../dataset/Indian-pines/Indian_pines_gt.mat',
                                   key_image='indian_pines_corrected',
                                   key_label='indian_pines_gt'), window_size=21)
    print(np.unique(a[10]), np.unique(b))
    ds = SpectrumDataset(a[:, :, :, :, np.newaxis], b)
    print(ds.__class__.__name__)
    a, b = ds[10]
    print(a.shape, b.shape)
    print(torch.unique(a), torch.unique(b))
    print(len(ds))
