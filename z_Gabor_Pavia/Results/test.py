from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import torch
from torch import nn, optim
import os
import PIL
import PIL.Image
from datetime import datetime
from sklearn.decomposition import PCA
from scipy.io import loadmat
import pandas as pd
import time
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from zdataset import *
from z_setDataset import *
from z_pytorch_regGabor2DNet import *
from z_Gabor_Pavia.lightSpectrumDataset import splitTrainTestIndex, LightSpectrumDataset
from z_Gabor_Pavia.z_seedInitializer import randomSeedInitial

'''
def show_batch():
    for step, (batch_x, batch_y) in enumerate(train_loader):
        print("steop:{}, batch_x:{}, batch_y:{}".format(step, batch_x, batch_y))
'''
import datetime
import logging
import os
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from z_Gabor_Pavia.lightSpectrumDataset import splitTrainTestIndex, LightSpectrumDataset
from z_Gabor_Pavia.spectrumDataset import SpectrumDataset
from z_Gabor_Pavia.z_pytorch_regGabor2DNet import RegGabor2DNet
from z_Gabor_Pavia.dataProcess import loadData, window_slides, splitTrainTestSet
from z_Gabor_Pavia.evaluation import evaluation
from z_Gabor_Pavia.z_seedInitializer import randomSeedInitial


def main(ds_name='PiaviaU'):
    """
    :param randomSeed: 1 or non-int: random seed
    :param ds_name:
    :return:
    """
    tria_start_time = time.time()
    push_info = f'Exp start at {datetime.datetime.now()}\n'

    '''
    if isinstance(randomSeed, int) and randomSeed != 1:
        seed = randomSeed
        randomSeedInitial(seed)
    else:
        # seed = random.randint(0, 300)
        seed = int(time.time())
    print('seed:', seed)
    '''

    use_cuda = True
    net_dim = 2
    w_size = 15
    batch_size = 50
    if ds_name is not None and ds_name in ('Indian_pines', 'PiaviaU'):
        dataset_name = ds_name
    else:
        dataset_name = 'Indian_pines'
    dataset_params = {
        'Indian_pines': {'name': ['./dataset/Indian_pines_corrected.mat',
                                  './dataset/Indian_pines_gt.mat',
                                  'indian_pines_corrected', 'indian_pines_gt'],
                         'nick_name': 'IP',
                         'bands': 200,
                         'num_classes': 16,
                         'epochs': 300,
                         'use_pca': False},
        'PiaviaU': {'name': ['/dataset/PaviaU.mat',
                             '/dataset/PaviaU_gt.mat',
                             'paviaU', 'paviaU_gt'],
                    'nick_name': 'PU',
                    'bands': 103,
                    'num_classes': 9,
                    'epochs': 300,
                    'use_pca': False}
    }
    num_classes = dataset_params[dataset_name]['num_classes']
    n_bands = dataset_params[dataset_name]['bands']
    device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")
    use_pca = dataset_params[dataset_name]['use_pca']
    pca_components = 30
    expID = f'K{pca_components if use_pca else n_bands}'

    net = RegGabor2DNet(in_channels=pca_components if use_pca else n_bands, num_classes=num_classes).to(device)
    extra_info = 'common'
    if extra_info is None or len(extra_info) == 0:
        extra_info = 'common'
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(net.parameters(), lr=0.0001)
    optimizer = optim.Adam(net.parameters(), lr=0.0076)
    # lr_lambda = lambda epoch: 0.1 ** ((epoch - 1) / 50) if epoch > 1 else 1.
    lr_lambda = lambda epoch: 0.995 ** epoch
    lr_schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=-1)
    if dataset_name == 'Indian_pines':
        exp_sample_amount_dict = {
            10: [10 for _ in range(num_classes)],
            50: [33, 50, 50, 50, 50, 50, 20, 50, 14, 50, 50, 50, 50, 50, 50, 50],
            100: [33, 100, 100, 100, 100, 100, 20, 100, 14, 100, 100, 100, 100, 100, 100, 75],
            200: [33, 200, 200, 181, 200, 200, 20, 200, 14, 200, 200, 200, 143, 200, 200, 75]
        }
    else:
        exp_sample_amount_dict = {
            10: [10 for _ in range(num_classes)],
            20: [20 for _ in range(num_classes)],
            30: [30 for _ in range(num_classes)],
            50: [50 for _ in range(num_classes)],
            100: [100 for _ in range(num_classes)],
            200: [200 for _ in range(num_classes)]
        }
    exp_sample_amount = 100
    is_normalize = True

    if (not use_pca and w_size > 15) or (use_pca and pca_components > 30) or (
            not use_pca and dataset_name == 'PiaviaU'):
        print('RAM Light...')
        fea, labels = loadData(*dataset_params[dataset_name]['name'], use_pca=use_pca, pca_components=pca_components,
                               normalize=is_normalize)
        train_index, test_index = splitTrainTestIndex(labels, trainNumList=exp_sample_amount_dict[exp_sample_amount],
                                                      removeZeroLabels=True)
        train_loader = LightSpectrumDataset(fea, labels, index=train_index, window_size=w_size, use_lstm=False,
                                            dim=net_dim, removeZeroLabels=True)
        test_loader = LightSpectrumDataset(fea, labels, index=test_index, window_size=w_size, use_lstm=False,
                                           dim=net_dim, removeZeroLabels=True)
        y_test = test_loader.getY()
    else:
        print('All data in RAM...')
        X_train, X_test, y_train, y_test = splitTrainTestSet(
            *window_slides(
                *loadData(*dataset_params[dataset_name]['name'], use_pca=use_pca, pca_components=pca_components,
                          normalize=is_normalize),
                window_size=w_size),
            trainNumList=exp_sample_amount_dict[exp_sample_amount], normalize=False)
        if net_dim == 3:
            X_train = X_train[:, :, :, :, np.newaxis]
            X_test = X_test[:, :, :, :, np.newaxis]

        print('Xtrain shape: ', X_train.shape, X_train.max(), X_train.min())
        print('Xtest  shape: ', X_test.shape, X_test.max(), X_test.min())
        print(np.unique(y_test))
        train_loader = SpectrumDataset(X_train, y_train, channel_first=False)
        test_loader = SpectrumDataset(X_test, y_test, channel_first=False)
    dataset_type = train_loader.__class__.__name__
    train_loader = DataLoader(dataset=train_loader, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(dataset=test_loader, batch_size=batch_size * 4, shuffle=False, pin_memory=True,
                             drop_last=False)
    '''
    if is_normalize:
        save_dir = os.path.join('output', dataset_name, net.__class__.__name__, 'nt' + str(exp_sample_amount),
                                'ws' + str(w_size),
                                'expID' + str(expID), 'normalize',
                                f'seed{seed}', extra_info + dataset_type)
    else:
        save_dir = os.path.join('output', dataset_name, net.__class__.__name__, 'nt' + str(exp_sample_amount),
                                'ws' + str(w_size),
                                'expID' + str(expID), 'non_normalize'
                                                      f'seed{seed}', extra_info + dataset_type)
    
    writer = SummaryWriter(save_dir)
    logging.basicConfig(filename=os.path.join(save_dir, 'log.txt'), level=logging.INFO,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
    print('save_dir: %s' % save_dir)
    '''
    epochs = dataset_params[dataset_name]['epochs']
    for epoch in range(epochs):
        net.train()
        total_loss = 0
        total_correct = 0.
        tbar = tqdm(train_loader)
        count = 0
        for i, (inputs, labels) in enumerate(tbar):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # 优化器梯度归零
            optimizer.zero_grad()
            # 正向传播 +　反向传播 + 优化
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            # print(outputs, labels)
            # input('************')

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_correct += torch.sum(torch.max(outputs, 1)[1] == labels.data)
            count += labels.shape[0]

            tbar.set_description(desc='  [Epoch: %d / %d]   [loss avg: %.8f]   [acc: %.2f]' % (
                net.__class__.__name__ + dataset_name + str(
                    pca_components if use_pca else n_bands
                ) + epoch + 1, epochs, total_loss / (i + 1),
                total_correct / count))
        lr_schedule.step()
        curr_lr = round(optimizer.param_groups[0]["lr"], 8)
        print('lr', curr_lr)
        '''
        writer.add_scalar(tag='train loss', scalar_value=total_loss / len(train_loader), global_step=epoch + 1)
        writer.add_scalar(tag='train acc', scalar_value=total_correct / count, global_step=epoch + 1)
        logging.info(f'lr: {curr_lr}, tr_loss: {total_loss / len(train_loader)}, tr_acc: {total_correct / count}')
        '''
    ##################################################
    # 模型测试
    tbar = tqdm(test_loader)
    count = 0
    # 模型测试
    net.eval()
    for inputs, _ in tbar:
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = net(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs), axis=0)
    # 生成分类报告
    print('test accuracy', accuracy_score(y_test, y_pred_test))
    logging.info(f'test accuracy: {accuracy_score(y_test, y_pred_test)}')

    '''
    print('Finished Training', 'seed=%s' % seed)
    torch.save(net.state_dict(), os.path.join(save_dir, f'{net.__class__.__name__}_params.pkl'))
    torch.save(net, os.path.join(save_dir, f'{net.__class__.__name__}.pkl'))
    print('### evaluation for test...')
    cost_time = time.time() - tria_start_time
    push_info += evaluation(test_loader, y_test, net, name=dataset_params[dataset_name]['nick_name'], device=device,
                            save_path=os.path.join(save_dir, f'classification_report.txt'))
    torch.cuda.empty_cache()
    writer.close()
    push_info += f'\n#info saved in {save_dir}'
    push_info += f'Exp end at {datetime.datetime.now()}'
    # send_wechat(msg=push_info, title=net.__class__.__name__ + dataset_params[dataset_name]['nick_name'] + str(seed)
    # + 'time' + str(int(cost_time)))
    del net
    '''

if __name__ == '__main__':
    main(ds_name='PiaviaU')
