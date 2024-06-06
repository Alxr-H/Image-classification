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
from z_pytorch_model import *
from z_pytorch_regGabor2DNet import *
from z_Gabor_Pavia.lightSpectrumDataset import splitTrainTestIndex, LightSpectrumDataset, getpredictindex
from z_Gabor_Pavia.z_seedInitializer import randomSeedInitial
from z_Gabor_Pavia.evaluation import predict

'''
def show_batch():
    for step, (batch_x, batch_y) in enumerate(train_loader):
        print("steop:{}, batch_x:{}, batch_y:{}".format(step, batch_x, batch_y))
'''


# 建立文件夹
def add_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
        print('Created {:s}'.format(folder_name))
    return folder_name
    # end of add_folder


# 基础参数的设置
def settings(args):
    args.data_name = 'IP'
    args.dataset_path = 'dataset/'
    args.n_cols = 145  # number of columns
    args.n_rows = 145  # number of rows
    args.n_channels = 200  # number of bands
    args.dim = 15  # patch size
    args.filter_size = 5  # filter size
    args.n_classes = 16  # number of predefined classes
    args.n_perclass = [33, 50, 50, 50, 50, 50, 20, 50, 14, 50, 50, 50, 50, 50, 50,
                       50]  # number of training samples per class
    # [33,100,100,100,100,100,20,100,14,100,100,100,100,100,100,75] 1342
    # [33,50,50,50,50,50,20,50,14,50,50,50,50,50,50,50] 717
    # [33,200,200,181,200,200,20,200,14,200,200,200,143,200,200,75] 2466
    args.n_validate = 50  # number of validation samples per class
    args.mk = 0  # whether make data augmentation
    args.id_set = 2  # the index of training sets
    args.bnum = 2  # number of convolutional blocks
    ###########################################################################

    # Other options
    if args.default_settings:
        args.n_epochs = 100
        args.batch_size = 50
        args.learning_rate = 0.0076
        args.std_mult = 0.4
        args.delay = 12
        args.n_theta = 4
        args.n_omega = 4
        args.gains = 2
        args.lr_div = 10
        args.even_initial = True

    # options for naming
    if args.mk:
        args.name_aug = '_Aug'
    else:
        args.name_aug = '_NoAug'
    if args.even_initial:
        args.name_init = ''
    else:
        args.name_init = '_RandInit'

    check_file_name = args.data_name + args.name_aug + args.name_init + '_P' + str(args.dim) + \
                      '_t' + str(args.n_perclass) + '_epo' + str(args.n_epochs) + '_model_' + str(args.id_set)
    args.log_path = add_folder('./logs')
    args.checkpoint_path = add_folder('./checkpoints') + '/' + check_file_name + '.ckpt'
    args.checkpoint_path_valid = add_folder('./checkpoints') + '/' + check_file_name + '_valid.ckpt'
    args.checkpoint_path_loss = add_folder('./checkpoints') + '/' + check_file_name + '_loss.ckpt'
    return args
    # end of settings


def main(args):
    ##### SETUP AND LOAD DATA #####
    args = settings(args)
    # 加载数据集
    X, y = loadData(args.data_name)
    X = normalization(X)

    net_dim = 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = RegGabor2DNet(in_channels=args.n_channels, num_classes=args.n_classes)
    net.load_state_dict(torch.load("./model/net_parameter.pkl"))

    output = predict(X, y, args.dim, net, net_dim, device, is_lstm=False)
    print(output)

    results_file = 'Results/' + 'salinas2D_pytorch' + '_img' + '.mat'
    sio.savemat(results_file,
                {'output': output
                 })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="data directory", default='./data')
    parser.add_argument("--default_settings", help="use default settings", type=bool, default=True)
    parser.add_argument("--combine_train_val", help="combine the training and validation sets for testing", type=bool,
                        default=False)
    args = parser.parse_args(args=[])
    main(args)
