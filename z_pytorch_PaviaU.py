import random

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
import argparse
from tqdm import tqdm
from zdataset import *
from z_setDataset import *
from z_pytorch_regGabor2DNet import *

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
    args.data_name = 'PU'
    args.dataset_path = 'dataset/'
    args.n_cols = 340  # number of columns
    args.n_rows = 610  # number of rows
    args.n_channels = 103  # number of bands
    args.dim = 15  # patch size
    args.filter_size = 5  # filter size
    args.n_classes = 9  # number of predefined classes
    args.n_perclass = 50  # number of training samples per class
    # [33,100,100,100,100,100,20,100,14,100,100,100,100,100,100,75] 1342
    # [33,50,50,50,50,50,20,50,14,50,50,50,50,50,50,50] 717
    # [33,200,200,181,200,200,20,200,14,200,200,200,143,200,200,75] 2466
    args.n_validate = 50  # number of validation samples per class
    args.mk = 0  # whether make data augmentation
    args.id_set = 1  # the index of training sets
    args.bnum = 2  # number of convolutional blocks

    # data = load_mat_data(args)
    # data['test_x'] = np.vstack((data['test_x'], data['valid_x']))
    # data['test_y'] = np.hstack((data['test_y'], data['valid_y']))
    ###########################################################################

    # Other options
    if args.default_settings:
        args.n_epochs = 150
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
    # np.random.seed(42)
    ###############################
    # 加载数据集
    X, y = loadData(args.data_name)  # 加载数据集
    # print(X.shape, y.shape) #(145, 145, 200) (145, 145)
    X = normalization(X)  # 数据归一化 # print(X)

    len_data = trainall(y, args.n_classes)
    X, y = createImageCubes2(X, y, args.dim, len_data)  # padding图像、提取patch块、清洗数据
    print(X.shape, y.shape, y.shape[0])  # (10249, 15, 15, 200) (10249,) 10249
    # Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.95)    #按比例抽样
    Xtrain, ytrain, Xtest, ytest = Sample_drawn_fixnum(args.n_classes, args.n_perclass, args.n_channels, X,
                                                       y)  # 每类固定一样数量抽样
    print(Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape)
    # (717, 15, 15, 200) (9532, 15, 15, 200) (717,) (9532,)

    # 为了适应 pytorch 结构，数据要做 transpose
    Xtrain = Xtrain.transpose(0, 3, 1, 2)
    Xtest = Xtest.transpose(0, 3, 1, 2)
    print('after transpose: Xtrain shape: ', Xtrain.shape)
    print('after transpose: Xtest  shape: ', Xtest.shape)
    # (717, 200, 15, 15)(9532, 200, 15, 15)

    # 创建 trainloader 和 testloader
    trainset = TrainDS(Xtrain, ytrain)
    testset = TestDS(Xtest, ytest)

    # 从数据库中每次抽出batch size个样本
    # num_workers工作者数量,默认是0.使用多少个子进程来导入数据(多线程来读数据).
    train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=50, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=50, shuffle=False)

    #######################################
    # 训练
    # 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)
    # device = torch.device("cuda")

    # 网络,损失函数,优化器
    # 网络放到GPU上
    net = CNNNet(in_channels=args.n_channels, num_classes=args.n_classes).to(device)
    # net = CNN_modelNet(in_channels=args.n_channels, num_classes=args.n_classes).to(device)
    # summary(net, (30, 15, 15))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0076)
    lr_lambda = lambda epoch: 0.1 ** ((epoch - 1) / 50) if epoch > 1 else 1.
    # lr_lambda = lambda epoch: 0.995 ** epoch
    lr_schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=-1)

    # 模型输入需要的形状[batch,in_channels,h,w]
    # 开始训练
    train_loss = np.zeros(args.n_epochs)

    net.train()
    for epoch in range(args.n_epochs):
        total_loss = 0
        tbar = tqdm(train_loader)
        total_correct = 0.
        count = 0
        for i, (inputs, labels) in enumerate(tbar):  # train_loader训练集
            # enumerate()用于可迭代\可遍历的数据对象组合为一个索引序列，同时列出数据和数据下标.（例如将i和（inputs，labels）组成一个索引序列）
            inputs = inputs.to(device)  # 数据在GPU上运行
            labels = labels.to(device)
            # 优化器梯度归零
            optimizer.zero_grad()  # 梯度归零
            # 正向传播 +　反向传播 + 优化
            outputs = net(inputs)  # 正向传播
            loss = criterion(outputs, labels)  # 损失
            loss.backward()   # 反向传播得到每个参数的梯度
            optimizer.step()  # 通过梯度下降一步参数更新。进行单次优化（一旦梯度被如backward()之类的函数计算好后，我们就可以调用这个函数。）
            # 打印batch级别日记
            total_loss += loss.item()
            total_correct += torch.sum(torch.max(outputs, 1)[1] == labels.data)
            count += labels.shape[0]

            tbar.set_description(f'Stage ({epoch + 1}/{args.n_epochs}) loss: {total_loss / (i + 1)}')
        train_loss_ = total_loss / len(train_loader)
        train_loss[epoch] = train_loss_
        lr_schedule.step()
        curr_lr = round(optimizer.param_groups[0]["lr"], 8)
        print('lr', curr_lr)
        # 打印epoch级别日志
        print("\n[Epoch: %d], [loss: %.8f], [acc: %.4f]" % (
            epoch + 1, train_loss_, total_correct / count))

    print('Finished Training')
    # print(net.state_dict().keys())  # 打印模型参数
    # 保存模型参数
    torch.save(net.state_dict(), "./model/net_parameter_pu_perclass50.pkl")

    #####################################
    # 测试
    # 加载模型参数
    # net = RegGabor2DNet(in_channels=args.n_channels, num_classes=args.n_classes).to(device)
    # net.load_state_dict(torch.load("./model/net_parameter.pkl"))
    net.eval()
    count = 0
    # 模型测试
    for inputs, _ in test_loader:  # test_loader测试集
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = net(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        # detach()阻断反向传播。经过detach()方法后，变量仍然在GPU上，cpu()将数据移至CPU中。numpy()将cpu上的tensor转为numpy数据。
        # numpy.argmax(array, axis) 用于返回一个numpy数组中最大值的[索引值]
        if count == 0:
            y_pred_test = outputs
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))  # np.concatenate是numpy中对array进行拼接的函数

    # 生成分类报告
    print('test accuracy', accuracy_score(ytest, y_pred_test))
    classification = classification_report(ytest, y_pred_test, digits=4)
    print(classification)

    results_file = 'Results/' + 'pu_2D_pytorch_perclass50' + '_img' + '.mat'
    sio.savemat(results_file,
                {'train_loss': train_loss
                 })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="data directory", default='./data')
    parser.add_argument("--default_settings", help="use default settings", type=bool, default=True)
    parser.add_argument("--combine_train_val", help="combine the training and validation sets for testing", type=bool,
                        default=False)
    args = parser.parse_args(args=[])
    main(args)