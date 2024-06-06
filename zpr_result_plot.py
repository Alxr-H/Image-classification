import argparse
from zdataset import *
from z_pytorch_model import *
from z_pytorch_regGabor2DNet import *


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
    args.n_perclass = [33, 100, 100, 100, 100, 100, 20, 100, 14, 100, 100, 100, 100, 100, 100,
                       75]  # number of training samples per class
    # [33,100,100,100,100,100,20,100,14,100,100,100,100,100,100,75] 1342
    # [33,50,50,50,50,50,20,50,14,50,50,50,50,50,50,50] 717
    # [33,200,200,181,200,200,20,200,14,200,200,200,143,200,200,75] 2466
    args.n_validate = 50  # number of validation samples per class
    args.mk = 0  # whether make data augmentation
    args.id_set = 2  # the index of training sets
    args.bnum = 2  # number of convolutional blocks

    # data = load_mat_data(args)
    # data['test_x'] = np.vstack((data['test_x'], data['valid_x']))
    # data['test_y'] = np.hstack((data['test_y'], data['valid_y']))
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

    ###############################
    # 加载数据集
    X, y = loadData(args.data_name)  # 加载数据集
    X = normalization(X)  # 数据归一化 # print(X)
    X, y = createImageCubes(X, y, windowSize=args.dim)  # padding图像、提取patch块、清洗数据
    Xtrain, ytrain, Xtest, ytest = Sample_drawn_fixnum_list(args.n_classes, args.n_perclass, args.n_channels, X,
                                                            y)  # 每类固定不一样数量抽样

    # 为了适应 pytorch 结构，数据要做 transpose
    Xtrain = Xtrain.transpose(0, 3, 1, 2)
    Xtest = Xtest.transpose(0, 3, 1, 2)
    X = X.transpose(0, 3, 1, 2)
    print('after transpose: Xtrain shape: ', Xtrain.shape)
    print('after transpose: Xtest  shape: ', Xtest.shape)
    print('after transpose: Xtest  shape: ', X.shape)
    # (717, 200, 15, 15)(9532, 200, 15, 15)(10249, 200, 15, 15)

    # 创建 trainloader 和 testloader
    trainset = TrainDS(Xtrain, ytrain)
    testset = TestDS(Xtest, ytest)
    predset = PreDS(X, y)

    # 从数据库中每次抽出batch size个样本
    train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=64, shuffle=False)
    pred_loader = torch.utils.data.DataLoader(dataset=predset, batch_size=64, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = RegGabor2DNet(in_channels=args.n_channels, num_classes=args.n_classes).to(device)
    # summary(net, (30, 15, 15))

    #####################################
    # 预测
    # 加载模型参数
    net.load_state_dict(torch.load("./model/net_parameter.pkl"))
    net.eval()
    # 预测
    # 预测结果
    count = 0
    # 模型测试
    for inputs, _ in pred_loader:  # test_loader测试集
        inputs = inputs.to(device)

        outputs = net(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        # detach()阻断反向传播。经过detach()方法后，变量仍然在GPU上，cpu()将数据移至CPU中。numpy()将cpu上的tensor转为numpy数据。
        # numpy.argmax(array, axis) 用于返回一个numpy数组中最大值的[索引值]
        if count == 0:
            y_pred = outputs
            count = 1
        else:
            y_pred = np.concatenate((y_pred, outputs))  # np.concatenate是numpy中对array进行拼接的函数

    # 生成分类报告
    print('pre accuracy', accuracy_score(y, y_pred))
    classification = classification_report(y, y_pred, digits=4)
    print(classification)

    ########################
    # 将预测结果匹配到图像中
    data, labels = loadData(args.data_name)
    print(labels.shape)  # (145,145),145*145=21025(包含零元素)
    # 将预测的结果匹配到图像中
    output_final = np.zeros((labels.shape[0], labels.shape[1]))
    k = 0
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if labels[i][j] != 0:
                output_final[i][j] = y_pred[k] + 1
                k += 1
    print(output_final.shape)  # (145,145)

    plt.figure(dpi=500, figsize=(7, 7))
    plt.imshow(output_final, cmap='gist_ncar')
    plt.colorbar()
    plt.axis('off')
    # plt.savefig("dataset/111.tif")
    plt.pause(6000)
    '''
    results_file = 'Results/' + 'indian2D_pytorch' + '_img' + '.mat'
    sio.savemat(results_file,
                {'output_final': output_final,  # 'train_loss': train_loss
                 })
    '''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="data directory", default='./data')
    parser.add_argument("--default_settings", help="use default settings", type=bool, default=True)
    parser.add_argument("--combine_train_val", help="combine the training and validation sets for testing", type=bool,
                        default=False)
    args = parser.parse_args(args=[])
    main(args)
