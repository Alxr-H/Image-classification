from operator import truediv

import numpy as np
import torch
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, cohen_kappa_score
from torch import nn

from z_Gabor_Pavia.dataProcess import zeroPadding, window_slides
import json


def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def reports(test_loader, y_test, net: nn.Module, name='IP', device='cpu'):
    net.eval()
    count = 0
    # 模型测试
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        outputs = net(inputs)
        if isinstance(outputs, tuple or list):
            outputs = outputs[0]
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred = outputs
            count = 1
        else:
            y_pred = np.concatenate((y_pred, outputs))

    if name == 'IP':
        target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
            , 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                        'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                        'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                        'Stone-Steel-Towers']
    elif name == 'SA':
        target_names = ['Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 'Fallow_rough_plow',
                        'Fallow_smooth',
                        'Stubble', 'Celery', 'Grapes_untrained', 'Soil_vinyard_develop', 'Corn_senesced_green_weeds',
                        'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk',
                        'Vinyard_untrained', 'Vinyard_vertical_trellis']
    elif name == 'PU':
        target_names = ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']

    classification = classification_report(y_test, y_pred, target_names=target_names, digits=4)
    oa = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred)

    return classification, confusion, oa * 100, each_acc * 100, aa * 100, kappa * 100


def evaluation(test_loader, y_test, net: nn.Module, name='IP', device='cpu',
               save_path="output/classification_report.txt") -> str:
    classification, confusion, oa, each_acc, aa, kappa = reports(test_loader, y_test, net,
                                                                 name, device)
    print('EachA', each_acc)
    print('OA', oa)
    print('AvgA', aa)
    print('kappa', kappa)
    print(classification)
    classification = str(classification)
    confusion = str(confusion)
    with open(save_path, 'w') as x_file:
        x_file.write('\n')
        x_file.write('{} Kappa accuracy (%)'.format(kappa))
        x_file.write('\n')
        x_file.write('{} Overall accuracy (%)'.format(oa))
        x_file.write('\n')
        x_file.write('{} Average accuracy (%)'.format(aa))
        x_file.write('\n')
        x_file.write('{} Each accuracy (%)'.format(each_acc))
        x_file.write('\n')
        x_file.write('\n')
        x_file.write('{}'.format(classification))
        x_file.write('\n')
        x_file.write('{}'.format(confusion))
    with open(save_path[:-4] + '.json', 'w', encoding='utf-8') as f:
        json.dump({
            'EachA': {i + 1: each_acc[i] for i in range(len(each_acc))},
            'OA': oa,
            'AA': aa,
            'Kappa': kappa
        }, f, indent=4)
    print('json saved at ' + save_path[:-4] + '.json')
    # json 读取
    # with open("res.json", 'r', encoding='utf-8') as fw:
    #     injson = json.load(fw)
    # print(injson)
    # print(type(injson))
    return '{} Each accuracy (%)'.format(each_acc) + '\n' + \
           '{} Overall accuracy (%)'.format(oa) + '\n' + \
           '{} Average accuracy (%)'.format(aa) + '\n' + \
           '{} Kappa accuracy (%)'.format(kappa) + '\n' + \
           '{}'.format(classification)


def predict(X, y, patch_size: int, net: nn.Module, dim=2, device='cpu', is_lstm=False):
    assert dim in (2, 3)
    net = net.to(device)
    net.eval()
    height = y.shape[0]
    width = y.shape[1]
    X = zeroPadding(X, patch_size // 2)
    # 逐像素预测类别
    outputs = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            # if int(y[i, j]) == 0:
                # continue
            # else:
            image_patch = X[i:i + patch_size, j:j + patch_size, :]
            '''
            if is_lstm:
                if dim == 2:
                    image_patch = image_patch.reshape(1, image_patch.shape[0], image_patch.shape[1],
                                                      image_patch.shape[2], 1)
                    X_test_image = torch.FloatTensor(image_patch.transpose(0, 3, 4, 1, 2)).to(device)
                else:
                    image_patch = image_patch.reshape(1, image_patch.shape[0], image_patch.shape[1],
                                                      image_patch.shape[2], 1, 1)
                    X_test_image = torch.FloatTensor(image_patch.transpose(0, 4, 5, 3, 1, 2)).to(device)
            else:
            '''
            if dim == 3:
                image_patch = image_patch.reshape(1, image_patch.shape[0], image_patch.shape[1],
                                                  image_patch.shape[2], 1)
                X_test_image = torch.FloatTensor(image_patch.transpose(0, 4, 3, 1, 2)).to(device)
            else:  # dim == 2:
                image_patch = image_patch.reshape(1, image_patch.shape[0], image_patch.shape[1],
                                                  image_patch.shape[2])
                X_test_image = torch.FloatTensor(image_patch.transpose(0, 3, 1, 2)).to(device)
            prediction = net(X_test_image)
            prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
            outputs[i][j] = prediction + 1
        if i % 20 == 0:
            print('... ... row ', i, ' handling ... ...')
    return outputs.astype(np.uint8)


def quick_predict(X, y, patch_size: int, net: nn.Module, dim=2, device='cpu', is_lstm=False,
                  all_process=False, batch_size=512):
    assert dim in (2, 3)
    net = net.to(device)
    net.eval()
    height = y.shape[0]
    width = y.shape[1]
    patch_fea, patch_labels = window_slides(X, y, window_size=patch_size, removeZeroLabels=False)
    # 逐像素预测类别
    outputs = np.zeros((height * width,), dtype=np.uint8)
    count_id = 0
    while count_id < outputs.shape[0]:
        image_patch = patch_fea[count_id: count_id + batch_size, ...]
        if is_lstm:
            if dim == 2:
                image_patch = image_patch.reshape(image_patch.shape[0], image_patch.shape[1], image_patch.shape[2],
                                                  image_patch.shape[3], 1)
                X_test_image = torch.FloatTensor(image_patch.transpose(0, 3, 4, 1, 2)).to(device)
            else:
                image_patch = image_patch.reshape(image_patch.shape[0], image_patch.shape[1],
                                                  image_patch.shape[2], image_patch.shape[3], 1, 1)
                X_test_image = torch.FloatTensor(image_patch.transpose(0, 4, 5, 3, 1, 2)).to(device)
        else:
            if dim == 3:
                image_patch = image_patch.reshape(image_patch.shape[0], image_patch.shape[1],
                                                  image_patch.shape[2], image_patch.shape[3], 1)
                X_test_image = torch.FloatTensor(image_patch.transpose(0, 4, 3, 1, 2)).to(device)
            else:  # dim == 2:
                image_patch = image_patch.reshape(image_patch.shape[0], image_patch.shape[1],
                                                  image_patch.shape[2], image_patch.shape[3])
                X_test_image = torch.FloatTensor(image_patch.transpose(0, 3, 1, 2)).to(device)
        with torch.no_grad():
            prediction = net(X_test_image)
        if isinstance(prediction, tuple or list):
            prediction = prediction[0]
        prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
        if not all_process:
            prediction += 1
        outputs[count_id: count_id + batch_size] = prediction
        count_id += batch_size
        print(count_id, outputs.shape)
    if not all_process:  # 背景赋值0，其它+1
        for i in range(height * width):
            if patch_labels[i] == 0:
                outputs[i] = 0
    return outputs.reshape(height, width)
