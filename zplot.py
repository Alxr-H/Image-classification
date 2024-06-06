from scipy.io import loadmat  # 用于加载mat文件
import numpy as np
from matplotlib import pyplot as plt
import spectral

'''
mat= loadmat("./dataset/PaviaU.mat")#loadmat方法加载数据后会返回一个Python字典的数据结构
labels= loadmat('./dataset/PaviaU_gt.mat')['paviaU_gt']#标签数据

features= mat['paviaU']
features_shape = features.shape  #(145, 145, 200)
input_size = features_shape[2]  #200

print(features.shape)  #(145, 145)
## print(np.unique(np.reshape(labels,(-1))))  #[0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16]
labels = np.reshape(labels, (-1, 1))  #reshape(-1,1)转换成1列 #labels.shape=(21025, 1)


#每类取50个样本，epoch=100
result_G = loadmat("./Results/PU_Reg2DGabor_NoAug_SaveBest_lr_auto_ks_5_Patch_15_t50_v50_perC_bs50_epo150_1_mywrite.mat")
print(result_G.keys())
result_C = loadmat("./Results/PU_Reg2DGabor_NoAug_SaveBest_lr_auto_ks_5_Patch_15_t50_v50_perC_bs50_epo150_1_mywrite_cnn.mat")

train_acc_G = result_G['train_acc'][-1]
train_loss_G = result_G['train_loss'][-1]
print(train_acc_G.shape)
train_acc_C = result_C['train_acc'][-1]
train_loss_C = result_C['train_loss'][-1]
#print(result)

#Train_acc
plt.plot(train_acc_G, label='GaborNet')
plt.plot(train_acc_C, label='CNN')
plt.xlabel('epoch')
plt.ylabel('train_acc')
plt.ylim([0.0, 1.0])
plt.legend(loc='lower right')
plt.show()
#Train_loss
plt.plot(train_loss_G, label='GaborNet')
plt.plot(train_loss_C, label='CNN')
plt.xlabel('epoch')
plt.ylabel('train_loss')
plt.ylim([0.0, 1.0])
plt.legend(loc='lower right')
plt.show()
'''
if __name__ == '__main__':
    a1 = loadmat("./dataset/Indian_pines_gt.mat")['indian_pines_gt']
    '''
   result_C = loadmat(
      "./Results/PU_Reg2DGabor_NoAug_SaveBest_lr_auto_ks_5_Patch_15_t50_v50_perC_bs50_epo150_1_mywrite_cnn.mat")
   train_loss_C = result_C['train_loss'][-1]
   print(train_loss_C.shape)
   plt.plot(train_loss_C, label='CNN')
   plt.xlabel('epoch')
   plt.ylabel('train_loss')
   plt.ylim([0.0, 1.0])
   plt.legend(loc='lower right')
   plt.show()
   '''
    '''
   a2 = loadmat("./Results/pu_2D_pytorch_test_img.mat")
   print(a2.keys())
   a3 = a2['train_loss'][-1]
   print(a3.shape)
   plt.plot(a3, label='CNN')
   plt.xlabel('epoch')
   plt.ylabel('train_loss')
   plt.ylim([0.0, 1.0])
   plt.legend(loc='lower right')
   plt.show()
   '''
    # plt.figure(dpi=500)
    # plt.imshow(a2)
    # predict_image = spectral.imshow(classes=a1.astype(int), figsize=(7, 7))
    # predict_image = spectral.imshow(classes=a2.astype(int), figsize=(7, 7))
    # plt.savefig("dataset/123.tif")
    # plt.show()

    a2 = loadmat("./Results/pu_2D_pytorch_test_img.mat")
    print(a2.keys())
    a3 = a2['train_loss'][-1]
    print(a3.shape)
    plt.plot(a3, label='Gabor')
    plt.xlabel('epoch')
    plt.ylabel('train_loss')
    plt.ylim([0.0, 1.0])
    plt.legend(loc='lower right')
    plt.show()

    a4 = \
        loadmat(
            "./Results/PUD_pytorch_img_plot.mat")[
            'output_final']
    plt.figure(dpi=500, figsize=(7, 7))
    plt.imshow(a4, cmap='gist_ncar')
    plt.colorbar()
    plt.axis('off')
    plt.savefig("dataset/111.tif")

    plt.pause(6000)
