import argparse
from zdataset import *
from z_pytorch_model import *
from z_pytorch_regGabor2DNet import *

if __name__ == '__main__':
    a2 = loadmat("./Results/salinas2D_pytorch_img.mat")[
        'output']
    plt.figure(dpi=500, figsize=(7, 7))
    plt.imshow(a2, cmap='jet')
    plt.colorbar()
    plt.axis('off')
    plt.savefig("dataset/111.tif")
    plt.show()
