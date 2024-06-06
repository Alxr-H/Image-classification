"""
Detailed Gabor-Nets Implementation
"""

import numpy as np
import torch
from math import pi

# 获得二维卷积核w
def get_2DGabor_kernels(theta: torch.Tensor, omega: torch.Tensor, sigma: torch.Tensor, phase: torch.Tensor,
                        kernel_size: int, g_type='Re', device='cpu'):
    """
    Create 2D Gabor kernels

    :param theta:
    :param omega:
    :param sigma:
    :param phase:
    :param kernel_size: 卷积核大小
    :param g_type:
    :param device
    :param requires_grad: whether to use gradients (boolean)

    :return:
    """
    sh = list(theta.shape)              # [out,in]
    # 在索引为-1上增加一个维度
    sigma = torch.unsqueeze(sigma, -1)  # [out,in, 1]
    phase = torch.unsqueeze(phase, -1)  # [out,in, 1]

    # omega,theta[out,in] # 按元素相乘
    Freq_X = omega * torch.cos(theta)  # [out, in]
    Freq_Y = omega * torch.sin(theta)  # [out, in]
    Freq_X = torch.unsqueeze(Freq_X, -1)  # [out, in, 1]
    Freq_Y = torch.unsqueeze(Freq_Y, -1)  # [out, in, 1]

    # Create the Euclidean coordinates of the kernel
    # the first row is y-axis, the second row is x-axis
    coords = L2_grid(kernel_size)
    coords_X = coords[1, :][np.newaxis, np.newaxis, :]  # (1,1, 25)
    coords_Y = coords[0, :][np.newaxis, np.newaxis, :]  # (1,1, 25)

    # envelop construction k
    coords_X = torch.tensor(coords_X, dtype=torch.float32, device=device)
    coords_Y = torch.tensor(coords_Y, dtype=torch.float32, device=device)

    # coords_X=[1,1,25] , sigma=[16,3,1] # 广播机制
    Envelop = (coords_X ** 2 + coords_Y ** 2) / (sigma ** 2)
    Envelop = (torch.exp(-0.5 * Envelop)) / (2 * np.pi * (sigma ** 2))  # (out, in, 25)
    # #print(Envelop.shape)  # torch.Size([16, 3, 25])=(out, in, 25)

    # Freq_X=[out, in, 1] coords_X=[1,1,25]  # result=[16, 3, 25]=(out, in, 25)
    Freq = Freq_X * coords_X + Freq_Y * coords_Y  # (out, in, 25)
    # construct corresponding Gabor kernels
    if g_type == 'Re':
        kernels = Envelop * torch.cos(Freq + phase)
        # print(kernels.shape)  # [16, 3, 25]=(out, in, 25)
    elif g_type == 'Im':
        kernels = Envelop * torch.sin(Freq + phase)
    else:
        print('Error Gabor filter type')
        exit(0)
    return torch.reshape(kernels, sh + [kernel_size, kernel_size])
    # return=[n_out,n_in,kernel_size,kernel_size]


def L2_grid(kernel_size):
    """
    get the Euclidean coordinates of the kernel 核的欧氏坐标

    :param kernel_size: size of the target kernel

    :return: coordinates of the target kernel
    """
    # Get neighbourhoods
    center = kernel_size // 2
    lin = np.arange(kernel_size)
    # numpy.meshgrid()函数可以让我们快速生成坐标矩阵X,Y  #J,I是坐标矩阵  #输入的x,y,就是网格点的横纵坐标列向量（非矩阵）
    J, I = np.meshgrid(lin, lin)  # J-X I-Y
    J -= center  # X
    I -= center  # Y
    # np.reshape(I, -1)展开成一行 # np.vstack()垂直堆叠
    return np.vstack((np.reshape(I, -1), np.reshape(J, -1)))

# 参数的初始化
# 参数的形状都为(n_out, n_in)=(n_theta x n_omega,band)
def get_theta(n_in, n_theta, n_omega, even_initial=False, device='cpu', requires_grad=False):
    """
    Get the list of theta

    :param n_in: number of the channels of inputs (int)
    :param n_theta: number of the initialization values of theta (int)
    :param n_omega: number of the initialization values of theta (int)
    :param even_initial: whether adopt the even initialization strategy (boolean) 是否采用均匀初始化策略(布尔值)
    :param device
    :param requires_grad: whether to use gradients (boolean)
    :return: a list of theta
    """
    if even_initial:
        # np.linspace主要用来创建等差数列 # endpoint：True则包含stop；False则不包含stop # 转化为一列
        init = np.linspace(0, 1, num=n_theta, endpoint=False, dtype=np.float32).reshape((-1, 1)) * np.pi
        # np.tile(a,(2,1))第一个参数为Y轴扩大倍数，第二个为X轴扩大倍数
        init = np.tile(init, (n_omega, 1))
        init = np.tile(init, (1, n_in))
    else:
        n_out = n_theta * n_omega
        init = np.random.rand(n_out, n_in) * (2 * np.pi)

    # theta最后的形状为(n_out, n_in)
    return torch.tensor(init, dtype=torch.float32, device=device, requires_grad=requires_grad)


def get_omega(n_in, n_theta, n_omega, even_initial=False, mean=0, device='cpu', requires_grad=False):
    """
    Get the list of omega

    :param n_in: number of the channels of inputs (int)
    :param n_theta: number of the initialization values of theta (int)
    :param n_omega: number of the initialization values of theta (int)
    :param even_initial: whether adopt the even initialization strategy (boolean)
    :param mean: the mean of normal distribution in the random initialization strategy
    :param name: (default: omega)
    :param device
    :param requires_grad

    :return: a list of omega
    """
    if even_initial:
        # np.logspace() 对数等比数列
        init = np.logspace(0, n_omega - 1, num=n_omega, base=1 / 2) * (np.pi / 2)
        init = init[:, np.newaxis]            # [4,1]
        init = np.tile(init, [1, n_theta])    # [16,1]
        init = np.reshape(init, [1, -1])      # [1,16]
        init = np.tile(init, (n_in, 1))       # [n_in,16]=[n_in,n_out]
        # 最后形状为(n_out, n_in)
        return torch.tensor(init.transpose((1, 0)), dtype=torch.float32, device=device, requires_grad=requires_grad)
    else:
        n_out = n_omega * n_theta
        stddev = np.pi / 8
        # 该函数返回从单独的正态分布中提取的随机数的张量，该正态分布的均值是mean，标准差是std
        return torch.normal(mean=mean, std=stddev, size=[n_out, n_in], dtype=torch.float32, device=device,
                            requires_grad=requires_grad)


def get_sigma(n_in, n_out, kernel_size=5, even_initial=False, mean=0, device='cpu', requires_grad=False):
    """
    Get the list of sigma

    :param n_in: number of the channels of inputs (int)
    :param n_out: number of the channels of outputs (int)
    :param kernel_size: size of kernels
    :param even_initial: whether adopt the even initialization strategy
    :param mean: the mean of normal distribution in the random initialization strategy
    :param device
    :param requires_grad
    :return: a list of sigma
    """
    if even_initial:
        return torch.ones(n_out, n_in, dtype=torch.float32, device=device, requires_grad=requires_grad) * kernel_size / 8
    else:
        stddev = (5 / 4) * (1 / 2)
        return torch.normal(mean=mean, std=stddev, size=[n_out, n_in], dtype=torch.float32, device=device,
                            requires_grad=requires_grad)


def get_phase(n_in, n_out, even_initial=False, device='cpu', requires_grad=False):
    """
    P
    Get the list of phase offsets

    :param n_in: number of the channels of inputs (int)
    :param n_out: number of the channels of outputs (int)
    :param even_initial: whether adopt the even initialization strategy 是否采用均匀初始化策略
    :param name: (default: P)
    :param device
    :param requires_grad
    :return: a list of phase offsets
    """
    if even_initial:
        # initialization corresponding to each output
        return torch.zeros(n_out, n_in, dtype=torch.float32, device=device, requires_grad=requires_grad)
    else:
        # torch.randn标准正态分布
        return torch.randn(n_out, n_in, dtype=torch.float32, device=device, requires_grad=requires_grad) * (2 * np.pi)


if __name__ == '__main__':
    a = get_2DGabor_kernels(
        theta=get_theta(3, 4, 4, device='cuda', even_initial=True),
        omega=get_omega(3, 4, 4, device='cuda', even_initial=True),
        sigma=get_sigma(3, 16, device='cuda', even_initial=True),
        phase=get_phase(3, 16, device='cuda', even_initial=True),
        kernel_size=5,
        device='cuda'
    )
    # print(a)
    print(a.shape, a.requires_grad)
    # torch.Size([16, 3, 5, 5]) False
    # reture=[out,in,kernel_size,kernel_size]
    # requires_grad=False时表示不需要计算梯度
