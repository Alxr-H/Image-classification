import torch
import torch.nn.functional as F
from torch import nn

from z_pytorch_ops import get_theta, get_omega, get_sigma, get_phase, get_2DGabor_kernels


class RegGaborConv2d(nn.Module):
    def __init__(self, in_channels, n_theta, n_omega, kernel_size, strides=1, padding=0, even_initial=True,
                 use_bias=True, requires_grad=True):
        """

        @param in_channels: img band number
        @param n_theta: number of theta initializations (int)
        @param n_omega: number of omegas initializations (int)
        @param kernel_size: 5
        @param strides: 1
        @param padding: 2
        @param even_initial: whether adopt the even initialization strategy, false for random initialization
        @param use_bias:
        @param requires_grad:
        """
        super(RegGaborConv2d, self).__init__()
        self.requires_grad = requires_grad
        self.even_initial = even_initial
        self.padding = padding
        self.strides = strides
        self.kernel_size = kernel_size
        self.n_omega = n_omega
        self.n_theta = n_theta
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(n_theta * n_omega))
        else:
            self.bias = None
        n_out = n_theta * n_omega
        self.theta = nn.Parameter(get_theta(in_channels, n_theta, n_omega, even_initial=even_initial,
                                            requires_grad=requires_grad))
        self.omega = nn.Parameter(get_omega(in_channels, n_theta, n_omega, even_initial=even_initial,
                                            requires_grad=requires_grad))
        self.sigma = nn.Parameter(get_sigma(in_channels, n_out, kernel_size=kernel_size, even_initial=even_initial,
                                            requires_grad=requires_grad))
        self.phase = nn.Parameter(get_phase(in_channels, n_out, even_initial=False, requires_grad=requires_grad))

    def forward(self, inputs):
        W = get_2DGabor_kernels(theta=self.theta, omega=self.omega, sigma=self.sigma, phase=self.phase,
                                kernel_size=self.kernel_size, device=inputs.device)
        return F.conv2d(inputs, W, self.bias, stride=self.strides, padding=self.padding)


def regGaborConv2d(x, n_theta, n_omega, kernel_size, strides=1, padding=0, even_initial=True, bias=None,
                   device='cpu', requires_grad=False):
    xsh = list(x.shape)
    n_out = n_theta * n_omega
    theta = get_theta(xsh[1], n_theta, n_omega, even_initial=even_initial, device=device, requires_grad=requires_grad)
    omega = get_omega(xsh[1], n_theta, n_omega, even_initial=even_initial, device=device, requires_grad=requires_grad)
    sigma = get_sigma(xsh[1], n_out, kernel_size=kernel_size, even_initial=even_initial,
                      device=device, requires_grad=requires_grad)
    phase = get_phase(xsh[1], n_out, even_initial=False, device=device, requires_grad=requires_grad)
    W = get_2DGabor_kernels(theta, omega, sigma, phase, kernel_size=kernel_size, device=device)
    return F.conv2d(x, W, bias, stride=strides, padding=padding)
