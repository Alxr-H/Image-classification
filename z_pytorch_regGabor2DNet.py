"""
@Time    : 2022/10/26 22:15
@Author  : Lin Luo
@FileName: regGabor2DNet.py
@describe TODO
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torchsummary import summary

from z_pytorch_layer import RegGaborConv2d


class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias: bool = True):
        super(MyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.normal_(self.weight, std=0.2)
        if self.bias is not None:
            init.constant_(self.bias, val=1e-2)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)


class RegGabor2DNet(nn.Module):
    def __init__(self, in_channels=5, num_classes=16, even_initial=True):
        """
        相位诱导的 Gabor 卷积核 网络，2块
        @param in_channels:
        @param num_classes:
        @param even_initial: whether adopt the even initialization strategy, false for random initialization
        """
        super(RegGabor2DNet, self).__init__()
        self.gfc = nn.Sequential(
            RegGaborConv2d(in_channels=in_channels, n_theta=4, n_omega=4, kernel_size=5, padding=2,
                           even_initial=even_initial),
            nn.LeakyReLU(0.2),
            RegGaborConv2d(in_channels=16, n_theta=4, n_omega=4, kernel_size=5, padding=2, even_initial=even_initial),
            nn.BatchNorm2d(num_features=16),
            RegGaborConv2d(in_channels=16, n_theta=8, n_omega=4, kernel_size=5, padding=2, even_initial=even_initial),
            nn.LeakyReLU(0.2),
            RegGaborConv2d(in_channels=32, n_theta=8, n_omega=4, kernel_size=5, padding=2, even_initial=even_initial),
            nn.BatchNorm2d(num_features=32)
        )
        self.cls = nn.Sequential(
            MyLinear(32, 64),
            nn.LeakyReLU(0.2),
            MyLinear(64, num_classes)
        )

    def forward(self, inputs):
        out = self.gfc(inputs)
        # print(out.shape)
        # torch.Size([2, 32, 32, 32])  # torch.Size([64, 32, 15, 15]) # [batch,out, patchsize, patchsize]
        out = torch.mean(out, dim=(2, 3))
        # print(out.shape)    # torch.Size([2, 32])   # torch.Size([64, 32])
        return self.cls(out)


class RegGabor2DNet1(nn.Module):
    def __init__(self, in_channels=5, num_classes=16, even_initial=True):
        """
        相位诱导的 Gabor 卷积核 网络，2块
        @param in_channels:
        @param num_classes:
        @param even_initial: whether adopt the even initialization strategy, false for random initialization
        """
        super(RegGabor2DNet1, self).__init__()
        self.conv1 = RegGaborConv2d(in_channels=in_channels, n_theta=4, n_omega=4, kernel_size=5, padding=2,
                                    even_initial=even_initial)
        self.relu1 = nn.LeakyReLU(0.2)
        self.conv2 = RegGaborConv2d(in_channels=16, n_theta=4, n_omega=4, kernel_size=5, padding=2, even_initial=even_initial)
        self.gfc = nn.Sequential(
            nn.BatchNorm2d(num_features=16),
            RegGaborConv2d(in_channels=16, n_theta=8, n_omega=4, kernel_size=5, padding=2, even_initial=even_initial),
            nn.LeakyReLU(0.2),
            RegGaborConv2d(in_channels=32, n_theta=8, n_omega=4, kernel_size=5, padding=2, even_initial=even_initial),
            nn.BatchNorm2d(num_features=32)
        )
        self.cls = nn.Sequential(
            MyLinear(32, 64),
            nn.LeakyReLU(0.2),
            MyLinear(64, num_classes)
        )

    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.gfc(out)
        # print(out.shape)
        # torch.Size([2, 32, 32, 32])  # torch.Size([64, 32, 15, 15]) # [batch,out, patchsize, patchsize]
        out = torch.mean(out, dim=(2, 3))
        # print(out.shape)    # torch.Size([2, 32])   # torch.Size([64, 32])
        return self.cls(out)


class CNNNet(nn.Module):
    def __init__(self, in_channels=5, num_classes=16, even_initial=True):
        """
        相位诱导的 Gabor 卷积核 网络，2块
        @param in_channels:
        @param num_classes:
        @param even_initial: whether adopt the even initialization strategy, false for random initialization
        """
        super(CNNNet, self).__init__()
        self.gfc = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=16),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=32)
        )
        self.cls = nn.Sequential(
            MyLinear(32, 64),
            nn.LeakyReLU(0.2),
            MyLinear(64, num_classes)
        )

    def forward(self, inputs):
        out = self.gfc(inputs)
        # print(out.shape)
        # torch.Size([2, 32, 32, 32])  # torch.Size([64, 32, 15, 15]) # [batch,out, patchsize, patchsize]
        out = torch.mean(out, dim=(2, 3))
        # print(out.shape)    # torch.Size([2, 32])   # torch.Size([64, 32])
        return self.cls(out)


if __name__ == '__main__':
    for i in range(10000):
        print(i, "Running -----------------------------------------------------------------------")
        in_c = 103
        a = torch.rand(2, in_c, 15, 15) * 2 - 1
        # a.shape=[2,103,32,32]
        nc = 9
        m = RegGabor2DNet(in_channels=in_c, num_classes=nc)
        print(in_c, nc)  # 103 9
        b = m(a)

        # 展现模型结构
        # device = torch.device("cuda")
        # model = RegGabor2DNet(in_channels=in_c, num_classes=nc).to(device)
        # summary(model, (103, 5, 5))

        print(b.shape)  # torch.Size([2, 9]) # 因为有两个batch
        print(torch.softmax(b, dim=1))  # dim=1指代的是列
        # if np.isnan(b.detach().numpy()).any() or np.isnan(b.detach().numpy()).any():
        break
