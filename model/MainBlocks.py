# -*- encoding: utf-8 -*-
# @File    :   MainBlocks.py
# @Time    :   2021/05/26 23:12:27
# @Author  :   Gorkem Can Ates
# @Contact :   g.canates@gmail.com
# @Desc    :   None

from berries.model.base import BaseModel
import torch
import torch.nn as nn


class conv_block(BaseModel):
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, norm_type=None):
        nn.Module.__init__(self)
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=kernel_size, stride=1, padding=padding)
        self.norm_type = norm_type
        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.norm_type is not None:
            x = self.norm(x)
        x = self.relu(x)
        return x


class DepthWiseConv2D(BaseModel):
    def __init__(self, in_features, kernels_per_layer=1):
        nn.Module.__init__(self)
        self.depthwise = nn.Conv2d(
            in_features, in_features * kernels_per_layer, kernel_size=3, padding=1, groups=in_features, dilation=1)
        self.pointwise = nn.Conv2d(in_features * kernels_per_layer, in_features, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DSconv_block(BaseModel):
    def __init__(self, in_features, out_features, norm_type):
        nn.Module.__init__(self)
        self.norm_type = norm_type
        self.DSconv = DepthWiseConv2D(in_features)
        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if in_features >= 32 else in_features, in_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(in_features)
        self.relu = nn.ReLU()
        self.conv2d = conv_block(in_features, out_features, kernel_size=1, padding=0, norm_type=norm_type)

    def forward(self, x):
        x = self.DSconv(x)
        if self.norm_type is not None:
            x = self.norm(x)
        x = self.relu(x)
        x = self.conv2d(x)
        return x


class x_block(BaseModel):
    def __init__(self, in_features, out_features, norm_type):
        nn.Module.__init__(self)
        self.res = conv_block(in_features, out_features, kernel_size=1, padding=0, norm_type=norm_type)
        self.dc1 = DSconv_block(in_features, out_features, norm_type=norm_type)
        self.dc2 = DSconv_block(out_features, out_features, norm_type=norm_type)

    def forward(self, x):
        res = self.res(x)
        x = self.dc1(x)
        x = self.dc2(x)
        x = self.dc2(x)
        x += res
        return x


class FSM(BaseModel):
    def __init__(self, norm_type, device):
        nn.Module.__init__(self)
        self.norm_type = norm_type
        self.device = device

    def forward(self, x):
        channel_num = x.shape[1]
        res = x
        x = conv_block(
            in_features=int(channel_num), out_features=int(channel_num // 8), norm_type=self.norm_type).to(self.device)(
            x)

        ip = x
        batchsize, channels, dim1, dim2 = ip.shape
        intermediate_dim = int(channels // 2)

        theta = nn.Conv2d(int(channel_num // 8), intermediate_dim, kernel_size=1, padding=0, bias=False).to(
            self.device)(ip)
        theta = torch.reshape(theta, (batchsize, -1, intermediate_dim))

        phi = nn.Conv2d(int(channel_num // 8), intermediate_dim, kernel_size=1, padding=0, bias=False).to(self.device)(
            ip)
        phi = torch.reshape(phi, (batchsize, -1, intermediate_dim))

        f = torch.bmm(theta, phi.view(batchsize, intermediate_dim, phi.shape[1]))
        f = f / (float(f.shape[1]))

        g = nn.Conv2d(int(channel_num // 8), intermediate_dim, kernel_size=1, padding=0, bias=False).to(self.device)(ip)
        g = torch.reshape(g, (batchsize, -1, intermediate_dim))

        y = torch.bmm(f, g)
        y = torch.reshape(y, (batchsize, intermediate_dim, dim1, dim2))
        y = nn.Conv2d(intermediate_dim, channels, kernel_size=1, padding=0, bias=False).to(self.device)(y)
        y += ip
        x = y
        x = conv_block(in_features=channels, out_features=int(channel_num), norm_type=self.norm_type).to(self.device)(x)
        x += res

        return x


class ResConv(BaseModel):
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, stride=1, norm_type=None):
        nn.Module.__init__(self)
        self.norm_type = norm_type
        if self.norm_type == 'gn':
            self.norm1 = nn.GroupNorm(32 if (in_features >= 32 and in_features%32==0) else in_features, in_features)
            self.norm2 = nn.GroupNorm(32 if (out_features >= 32 and out_features%32==0) else out_features, out_features)

        if self.norm_type == 'bn':
            self.norm1 = nn.BatchNorm2d(in_features)
            self.norm2 = nn.BatchNorm2d(out_features)

        self.pack = nn.Sequential(self.norm1,
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=in_features, out_channels=out_features,
                                            kernel_size=kernel_size, stride=stride, padding=padding),
                                  self.norm2,
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=out_features, out_channels=out_features,
                                            kernel_size=kernel_size, stride=1, padding=1))
        self.skip = nn.Sequential(nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                                            stride=stride, padding=1),
                                  self.norm2)

    def forward(self, x):
        res = self.skip(x)
        x = self.pack(x)
        x += res
        return x
