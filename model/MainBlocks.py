from berries.model.base import BaseModel
import torch
import torch.nn as nn


class conv_block(BaseModel):
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, norm_type=None):
        nn.Module.__init__(self)
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=kernel_size, stride=1, padding=padding)
        if norm_type == 'gn':
            self.norm = nn.GroupNorm(32, out_features)
        if norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.relu(x)
        return x


class DepthWiseConv2D(BaseModel):
    def __init__(self, in_features, out_features, kernels_per_layer=1):
        nn.Module.__init__(self)
        self.depthwise = nn.Conv2d(
            in_features, out_features * kernels_per_layer, kernel_size=3, padding=1, groups=in_features, dilation=1)
        self.pointwise = nn.Conv2d(out_features * kernels_per_layer, out_features, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DSconv_block(BaseModel):
    def __init__(self, in_features, out_features, norm_type):
        nn.Module.__init__(self)
        self.DSconv = DepthWiseConv2D(in_features, in_features)
        if norm_type == 'gn':
            self.norm = nn.GroupNorm(32, in_features)
        if norm_type == 'bn':
            self.norm = nn.BatchNorm2d(in_features)
        self.relu = nn.ReLU()
        self.conv2d = conv_block(in_features, out_features, kernel_size=1, padding=0, norm=norm_type)

    def forward(self, x):
        x = self.DSconv(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.relu(x)
        x = self.conv2d(x)
        return x


class x_block(BaseModel):
    def __init__(self, in_features, out_features, norm_type):
        nn.Module.__init__()
        self.res = conv_block(in_features, out_features, kernel_size=1, padding=0, norm_type=norm_type)
        self.dc = DSconv_block(out_features, out_features, norm_type=norm_type)

    def forward(self, x):
        res = self.res(x)
        for i in range(3):
            x = self.dc(x)
        x += res
        return x


class FSM(BaseModel):
    def __init__(self, norm_type):
        nn.Module.__init__()
        self.norm_type = norm_type

    def forward(self, x):
        channel_num = x.shape[-1]
        res = x
        x = conv_block(channel_num, int(channel_num // 8), self.norm_type)(x)
        ip = x
        batchsize, channels, dim1, dim2 = ip.shape
        intermediate_dim = channels // 2

        theta = nn.Conv2d(int(channel_num // 8), intermediate_dim, kernel_size=1, padding=0, bias=False)(ip)
        theta = torch.reshape(theta, (batchsize, -1, intermediate_dim))

        phi = nn.Conv2d(int(channel_num // 8), intermediate_dim, kernel_size=1, padding=0, bias=False)(ip)
        phi = torch.reshape(phi, (batchsize, -1, intermediate_dim))

        f = torch.bmm(theta, phi.view(batchsize, intermediate_dim, phi.shape[1]))
        f = f / (float(f.shape[-1]))

        g = nn.Conv2d(int(channel_num // 8), intermediate_dim, kernel_size=1, padding=0, bias=False)(ip)
        g = torch.reshape(g, (batchsize, -1, intermediate_dim))

        y = torch.bmm(f, g)
        y = torch.reshape(y, (batchsize, intermediate_dim, dim1, dim2))
        y = nn.Conv2d(intermediate_dim, channels, kernel_size=1, padding=0, bias=False)(y)
        y += ip
        x = y
        x = conv_block(in_features=channels, out_features=int(channel_num), norm_type=self.norm_type)(x)
        x += res

        return x
