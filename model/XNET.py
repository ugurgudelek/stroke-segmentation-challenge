# -*- encoding: utf-8 -*-
# @File    :   XNET.py
# @Time    :   2021/05/26 23:12:27
# @Author  :   Gorkem Can Ates
# @Contact :   g.canates@gmail.com
# @Desc    :   None

from berries.model.base import BaseModel
import torch
import torch.nn as nn
from model.MainBlocks import conv_block, DepthWiseConv2D, DSconv_block, x_block, FSM


class XNET(BaseModel):
    def __init__(self, in_channels, out_channels, device, k=1, norm_type='bn', upsample_mode='nearest'):
        nn.Module.__init__(self)
        self.upsample = nn.Upsample(scale_factor=2, mode=upsample_mode)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.encode1 = x_block(in_features=in_channels, out_features=int(64 * k), norm_type=norm_type)
        self.encode2 = nn.Sequential(
            self.maxpool,
            x_block(in_features=int(64 * k), out_features=int(128 * k), norm_type=norm_type))
        self.encode3 = nn.Sequential(
            self.maxpool,
            x_block(in_features=int(128 * k), out_features=int(256 * k), norm_type=norm_type))
        self.encode4 = nn.Sequential(
            self.maxpool,
            x_block(in_features=int(256 * k), out_features=int(512 * k), norm_type=norm_type))
        self.encode5 = nn.Sequential(
            self.maxpool,
            x_block(in_features=int(512 * k), out_features=int(1024 * k), norm_type=norm_type))

        self.fsm = FSM(norm_type=norm_type, device=device)

        self.decode1 = nn.Sequential(
            self.upsample, conv_block(in_features=int(1024 * k), out_features=int(512 * k), norm_type=norm_type))
        self.decode2 = nn.Sequential(
            x_block(in_features=int(1024 * k), out_features=int(512 * k), norm_type=norm_type),
            self.upsample,
            conv_block(in_features=int(512 * k), out_features=int(256 * k), norm_type=norm_type))
        self.decode3 = nn.Sequential(
            x_block(in_features=int(512 * k), out_features=int(256 * k), norm_type=norm_type),
            self.upsample,
            conv_block(in_features=int(256 * k), out_features=int(128 * k), norm_type=norm_type))
        self.decode4 = nn.Sequential(
            x_block(in_features=int(256 * k), out_features=int(128 * k), norm_type=norm_type),
            self.upsample,
            conv_block(in_features=int(128 * k), out_features=int(64 * k), norm_type=norm_type))
        self.decode5 = nn.Sequential(
            x_block(in_features=int(128 * k), out_features=int(64 * k), norm_type=norm_type),
            nn.Conv2d(in_channels=int(64 * k), out_channels=out_channels, kernel_size=1, padding=0, stride=1))
        self.initialize_weights()

    def forward(self, x):
        x1 = self.encode1(x)
        x2 = self.encode2(x1)
        x3 = self.encode3(x2)
        x4 = self.encode4(x3)
        x = self.encode5(x4)
        x = self.fsm(x)
        x = self.decode1(x)
        x = self.decode2(torch.cat((x, x4), dim=1))
        x = self.decode3(torch.cat((x, x3), dim=1))
        x = self.decode4(torch.cat((x, x2), dim=1))
        x = self.decode5(torch.cat((x, x1), dim=1))

        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def test(batchsize):
    in_channels = 3
    in1 = torch.rand(batchsize, in_channels, 512, 512).to('cuda')
    model = XNET(in_channels=in_channels, device='cuda', out_channels=3, k=0.25, norm_type='gn').to('cuda')

    out1 = model(in1)
    return out1.shape


test(batchsize=4)
# pytorch_total_params = sum(p.numel() for p in model.parameters())