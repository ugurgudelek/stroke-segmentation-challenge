# -*- encoding: utf-8 -*-
# @File    :   ResUnet.py
# @Time    :   2021/05/29 19:46:27
# @Author  :   Gorkem Can Ates
# @Contact :   g.canates@gmail.com
# @Desc    :   None

from berries.model.base import BaseModel
import torch
import torch.nn as nn
import torchvision
from torchvision.models import vgg19
from model.MainBlocks import conv_block, DoubleASPP, SqueezeExciteBlock


class DoubleUnet(BaseModel):
    def __init__(self, out_features=3, k=1, norm_type='bn', upsample_mode='bilinear'):
        nn.Module.__init__(self)

        self.mu = torch.tensor([0.485, 0.456, 0.406], requires_grad=False).view((1, 3, 1, 1))
        self.sigma = torch.tensor([0.229, 0.224, 0.225], requires_grad=False).view((1, 3, 1, 1))

        for params in vgg19(pretrained=True).features.parameters():
            params.requires_grad = True

        self.vgg1 = vgg19(pretrained=True).features[:4]
        self.vgg2 = vgg19(pretrained=True).features[4:9]
        self.vgg3 = vgg19(pretrained=True).features[9:18]
        self.vgg4 = vgg19(pretrained=True).features[18:27]
        self.vgg5 = vgg19(pretrained=True).features[27:]
        self.ASPP = DoubleASPP(in_features=512, out_features=int(64 * k), norm_type=norm_type)
        self.upsample = nn.Upsample(scale_factor=2, mode=upsample_mode)
        self.decode1_1 = nn.Sequential(
            convblock(in_features=int(64 * k) + 512, out_features=int(256 * k), norm_type=norm_type),
            conv_block(in_features=int(256 * k), out_features=int(256 * k), norm_type=norm_type),
            SqueezeExciteBlock(in_features=int(128 * k), reduction=8),
            nn.Upsample(scale_factor=2, mode=upsample_mode))
        self.decode1_2 = nn.Sequential(
            convblock(in_features=int(256 * k) + 256, out_features=int(128 * k), norm_type=norm_type),
            conv_block(in_features=int(128 * k), out_features=int(128 * k), norm_type=norm_type),
            SqueezeExciteBlock(in_features=int(128 * k), reduction=8),
            nn.Upsample(scale_factor=2, mode=upsample_mode))
        self.decode1_3 = nn.Sequential(
            convblock(in_features=int(128 * k) + 128, out_features=int(64 * k), norm_type=norm_type),
            conv_block(in_features=int(64 * k), out_features=int(64 * k), norm_type=norm_type),
            SqueezeExciteBlock(in_features=int(128 * k), reduction=8),
            nn.Upsample(scale_factor=2, mode=upsample_mode))
        self.decode1_4 = nn.Sequential(
            convblock(in_features=int(64 * k) + 64, out_features=int(32 * k), norm_type=norm_type),
            conv_block(in_features=int(32 * k), out_features=int(32 * k), norm_type=norm_type),
            SqueezeExciteBlock(in_features=int(128 * k), reduction=8),
            nn.Upsample(scale_factor=2, mode=upsample_mode))
        self.output1 = nn.Conv2d(int(32 * k), out_features, kernel_size=(1, 1))







        self.output2 = nn.Conv2d(int(32 * k), out_features, kernel_size=(1, 1))

        self.initialize_weights()

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()


def test(batchsize):
    in_channels = 3
    in1 = torch.rand(batchsize, in_channels, 512, 512).to('cuda')
    model = DoubleUnet(out_features=3, k=0.25, norm_type='bn').to('cuda')

    out1 = model(in1)
    total_params = sum(p.numel() for p in model.parameters())

    return out1.shape, total_params


if __name__ == '__main__':
    test(batchsize=8)