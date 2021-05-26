from berries.model.base import BaseModel
import torch
import torch.nn as nn
from model.MainBlocks import conv_block, DepthWiseConv2D, DSconv_block, x_block, FSM


class XNET(BaseModel):
    def __init__(self, in_channels, out_channels, norm_type='bn', upsample_mode='nearest'):
        nn.Module.__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode=upsample_mode)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.encode1 = x_block(in_features=in_channels, out_features=64, norm_type=norm_type)
        self.encode2 = nn.Sequential(
            self.maxpool,
            x_block(in_features=64, out_features=128, norm_type=norm_type))
        self.encode3 = nn.Sequential(
            self.maxpool,
            x_block(in_features=128, out_features=256, norm_type=norm_type))
        self.encode4 = nn.Sequential(
            self.maxpool,
            x_block(in_features=256, out_features=512, norm_type=norm_type))
        self.encode5 = nn.Sequential(
            self.maxpool,
            x_block(in_features=512, out_features=1024, norm_type=norm_type))

        self.fsm = FSM(norm_type=norm_type)

        self.decode1 = nn.Sequential(
            self.upsample, conv_block(in_features=1024, out_features=512, norm_type=norm_type))
        self.decode2 = nn.Sequential(
            x_block(in_features=512, out_features=512, norm_type=norm_type),
            self.upsample,
            conv_block(in_features=512, out_features=256, norm_type=norm_type))
        self.decode3 = nn.Sequential(
            x_block(in_features=256, out_features=256, norm_type=norm_type),
            self.upsample,
            conv_block(in_features=256, out_features=128, norm_type=norm_type))
        self.decode4 = nn.Sequential(
            x_block(in_features=128, out_features=128, norm_type=norm_type),
            self.upsample,
            conv_block(in_features=128, out_features=64, norm_type=norm_type))
        self.decode5 = nn.Sequential(
            x_block(in_features=64, out_features=64, norm_type=norm_type),
            nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1, padding=0, stride=1))
        self.initialize_weights()

    def forward(self, x):
        x1 = self.encode1(x)
        x2 = self.encode2(x1)
        x3 = self.encode3(x2)
        x4 = self.encode4(x3)
        x = self.encode5(x4)
        x = self.fsm(x)
        x = self.decode1(torch.cat((x, x4), dim=1))
        x = self.decode2(torch.cat((x, x3), dim=1))
        x = self.decode3(torch.cat((x, x2), dim=1))
        x = self.decode4(torch.cat((x, x1), dim=1))
        x = self.decode5(x)

        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.kaiming_normal_(m.bias)
