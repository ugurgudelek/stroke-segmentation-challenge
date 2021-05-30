# -*- encoding: utf-8 -*-
# @File    :   cnn.py
# @Time    :   2021/05/15 06:15:27
# @Author  :   Ugur Gudelek
# @Contact :   ugurgudelek@gmail.com
# @Desc    :   None

from berries.model.base import BaseModel
import torch
from torch import nn
from torchvision import models

import functools
import operator


class CNN(BaseModel):
    """Basic Pytorch CNN implementation"""

    def __init__(self, in_channels, out_channels, input_dim):
        nn.Module.__init__(self)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=20,
                      kernel_size=3,
                      stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=3,
                      stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        num_features_before_fcnn = functools.reduce(
            operator.mul,
            list(self.feature_extractor(torch.rand(1, *input_dim)).shape))

        self.classifier = nn.Sequential(
            nn.Linear(in_features=num_features_before_fcnn, out_features=100),
            nn.Linear(in_features=100, out_features=out_channels),
        )

    def forward(self, x):
        batch_size = x.size(0)

        out = self.feature_extractor(x)
        out = out.view(batch_size, -1)  # flatten the vector
        out = self.classifier(out)
        return out


class VGG16(BaseModel):
    def __init__(self, pre_trained, req_grad, bn, out_channels=1, input_dim=(3, 512, 512)):
        nn.Module.__init__(self)
        self.adaptavgpool = nn.AdaptiveAvgPool2d(2)
        if bn:
            self.features = models.vgg16_bn(pretrained=pre_trained).features[:-1]

        else:
            self.features = models.vgg16(pretrained=pre_trained).features[:-1]

        for param in self.features.parameters():
            param.requires_grad = req_grad

        num_features_before_fcnn = functools.reduce(
            operator.mul,
            list(self.adaptavgpool(self.features(torch.rand(1, *input_dim))).shape))

        self.classifier = nn.Sequential(
            nn.Linear(in_features=num_features_before_fcnn, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=out_channels),
        )

    def forward(self, x):
        batch_size = x.size(0)
        out = self.features(x)
        out = self.adaptavgpool(out)
        out = out.view(batch_size, -1)  # flatten the vector
        out = self.classifier(out)
        return out


class VGG19(BaseModel):
    def __init__(self, pre_trained, req_grad, bn, out_channels=2, input_dim=(3, 512, 512)):
        nn.Module.__init__(self)
        self.adaptavgpool = nn.AdaptiveAvgPool2d(2)
        if bn:
            self.features = models.vgg19_bn(pretrained=pre_trained).features[:-1]

        else:
            self.features = models.vgg19(pretrained=pre_trained).features[:-1]

        for param in self.features.parameters():
            param.requires_grad = req_grad

        num_features_before_fcnn = functools.reduce(
            operator.mul,
            list(self.adaptavgpool(self.features(torch.rand(1, *input_dim))).shape))

        self.classifier = nn.Sequential(
            nn.Linear(in_features=num_features_before_fcnn, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=out_channels),
        )

    def forward(self, x):
        batch_size = x.size(0)
        out = self.features(x)
        out = self.adaptavgpool(out)
        out = out.view(batch_size, -1)  # flatten the vector
        out = self.classifier(out)
        return out


class DenseNet(BaseModel):
    def __init__(self, net_type, pre_trained, req_grad, out_channels=1, input_dim=(3, 512, 512)):
        nn.Module.__init__(self)
        if net_type == 'densenet-121':
            self.features = models.densenet121(pretrained=pre_trained).features

        if net_type == 'densenet-161':
            self.features = models.densenet161(pretrained=pre_trained).features

        if net_type == 'densenet-169':
            self.features = models.densenet169(pretrained=pre_trained).features

        if net_type == 'densenet-201':
            self.features = models.densenet201(pretrained=pre_trained).features

        for param in self.features.parameters():
            param.requires_grad = req_grad

        num_features_before_fcnn = functools.reduce(
            operator.mul,
            list(self.features(torch.rand(1, *input_dim)).shape))

        self.classifier = nn.Sequential(
            nn.Linear(in_features=num_features_before_fcnn, out_features=out_channels),

        )

    def forward(self, x):
        batch_size = x.size(0)
        out = self.features(x)
        out = out.view(batch_size, -1)  # flatten the vector
        out = self.classifier(out)
        return out


class ResNet(BaseModel):
    def __init__(self, net_type, pre_trained, req_grad, out_channels=1, input_dim=(3, 512, 512)):
        nn.Module.__init__(self)
        if net_type == 'ResNet-18':
            self.features = nn.Sequential(*(list(models.resnet18(pretrained=pre_trained).children())[:-1]))

        if net_type == 'ResNet-34':
            self.features = nn.Sequential(*(list(models.resnet34(pretrained=pre_trained).children())[:-1]))

        if net_type == 'ResNet-50':
            self.features = nn.Sequential(*(list(models.resnet50(pretrained=pre_trained).children())[:-1]))

        if net_type == 'ResNet-101':
            self.features = nn.Sequential(*(list(models.resnet101(pretrained=pre_trained).children())[:-1]))

        if net_type == 'ResNet-152':
            self.features = nn.Sequential(*(list(models.resnet152(pretrained=pre_trained).children())[:-1]))

        for param in self.features.parameters():
            param.requires_grad = req_grad

        num_features_before_fcnn = functools.reduce(
            operator.mul,
            list(self.features(torch.rand(1, *input_dim)).shape))

        self.classifier = nn.Sequential(
            nn.Linear(in_features=num_features_before_fcnn, out_features=out_channels),

        )

    def forward(self, x):
        batch_size = x.size(0)
        out = self.features(x)
        out = out.view(batch_size, -1)
        out = self.classifier(out)
        return out


class CustomCNN(BaseModel):
    def __init__(self, in_channels, out_channels, input_dim):
        nn.Module.__init__(self)
        self.feature_extractor = nn.Sequential(
            conv_block(in_features=in_channels, out_features=32, padding=1, gn=True),
            conv_block(in_features=32, out_features=32, padding=1, gn=True),
            nn.MaxPool2d(kernel_size=2),
            conv_block(in_features=32, out_features=64, padding=1, gn=True),
            conv_block(in_features=64, out_features=64, padding=1, gn=True),
            nn.MaxPool2d(kernel_size=2),
            conv_block(in_features=64, out_features=128, padding=1, gn=True),
            conv_block(in_features=128, out_features=128, padding=1, gn=True),
            nn.MaxPool2d(kernel_size=2),
            conv_block(in_features=128, out_features=256, padding=1, gn=True),
            conv_block(in_features=256, out_features=256, padding=1, gn=True),
            nn.MaxPool2d(kernel_size=2),

        )

        num_features_before_fcnn = functools.reduce(
            operator.mul,
            list(self.feature_extractor(torch.rand(1, *input_dim)).shape))

        self.classifier = nn.Sequential(
            nn.Linear(in_features=num_features_before_fcnn, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=out_channels),

        )

    def forward(self, x):
        batch_size = x.size(0)

        out = self.feature_extractor(x)
        out = out.view(batch_size, -1)  # flatten the vector
        out = self.classifier(out)
        return out


class conv_block(BaseModel):
    def __init__(self, in_features, out_features, padding, kernel_size=3, gn=None):
        nn.Module.__init__(self)
        self.gn = gn
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=kernel_size, stride=1, padding=padding)
        self.norm = nn.GroupNorm(32, out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.gn is not None:
            x = self.norm(x)
        x = self.relu(x)
        return x


# model = torchvision.models.vgg19_bn(pretrained=True)
# n_feats = 1 #   whatever is your number of output features
# last_item_index = len(model.classifier)-1
# old_fc = model.classifier.__getitem__(last_item_index )
# new_fc = nn.Linear(in_features=old_fc.in_features, out_features= n_feats, bias=True)
# model.classifier.__setitem__(last_item_index , new_fc)