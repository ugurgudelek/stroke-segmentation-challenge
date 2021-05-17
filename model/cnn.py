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
            nn.Sigmoid()
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
        if bn:
            self.features = models.vgg16_bn(pretrained=pre_trained).features

        else:
            self.features = models.vgg16(pretrained=pre_trained).features

        for param in self.features.parameters():
            param.requires_grad = req_grad

        num_features_before_fcnn = functools.reduce(
            operator.mul,
            list(self.features(torch.rand(1, *input_dim)).shape))

        self.classifier = nn.Sequential(
            nn.Linear(in_features=num_features_before_fcnn, out_features=1024),
            nn.Linear(in_features=1024, out_features=512),
            nn.Linear(in_features=512, out_features=out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)
        out = self.features(x)
        out = out.view(batch_size, -1)  # flatten the vector
        out = self.classifier(out)
        return out


class VGG19(BaseModel):
    def __init__(self, pre_trained, req_grad, bn, out_channels=1, input_dim=(3, 512, 512)):
        nn.Module.__init__(self)
        if bn:
            self.features = models.vgg19_bn(pretrained=pre_trained).features

        else:
            self.features = models.vgg19(pretrained=pre_trained).features

        for param in self.features.parameters():
            param.requires_grad = req_grad

        num_features_before_fcnn = functools.reduce(
            operator.mul,
            list(self.features(torch.rand(1, *input_dim)).shape))

        self.classifier = nn.Sequential(
            nn.Linear(in_features=num_features_before_fcnn, out_features=1024),
            nn.Linear(in_features=1024, out_features=512),
            nn.Linear(in_features=512, out_features=out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)
        out = self.features(x)
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
            nn.Sigmoid()

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
            nn.Sigmoid()

        )

    def forward(self, x):
        batch_size = x.size(0)
        out = self.features(x)
        out = out.view(batch_size, -1)
        out = self.classifier(out)
        return out


