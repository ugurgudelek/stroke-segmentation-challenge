# -*- encoding: utf-8 -*-
# @File    :   ResUnet.py
# @Time    :   2021/06/05 21:41:27
# @Author  :   Gorkem Can Ates
# @Contact :   g.canates@gmail.com
# @Desc    :   None

from berries.model.base import BaseModel
import torch
import torch.nn as nn


class RecursiveModel(BaseModel):
    def __init__(self, model, K=1, device='cuda'):
        super(RecursiveModel, self).__init__()
        self.model = model
        self.K = K
        self.device = device

    def forward(self, x):
        out = torch.zeros_like(x).to(self.device)
        outputs = []
        for i in range(self.K):
            out = self.model(torch.cat((x, out), dim=1))
            outputs.append(out)
        return outputs
