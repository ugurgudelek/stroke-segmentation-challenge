import torch
from torch import nn


class Precision(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, yhat, y):
        preds = torch.round(yhat).detach()
        TP = torch.sum((preds == 1) * (y == 1))
        FP = torch.sum((preds == 1) * (y == 0))
        return TP / (TP + FP + self.eps)


class Recall(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, yhat, y):
        preds = torch.round(yhat).detach()
        TP = torch.sum((preds == 1) * (y == 1))
        FN = torch.sum((preds == 0) * (y == 1))
        return TP / (TP + FN + self.eps)


class FPR(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, yhat, y):
        preds = torch.round(yhat).detach()
        FP = torch.sum((preds == 1) * (y == 0))
        TN = torch.sum((preds == 0) * (y == 0))
        return FP / (FP + TN + self.eps)


class F1(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, yhat, y):
        preds = torch.round(yhat).detach()
        TP = torch.sum((preds == 1) * (y == 1))
        FN = torch.sum((preds == 0) * (y == 1))
        FP = torch.sum((preds == 1) * (y == 0))
        return TP / (TP + 0.5 * (FP + FN) + self.eps)
