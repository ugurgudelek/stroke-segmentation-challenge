import torch
from torch import nn


class Accuracy(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()

    def forward(self, yhat, y):
        preds = (torch.argmax(yhat, dim=1)).detach()
        acc = (torch.sum(preds == y)) / yhat.size(0)
        return acc.cpu().numpy().item()


class Precision(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, yhat, y):
        preds = (torch.argmax(yhat, dim=1)).detach()
        TP = torch.sum((preds == 1) * (y == 1))
        FP = torch.sum((preds == 1) * (y == 0))
        return (TP / (TP + FP + self.eps)).cpu().numpy().item()


class Recall(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, yhat, y):
        preds = (torch.argmax(yhat, dim=1)).detach()
        TP = torch.sum((preds == 1) * (y == 1))
        FN = torch.sum((preds == 0) * (y == 1))
        return (TP / (TP + FN + self.eps)).cpu().numpy().item()


class Specificity(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, yhat, y):
        preds = (torch.argmax(yhat, dim=1)).detach()
        FP = torch.sum((preds == 1) * (y == 0))
        TN = torch.sum((preds == 0) * (y == 0))
        return 1 - (FP / (FP + TN + self.eps)).cpu().numpy().item()

# torch.round
class MeanMetric(nn.Module):

    def __init__(self):
        super().__init__()
        self.recall = Recall()
        self.specificity = Specificity()

    def forward(self, yhat, y):

        recall = self.recall(yhat, y)
        specificity = self.specificity(yhat, y)
        return (recall + specificity) / 2


class F1(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, yhat, y):
        preds = (torch.argmax(yhat, dim=1)).detach()
        TP = torch.sum((preds == 1) * (y == 1))
        FN = torch.sum((preds == 0) * (y == 1))
        FP = torch.sum((preds == 1) * (y == 0))
        return (TP / (TP + 0.5 * (FP + FN) + self.eps)).cpu().numpy().item()
