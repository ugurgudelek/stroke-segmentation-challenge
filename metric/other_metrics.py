import torch
from torch import nn
import torch.nn.functional as F
import torchmetrics
import torchmetrics.functional as MF

class Precision_class1(nn.Module):
    def __init__(self, in_class=1, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.in_class = in_class

    def forward(self, yhat, y):
        yhat = F.softmax(yhat, dim=1)
        preds = (torch.argmax(yhat, dim=1))

        preds[preds != self.in_class] = 0
        TP = torch.sum((preds == self.in_class) * (y == self.in_class))
        FP = torch.sum((preds == self.in_class) * (y == 0))
        return (TP / (TP + FP + self.eps)).numpy().item()


class Precision_class2(nn.Module):
    def __init__(self, in_class=2, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.in_class = in_class

    def forward(self, yhat, y):
        yhat = F.softmax(yhat, dim=1)
        preds = (torch.argmax(yhat, dim=1))

        preds[preds != self.in_class] = 0
        TP = torch.sum((preds == self.in_class) * (y == self.in_class))
        FP = torch.sum((preds == self.in_class) * (y == 0))
        return (TP / (TP + FP + self.eps)).numpy().item()


class Recall_class1(nn.Module):
    def __init__(self, in_class=1, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.in_class = in_class

    def forward(self, yhat, y):
        yhat = F.softmax(yhat, dim=1)
        preds = (torch.argmax(yhat, dim=1))

        preds[preds != self.in_class] = 0
        TP = torch.sum((preds == self.in_class) * (y == self.in_class))
        FN = torch.sum((preds == 0) * (y == self.in_class))
        return (TP / (TP + FN + self.eps)).cpu().numpy().item()


class Recall_class2(nn.Module):
    def __init__(self, in_class=2, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.in_class = in_class

    def forward(self, yhat, y):
        yhat = F.softmax(yhat, dim=1)
        preds = (torch.argmax(yhat, dim=1))

        preds[preds != self.in_class] = 0
        TP = torch.sum((preds == self.in_class) * (y == self.in_class))
        FN = torch.sum((preds == 0) * (y == self.in_class))
        return (TP / (TP + FN + self.eps)).cpu().numpy().item()


class IoU_class1(nn.Module):
    def __init__(self, in_class=1, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.in_class = in_class

    def forward(self, yhat, y):
        preds = (torch.argmax(yhat, dim=1)).detach()
        # y1 = y
        # y1[y1 != self.in_class] = 0
        preds[preds != self.in_class] = 0

        intersection = torch.logical_and(y == self.in_class, preds)
        union = torch.logical_or(y == self.in_class, preds)
        iou = torch.sum(intersection) / (torch.sum(union) + self.eps)

        return iou


class IoU_class2(nn.Module):
    def __init__(self, in_class=2, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.in_class = in_class

    def forward(self, yhat, y):
        preds = (torch.argmax(yhat, dim=1)).detach()
        # y1 = y
        # y1[y1 != self.in_class] = 0
        preds[preds != self.in_class] = 0

        intersection = torch.logical_and(y == self.in_class, preds)
        union = torch.logical_or(y == self.in_class, preds)
        iou = torch.sum(intersection) / (torch.sum(union) + self.eps)

        return iou

# class IoU(nn.Module):
#     def __init__(self, num_classes=3, reduction='elementwise_mean'):
#         super().__init__()
#         self.num_classes = num_classes
#         self.reduction = reduction
#
#     def forward(self, yhat, y):
#         yhat = F.softmax(yhat, dim=1)
#         return MF.iou(yhat, y.type(torch.LongTensor), num_classes=self.num_classes, reduction=self.reduction).numpy().item()
#
#
# class Dice(nn.Module):
#     def __init__(self, reduction='elementwise_mean'):
#         super().__init__()
#         self.reduction = reduction
#
#     def forward(self, yhat, y):
#         yhat = F.softmax(yhat, dim=1)
#         return MF.dice_score(yhat, y.type(torch.LongTensor), reduction=self.reduction).numpy().item()
#
#
# class F1(nn.Module):
#     def __init__(self, num_classes=3, average='macro'):
#         super().__init__()
#         self.average = average
#         self.num_classes = num_classes
#
#     def forward(self, yhat, y):
#         yhat = F.softmax(yhat, dim=1)
#         return MF.f1(yhat, y.type(torch.LongTensor), num_classes=self.num_classes, average=self.average).numpy().item()
