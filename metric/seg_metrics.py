import torch
from torch import nn
import torch.nn.functional as F
import torchmetrics
import torchmetrics.functional as MF


class Accuracy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, yhat, y):
        preds = (torch.argmax(yhat, dim=1))
        acc = (torch.sum(preds == y)) / (preds.size(0) * preds.size(1) * preds.size(2))
        return acc.numpy().item()


class IoU(nn.Module):
    def __init__(self, num_classes=3, reduction='elementwise_mean'):
        super().__init__()
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, yhat, y):
        yhat = F.softmax(yhat, dim=1)
        return MF.iou(yhat, y.type(torch.LongTensor), num_classes=self.num_classes, reduction=self.reduction).numpy().item()


class Dice(nn.Module):
    def __init__(self, reduction='elementwise_mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, yhat, y):
        yhat = F.softmax(yhat, dim=1)
        return MF.dice_score(yhat, y.type(torch.LongTensor), reduction=self.reduction).numpy().item()


class F1(nn.Module):
    def __init__(self, num_classes=3, average='macro'):
        super().__init__()
        self.average = average
        self.num_classes = num_classes

    def forward(self, yhat, y):
        yhat = F.softmax(yhat, dim=1)
        return MF.f1(yhat, y.type(torch.LongTensor), num_classes=self.num_classes, average=self.average).numpy().item()


# class IoU(torch.nn.Module):
#     def __init__(self, num_classes=3, reduction='mean'):
#         super().__init__()
#         self.num_classes = num_classes
#         self.reduction = reduction
#
#     def forward(self, yhat, y, eps=1e-7):
#         true_1_hot = torch.eye(self.num_classes)[y.type(torch.LongTensor).squeeze(1)]
#         true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
#         probas1 = F.softmax(yhat, dim=1)
#         probas1 = torch.argmax(probas1, dim=1)
#
#         probas = torch.zeros((len(yhat), 3, 512, 512))
#         probas[probas1 == 1, 1] = 1
#         probas[probas1 == 2, 2] = 1
#
#         true_1_hot = true_1_hot.type(yhat.type())
#         dims = (0,) + tuple(range(2, y.ndimension()))
#         intersection = torch.sum(probas * true_1_hot, dims)
#         cardinality = torch.sum(probas + true_1_hot, dims)
#         union = cardinality - intersection
#         jacc_loss = (intersection / (union + eps)).mean()
#         return jacc_loss.numpy().item()

class F1_Class1(nn.Module):
    def __init__(self, in_class=1, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.in_class = in_class

    def forward(self, yhat, y):
        yhat = F.softmax(yhat, dim=1)
        preds = torch.argmax(yhat, dim=1)
        preds[preds != self.in_class] = 0
        TP = torch.sum((preds == self.in_class) * (y == self.in_class))
        FN = torch.sum((preds == 0) * (y == self.in_class))
        FP = torch.sum((preds == self.in_class) * (y == 0))
        return (TP / (TP + 0.5 * (FP + FN) + self.eps)).numpy().item()


class F1_Class2(nn.Module):
    def __init__(self, in_class=2, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.in_class = in_class

    def forward(self, yhat, y):
        yhat = F.softmax(yhat, dim=1)
        preds = torch.argmax(yhat, dim=1)
        preds[preds != self.in_class] = 0
        TP = torch.sum((preds == self.in_class) * (y == self.in_class))
        FN = torch.sum((preds == 0) * (y == self.in_class))
        FP = torch.sum((preds == self.in_class) * (y == 0))
        return (TP / (TP + 0.5 * (FP + FN) + self.eps)).numpy().item()


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


class Specificity(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, yhat, y):
        preds = (torch.argmax(yhat, dim=1)).detach()
        FP = torch.sum((preds == 1) * (y == 0))
        TN = torch.sum((preds == 0) * (y == 0))
        return 1 - (FP / (FP + TN + self.eps)).cpu().numpy().item()


class MeanMetric(nn.Module):

    def __init__(self):
        super().__init__()
        self.recall = Recall()
        self.specificity = Specificity()

    def forward(self, yhat, y):
        recall = self.recall(yhat, y)
        specificity = self.specificity(yhat, y)
        return (recall + specificity) / 2
