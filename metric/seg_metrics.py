import torch
from torch import nn
import torch.nn.functional as F


class Accuracy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, yhat, y):
        preds = (torch.argmax(yhat, dim=1))
        acc = (torch.sum(preds == y)) / (preds.size(0) * preds.size(1) * preds.size(2))
        return acc.numpy().item()


class IoU(torch.nn.Module):
    def __init__(self, num_classes=3, reduction='mean'):
        super(IoU, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, logits, true, eps=1e-7):
        true_1_hot = torch.eye(self.num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
        probas = torch.argmax(probas, dim=1)

        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        union = cardinality - intersection
        jacc_loss = (intersection / (union + eps)).mean()
        return jacc_loss


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
