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
        acc = (torch.sum(preds == y)) / (preds.size(0) * preds.size(1) *
                                         preds.size(2))
        return acc.numpy().item()


class IoU(torch.nn.Module):
    def __init__(self, num_classes=3, reduction='mean'):
        super().__init__()
        self.num_classes = num_classes
        self.reduction = reduction


    def forward(self, yhat, y, eps=1e-7):
        true_1_hot = torch.eye(self.num_classes)[y.type(
            torch.LongTensor).squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas1 = F.softmax(yhat, dim=1)

        probas1 = torch.argmax(probas1, dim=1)
        probas = torch.zeros_like(yhat)
        probas[probas1 == 1, 1] = 1
        probas[probas1 == 2, 2] = 1


        true_1_hot = true_1_hot.type(yhat.type())
        dims = (0,) + tuple(range(2, y.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        union = cardinality - intersection
        jacc_loss = (intersection / (union + eps)).mean()
        # return (1 - jacc_loss if self.return_loss else jacc_loss).numpy().item()
        return jacc_loss.numpy().item()


class DiceScore(torch.nn.Module):
    def __init__(self, num_classes=3, smooth=0, reduction='mean'):
        super(DiceScore, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.smooth = smooth

    def forward(self, yhat, y, eps=1e-7):
        true_1_hot = torch.eye(self.num_classes)[y.type(
            torch.LongTensor).squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas1 = F.softmax(yhat, dim=1)
        probas1 = torch.argmax(probas1, dim=1)

        probas = torch.zeros_like(yhat)
        probas[probas1 == 1, 1] = 1
        probas[probas1 == 2, 2] = 1
        true_1_hot = true_1_hot.type(yhat.type())
        dims = (0,) + tuple(range(2, y.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = ((2. * intersection + self.smooth) /
                     (cardinality + eps + self.smooth)).mean()
        return dice_loss.numpy().item()


class FBeta(nn.Module):
    def __init__(self, in_class, beta=1, eps=1e-6):
        super().__init__()

        self.in_class = in_class
        self.beta = beta
        self.eps = eps

    def forward(self, yhat, y):
        yhat = F.softmax(yhat, dim=1)
        preds = torch.argmax(yhat, dim=1)
        preds[preds != self.in_class] = 0
        TP = torch.sum((preds == self.in_class) * (y == self.in_class))
        FN = torch.sum((preds == 0) * (y == self.in_class))
        FP = torch.sum((preds == self.in_class) * (y == 0))

        f_beta = ((1 + self.beta ** 2) * TP) / \
                 ((1 + self.beta ** 2) * TP + (self.beta ** 2) * FN + FP)
        return f_beta.numpy().item()

