import torch
import torch.nn.functional as F
from torch.autograd import Variable





class DiceLoss(torch.nn.Module):
    def __init__(self, num_classes=3, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, logits, true, eps=1e-7):
        true_1_hot = torch.eye(self.num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = (2. * intersection / (cardinality + eps)).mean()
        return 1 - dice_loss


class IoULoss(torch.nn.Module):
    def __init__(self, num_classes=3, reduction='mean'):
        super(IoULoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, logits, true, eps=1e-7):
        true_1_hot = torch.eye(self.num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)

        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        union = cardinality - intersection
        jacc_loss = (intersection / (union + eps)).mean()
        return 1 - jacc_loss


class TrevskyLoss(torch.nn.Module):
    def __init__(self, num_classes=3, reduction='mean'):
        super(TrevskyLoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, logits, true, eps=1e-7):
        true_1_hot = torch.eye(self.num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)

        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        fps = torch.sum(probas * (1 - true_1_hot), dims)
        fns = torch.sum((1 - probas) * true_1_hot, dims)
        num = intersection
        denom = intersection + (alpha * fps) + (beta * fns)
        tversky_loss = (num / (denom + eps)).mean()
        return 1 - tversky_loss


class FocalLoss(torch.nn.Module):
    def __init__(self, weight=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, true):
        ce_loss = F.cross_entropy(logits, true, reduction=self.reduction, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss
