import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision


class DiceLoss(torch.nn.Module):
    def __init__(self, num_classes=3, smooth=0, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.smooth = smooth

    def forward(self, logits, true, eps=1e-7):
        true_1_hot = torch.eye(self.num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = ((2. * intersection + self.smooth) / (cardinality + eps + self.smooth)).mean()
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
    def __init__(self, weight=None, gamma=2, alpha=1, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, logits, true):
        ce_loss = F.cross_entropy(logits, true, reduction=self.reduction, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * ((1 - pt) ** self.gamma) * ce_loss).mean()
        return focal_loss


class EnhancedMixingLoss(torch.nn.Module):
    def __init__(self, gamma=1.1, alpha=0.48, smooth=1., epsilon=1e-7):
        super(EnhancedMixingLoss, self).__init__()
        self.focal_loss = FocalLoss(gamma=gamma, alpha=alpha)
        self.dice_loss = DiceLoss(smooth=smooth)

    def forward(self, logits, true):
        fcloss = self.focal_loss(logits, true)
        dcloss = self.dice_loss(logits, true)

        return fcloss - torch.log(dcloss)


########### VGGLOSS ###########
class VGGExtractor(nn.Module):
    def __init__(self, device, layers=[3, 8, 17, 26]):
        super(VGGExtractor, self).__init__()
        self.layers = layers
        self.mu = torch.tensor([0.485, 0.456, 0.406], requires_grad=False).view((1, 3, 1, 1)).to(device)
        self.sigma = torch.tensor([0.229, 0.224, 0.225], requires_grad=False).view((1, 3, 1, 1)).to(device)
        features = torchvision.models.vgg19(pretrained=True).features[:27].to(device)
        for param in features.parameters():
            param.requires_grad = False

        self.features = nn.ModuleList(list(features)).eval()

    def forward(self, x):
        x = (x - self.mu) / self.sigma

        results = []
        for i, vgg in enumerate(self.features):
            x = vgg(x)
            if i in self.layers:
                results.append(x)

        return results


class VGGLoss(nn.Module):
    def __init__(self, extractor, criterion, num_classes=3, K=1, device='cuda'):
        super(VGGLoss, self).__init__()
        self.extractor = extractor
        self.criterion = criterion
        self.num_classes = num_classes
        self.K = K
        self.device = device

    def forward(self, preds, target):
        probs = F.softmax(preds, dim=1)
        target_hot = torch.eye(self.num_classes).to(self.device)[target.squeeze(1)]
        target_hot = target_hot.permute(0, 3, 1, 2).float()

        probs = self.extractor(probs)
        target_hot = self.extractor(target_hot)

        N = len(probs)
        losses = 0
        for i in range(self.K):
            for j in range(N):
                losses += (i + 1) * self.criterion(probs[j], target_hot[j])

        coeff = 0.5 * self.K * (self.K + 1)
        loss = losses / (coeff * N)
        return loss


class CombinedVGGLoss(nn.Module):
    def __init__(self, main_criterion, vgg_criterion, balance=[1, 0.1], reduction='mean', device='cuda'):
        super(CombinedVGGLoss, self).__init__()
        self.vgg_loss = VGGLoss(VGGExtractor(device=device), vgg_criterion, device=device)
        self.main_loss = main_criterion
        self.balance = balance
        self.reduction = reduction

    def forward(self, pred, target):
        loss = self.balance[0] * self.main_loss(pred, target) + self.balance[1] * self.vgg_loss(pred, target)
        return loss
