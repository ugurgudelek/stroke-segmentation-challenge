import torch
from torch.autograd import Variable


class WeightedBCELoss(torch.nn.Module):
    def __init__(self, weights, reduction='mean'):
        super(WeightedBCELoss, self).__init__()
        self.weights = weights
        self.reduction = reduction

    def forward(self, output, target):
        output = Variable(torch.clamp(output, min=1e-8, max=1 - 1e-8))
        target = Variable(target)

        if self.weights is not None:
            assert len(self.weights) == 2 ## Sadece iki değişkenli liste
            loss = self.weights[1] * (target * torch.log(output)) + \
                   self.weights[0] * ((1 - target) * torch.log(1 - output))
        else:
            loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

        return torch.neg(torch.mean(loss))


