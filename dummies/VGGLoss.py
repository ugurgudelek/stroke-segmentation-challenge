import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from model.ResUnetPlus import ResUnetPlus
import xarray as xr
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


class VGGExtractor(nn.Module):
    def __init__(self, layers=[3, 8, 17, 26], device='cuda'):
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
    def __init__(self, extractor, criterion, K=1):
        super(VGGLoss, self).__init__()
        self.criterion = criterion
        self.K = K
        self.extractor = extractor

    def forward(self, preds, target_in):

        preds = F.softmax(preds, dim=1)
        target1 = target_in[:, 0, :, :]
        target1[torch.logical_and(target_in[:, 1, :, :] == 0, target_in[:, 2, :, :] == 0)] = 255
        target = torch.cat((target1.unsqueeze(1), target_in[:, 1:3, :, :]), dim=1) / 255
        preds = self.extractor(preds)
        target = self.extractor(target)

        N = len(preds)
        losses = 0
        for i in range(self.K):
            for j in range(N):
                losses += (i + 1) * self.criterion(preds[j], target[j])

        coeff = 0.5 * self.K * (self.K + 1)
        loss = losses / (coeff * N)
        return loss


class CombinedVGGLoss(nn.Module):
    def __init__(self, vgg_criterion, combining_criterion, balancing_term=0.1):
        super(CombinedVGGLoss, self).__init__()
        self.vgg_loss = VGGLoss(VGGExtractor(), vgg_criterion)
        self.main_loss = combining_criterion
        self.mu = balancing_term

    def forward(self, pred, target):
        loss = self.main_loss(pred, target) + self.mu * self.main_loss(pred, target)
        return loss


if __name__ == '__main__':

    def visualize_preds(layer, batch_no, size1, size2, name):
        layer = layer[:, batch_no, :, :]
        shape1 = layer.shape[1]
        shape2 = layer.shape[2]

        img = np.zeros((shape1 * size1, shape2 * size2))
        k = 0
        for i in range(size1):
            for j in range(size2):
                img[i * shape1:(i + 1) * shape1, j * shape2:(j + 1) * shape2] = layer[k, :, :]
                k += 1

        plt.figure()
        plt.title(name)
        plt.imshow(1 - img, cmap='gray')


    def visualize_vgg(layer, batch_no, size1, size2, name):
        layer = layer[batch_no, :, :, :]
        shape1 = layer.shape[1]
        shape2 = layer.shape[2]

        img = np.zeros((shape1 * size1, shape2 * size2))
        k = 0
        for i in range(size1):
            for j in range(size2):
                img[i * shape1:(i + 1) * shape1, j * shape2:(j + 1) * shape2] = layer[k, :, :]
                k += 1

        plt.figure()
        plt.title(name)
        plt.imshow(img)


    root = 'D:/Gorkem Can Ates/PycharmProjects/stroke-segmentation-challenge/input/stroke/'
    dataset = xr.open_dataset(root + 'nc/stroke-segmentation.nc')

    train_ids, test_ids = train_test_split(dataset.id.values,
                                           test_size=0.25,
                                           random_state=42,
                                           shuffle=True)

    train_dataset = dataset.sel({'id': train_ids})
    test_dataset = dataset.sel({'id': test_ids})

    file_name = 'D:/Gorkem Can Ates/PycharmProjects/stroke-segmentation-challenge/projects/stroke/ResUnetPlus-gn-k05-IoU-lr1e-4-bsize-5-pretrained-0-dataaug-2-TL-0/checkpoints/73'
    PATH = file_name + '/model-optim.pth'
    model = ResUnetPlus(in_features=3,
                        out_features=3,
                        k=0.5,
                        norm_type='gn',
                        upsample_type='bilinear')
    device = 'cuda'
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    model.to(device)

    images = torch.as_tensor(test_dataset.image.values, dtype=torch.float32)
    masks = torch.as_tensor(test_dataset.mask.values, dtype=torch.int)
    k = 4
    data = images[0:k].to(device)
    target_in = masks[0:k].to(device)
    target1 = target_in[:, 0, :, :]
    target1[torch.logical_and(target_in[:, 1, :, :] == 0, target_in[:, 2, :, :] == 0)] = 255
    target = torch.cat((target1.unsqueeze(1), target_in[:, 1:3, :, :]), dim=1) / 255

    preds = F.softmax(preds, dim=1)

    vgg = VGGExtractor()

    target_results = vgg(target.to(device))
    pred_results = vgg(preds.to(device))

    preds = model(data)

    mse_loss = nn.MSELoss()
    criterion = VGGLoss(mse_loss)

    loss = criterion(preds, target_in)
    loss.backward()

    pred_f1 = pred_results[0].detach().cpu().numpy()
    pred_f2 = pred_results[1].detach().cpu().numpy()
    pred_f3 = pred_results[2].detach().cpu().numpy()
    pred_f4 = pred_results[3].detach().cpu().numpy()

    target_f1 = target_results[0].detach().cpu().numpy()
    target_f2 = target_results[1].detach().cpu().numpy()
    target_f3 = target_results[2].detach().cpu().numpy()
    target_f4 = target_results[3].detach().cpu().numpy()

    batch_no = 1
    visualize_vgg(pred_f1, batch_no, 8, 8, 'pred_f1')
    visualize_vgg(pred_f2, batch_no, 8, 16, 'pred_f2')
    visualize_vgg(pred_f3, batch_no, 16, 16, 'pred_f3')
    visualize_vgg(pred_f4, batch_no, 16, 32, 'pred_f4')

    visualize_vgg(target_f1, batch_no, 8, 8, 'target_f1')
    visualize_vgg(target_f2, batch_no, 8, 16, 'target_f2')
    visualize_vgg(target_f3, batch_no, 16, 16, 'target_f3')
    visualize_vgg(target_f4, batch_no, 16, 32, 'target_f4')

    plt.figure()
    plt.title('pred')
    plt.imshow(preds[batch_no].detach().cpu().permute(1, 2, 0))

    plt.figure()
    plt.title('target')
    plt.imshow(target[batch_no].detach().cpu().permute(1, 2, 0))
