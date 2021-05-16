# -*- encoding: utf-8 -*-
# @File    :   stroke-experiment.py
# @Time    :   2021/05/16 00:45:06
# @Author  :   Ugur Gudelek, Gorkem Can Ates
# @Contact :   ugurgudelek@gmail.com
# @Desc    :   None
import torch
from torch.utils.data import Dataset
from pathlib import Path
import os
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader


class StrokeClassificationDataset:
    def __init__(self, root):
        self.root = root

        self.trainset = StrokeClassificationTorch()
        self.testset = StrokeClassificationTorch()


class StrokeClassificationTorch(Dataset):
    def __init__(self, image_dir, targetfolder, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.targetfolder = targetfolder
        self.images = os.listdir(image_dir)

    def __getitem__(self, item):
        image_path = os.path.join(self.image_dir, self.images[item])
        image = np.array(Image.open(image_path).convert('RGB'))
        if targetfolder == 'INME-YOK':
            target = np.zeros(1)
        else:
            target = np.ones(1)

        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    dataset = StrokeClassificationDataset()