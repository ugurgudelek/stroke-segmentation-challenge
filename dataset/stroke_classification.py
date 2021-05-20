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
import xarray as xr
from sklearn.model_selection import train_test_split


class StrokeClassificationDataset:
    def __init__(self, root, transform, test_size):
        self.root = root  # Path('./input/stroke')
        self.dataset = self.root / 'nc/stroke.nc'

        ds = xr.open_dataset(self.dataset)

        index_train, index_test = train_test_split(np.arange(len(ds.label)),
                                                   test_size=test_size,
                                                   random_state=42,
                                                   shuffle=True)

        self.train_dataset = ds.isel({'id': index_train})
        self.test_dataset = ds.isel({'id': index_test})

        self.trainset = StrokeClassificationTorch(images=self.train_dataset.image.values,
                                                  targets=self.train_dataset.label.values,
                                                  transform=transform)
        self.testset = StrokeClassificationTorch(images=self.test_dataset.image.values,
                                                 targets=self.test_dataset.label.values,
                                                 transform=transform)


class StrokeClassificationTorch(Dataset):
    def __init__(self, images, targets, transform):
        self.images = images
        self.targets = targets
        self.transform = transform

    def __getitem__(self, ix):
        x = self.images[ix]
        y = self.targets[ix]

        # Transform images
        x = torch.as_tensor(x, dtype=torch.float32)/255.
        y = torch.as_tensor(y, dtype=torch.long)
        if self.transform is not None:
            x = self.transform(x)

        return {'data': x, 'target': y }

    def __len__(self):
        return len(self.targets)


if __name__ == '__main__':
    dataset = StrokeClassificationDataset()
