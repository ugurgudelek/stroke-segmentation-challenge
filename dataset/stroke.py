# -*- encoding: utf-8 -*-
# @File    :   stroke.py
# @Time    :   2021/05/16 01:40:50
# @Author  :   Ugur Gudelek
# @Contact :   ugurgudelek@gmail.com
# @Desc    :   None

import xarray as xr
import numpy as np
from PIL import Image

from pathlib import Path
from tqdm import tqdm

import torch
from sklearn.model_selection import train_test_split

import albumentations as A
import albumentations.pytorch as Ap

from berries.datasets.base import BaseTorchDataset


class Stroke:
    WIDTH, HEIGHT = (512, 512)
    CLASSES = {'no-stroke': 0, 'stroke': 1}

    def __init__(self, root: Path):
        self.root = root

    def for_classification(self) -> xr.Dataset:

        # Read no stroke images
        no_stroke_image_paths = list(
            (self.root / 'raw/INMEYOK/PNG').glob('*.png'))
        no_stroke_images = Stroke.read_images(
            image_paths=no_stroke_image_paths)

        # Read stroke images
        stroke_image_paths = list((self.root / 'raw/KANAMA/PNG').glob('*.png'))
        stroke_image_paths.extend(
            list((self.root / 'raw/ISKEMI/PNG').glob('*.png')))
        stroke_images = Stroke.read_images(image_paths=stroke_image_paths)

        no_stroke_data = xr.Dataset(data_vars=dict(
            image=(['id', 'c', 'x', 'y'], list(no_stroke_images.values())),
            label=(['id'], [Stroke.CLASSES['no-stroke']] * len(no_stroke_images))),
            coords=dict(id=('id', list(no_stroke_images.keys())),
                        x=('x', range(Stroke.WIDTH)),
                        y=('y', range(Stroke.HEIGHT)),
                        c=('c', ['R', 'G', 'B'])))  # yapf: disable

        stroke_data = xr.Dataset(data_vars=dict(
            image=(['id', 'c', 'x', 'y'], list(stroke_images.values())),
            label=(['id'], [Stroke.CLASSES['stroke']] * len(stroke_images))),
            coords=dict(id=('id', list(stroke_images.keys())),
                        x=('x', range(Stroke.WIDTH)),
                        y=('y', range(Stroke.HEIGHT)),
                        c=('c', ['R', 'G', 'B'])))  # yapf: disable

        dataset = xr.concat([no_stroke_data, stroke_data], dim='id')

        save_path = self.root / Path('nc')
        save_path.mkdir(exist_ok=True)
        dataset.to_netcdf(path=save_path / 'stroke-classification.nc',
                          engine='netcdf4')

        return dataset

    def for_segmentation(self) -> xr.Dataset:

        # Read 	hemorrhage
        hemorrhage_image_paths = list(
            (self.root / 'raw/KANAMA/PNG').glob('*.png'))
        hemorrhage_mask_paths = list(
            (self.root / 'raw/KANAMA/MASK').glob('*.png'))
        hemorrhage_images = Stroke.read_images(
            image_paths=hemorrhage_image_paths)
        hemorrhage_masks = Stroke.read_images(
            image_paths=hemorrhage_mask_paths)

        hemorrhage_data = xr.Dataset(data_vars=dict(
            image=(['id', 'c', 'x', 'y'], list(hemorrhage_images.values())),
            mask=(['id', 'c', 'x', 'y'], [hemorrhage_masks[key] for key, _ in hemorrhage_images.items()])),
            coords=dict(id=('id', list(hemorrhage_images.keys())),
                        x=('x', range(Stroke.WIDTH)),
                        y=('y', range(Stroke.HEIGHT)),
                        c=('c', ['R', 'G', 'B'])))  # yapf: disable

        # Read 	ischemia
        ischemia_image_paths = list(
            (self.root / 'raw/ISKEMI/PNG').glob('*.png'))
        ischemia_mask_paths = list(
            (self.root / 'raw/ISKEMI/MASK').glob('*.png'))
        ischemia_images = Stroke.read_images(image_paths=ischemia_image_paths)
        ischemia_masks = Stroke.read_images(image_paths=ischemia_mask_paths)

        ischemia_data = xr.Dataset(data_vars=dict(
            image=(['id', 'c', 'x', 'y'], list(ischemia_images.values())),
            mask=(['id', 'c', 'x', 'y'], [ischemia_masks[key] for key, _ in ischemia_images.items()])),
            coords=dict(id=('id', list(ischemia_images.keys())),
                        x=('x', range(Stroke.WIDTH)),
                        y=('y', range(Stroke.HEIGHT)),
                        c=('c', ['R', 'G', 'B'])))  # yapf: disable

        dataset = xr.concat([hemorrhage_data, ischemia_data], dim='id')

        save_path = self.root / Path('nc')
        save_path.mkdir(exist_ok=True)
        dataset.to_netcdf(path=save_path / 'stroke-segmentation.nc',
                          engine='netcdf4')

    @staticmethod
    def read_images(image_paths: list[Path]) -> dict[str, np.ndarray]:
        images = {}
        for img_path in tqdm(image_paths):
            img_id = img_path.stem

            # Read image and drop alpha channel
            img = Image.open(img_path).convert('RGB')

            if img.size != (Stroke.WIDTH, Stroke.HEIGHT):
                # center crop
                w, h = img.size
                left = (w - Stroke.WIDTH) / 2
                top = (h - Stroke.HEIGHT) / 2
                right = (w + Stroke.WIDTH) / 2
                bottom = (h + Stroke.HEIGHT) / 2

                img = img.crop((left, top, right, bottom))

            img = np.asarray(img)
            img = np.transpose(img, (2, 0, 1))  # channel, width, height
            images[img_id] = np.asarray(img)

        return images


class StrokeClassificationDataset:
    def __init__(self, root, transform, test_size):
        self.root = root  # Path('./input/stroke')

        dataset = xr.open_dataset(self.root / 'nc/stroke-classification.nc')

        train_ids, test_ids = train_test_split(dataset.id.values,
                                               test_size=test_size,
                                               random_state=42,
                                               shuffle=True)

        self.train_dataset = dataset.sel({'id': train_ids})
        self.test_dataset = dataset.sel({'id': test_ids})

        self.trainset = StrokeClassificationTorch(
            images=self.train_dataset.image.values,
            targets=self.train_dataset.label.values,
            transform=transform)
        self.testset = StrokeClassificationTorch(
            images=self.test_dataset.image.values,
            targets=self.test_dataset.label.values,
            transform=transform)


class StrokeClassificationTorch(BaseTorchDataset):
    def __init__(self, images, targets, transform):
        self.images = images
        self.targets = targets
        self.transform = transform

    def __getitem__(self, ix):
        image = self.images[ix]
        label = self.targets[ix]
        image = torch.as_tensor(image, dtype=torch.float32) / 255.  # delete
        # Transform images
        if self.transform is not None:
            # image = self.transform(image=image)['image'] # true
            image = self.transform(image)  # delete
        label = torch.as_tensor(label, dtype=torch.long)

        return {'data': image, 'target': label}

    def __len__(self):
        return len(self.targets)


class StrokeSegmentationDataset:
    # KANAMA: channel Green - 1
    # ISKEMI: channel Blue  - 2
    def __init__(self, root, transform=None, test_size=0.25):
        self.root = root  # Path('./input/stroke')

        dataset = xr.open_dataset(self.root / 'nc/stroke-segmentation.nc')

        train_ids, test_ids = train_test_split(dataset.id.values,
                                               test_size=test_size,
                                               random_state=42,
                                               shuffle=True)

        self.train_dataset = dataset.sel({'id': train_ids})
        self.test_dataset = dataset.sel({'id': test_ids})

        self.trainset = StrokeSegmentationTorch(
            images=self.train_dataset.image.values,
            masks=self.train_dataset.mask.values,
            transform=transform)
        self.testset = StrokeSegmentationTorch(
            images=self.test_dataset.image.values,
            masks=self.test_dataset.mask.values,
            transform=None)


class StrokeSegmentationTorch(BaseTorchDataset):
    def __init__(self, images, masks, transform):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __getitem__(self, ix):
        image = self.images[ix]
        mask = self.masks[ix]

        if self.transform is not None:
            sample = self.transform(image=image, mask=mask)
            image = sample['image']
            mask = sample['mask']

            image = self.transform(image)
            mask = self.transform(mask)

        return {'data': image, 'target': mask}

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    # Stroke(root=Path('./input/stroke')).for_classification()
    # Stroke(root=Path('./input/stroke')).for_segmentation()

    # Declare an augmentation pipeline
    transform = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.05,
                           scale_limit=0.05,
                           rotate_limit=15,
                           p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        Ap.ToTensor()
    ])

    d = StrokeSegmentationDataset(root=Path('./input/stroke'),
                                  test_size=0.2,
                                  transform=transform)

    sample = d.trainset[10]
    print()

    StrokeSegmentationDataset(root=Path('./input/stroke'), test_size=0.25)
