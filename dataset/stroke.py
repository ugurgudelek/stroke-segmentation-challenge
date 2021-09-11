# -*- encoding: utf-8 -*-
# @File    :   stroke.py
# @Time    :   2021/05/16 01:40:50
# @Author  :   Ugur Gudelek
# @Contact :   ugurgudelek@gmail.com
# @Desc    :   None

import xarray as xr
import numpy as np
from PIL import Image
import pydicom as dicom
import cv2 as cv
import matplotlib.pyplot as plt
import scipy
import random

from pathlib import Path
from tqdm import tqdm

import torch
from sklearn.model_selection import train_test_split

import torchvision.transforms.functional as TF
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
    def from_path_classification(png_path: Path,
                                 infer_label=False,
                                 hold_prop=1.):

        image_paths = list((png_path).glob('*.png'))
        if hold_prop < 1.:
            image_paths = [
                p for p in image_paths if random.random() < hold_prop
            ]
        images = Stroke.read_images(image_paths=image_paths)

        labels = {
            name: Stroke.CLASSES['no-stroke']
            for name, _ in images.items()
        }

        if infer_label:
            label = dict()
            for name in images.keys():
                for p in Path('input/teknofest/raw').glob(f'**/{name}.png'):
                    if p.parent.parent.stem in ['KANAMA', 'ISKEMI']:
                        label = Stroke.CLASSES['stroke']
                    else:
                        label = Stroke.CLASSES['no-stroke']

                    labels[name] = label
                    break

        data = xr.Dataset(data_vars=dict(
            image=(['id', 'c', 'x', 'y'], list(images.values())),
            label=(['id'], [labels[name] for name in images.keys()])),
            coords=dict(id=('id', list(images.keys())),
                        x=('x', range(Stroke.WIDTH)),
                        y=('y', range(Stroke.HEIGHT)),
                        c=('c', ['R', 'G', 'B'])))  # yapf: disable

        return data

    @staticmethod
    def from_path_segmentation(png_path: Path,
                               mask_path: Path = None,
                               infer_mask=False,
                               hold_prop=1.) -> xr.Dataset:
        image_paths = list(png_path.glob('*.png'))
        if hold_prop < 1.:
            image_paths = [
                p for p in image_paths if random.random() < hold_prop
            ]
        images = Stroke.read_images(image_paths=image_paths)

        if mask_path is None and infer_mask == False:
            data = xr.Dataset(data_vars=dict(
                image=(['id', 'c', 'x', 'y'], list(images.values())),
                mask=(['id', 'c', 'x', 'y'], [np.zeros_like(image) for _, image in images.items()])
                ),
                coords=dict(id=('id', list(images.keys())),
                            x=('x', range(Stroke.WIDTH)),
                            y=('y', range(Stroke.HEIGHT)),
                            c=('c', ['R', 'G', 'B'])))  # yapf: disable

        else:
            if infer_mask:  # while infering, it creates zeros array for not-found images
                mask_paths = []
                for name in images.keys():
                    for p in Path('input/teknofest/raw').glob(
                            f'**/{name}.png'):
                        if p.parent.stem == 'MASK':
                            mask_paths.append(p)
                            break
            else:
                mask_paths = list(mask_path.glob('*.png'))
            masks = Stroke.read_masks(mask_paths=mask_paths)
            data = xr.Dataset(data_vars=dict(
                image=(['id', 'c', 'x', 'y'], list(images.values())),
                mask=(['id', 'c', 'x', 'y'], [masks.get(key, np.zeros_like(image)) for key, image in images.items()])
                ),
                coords=dict(id=('id', list(images.keys())),
                            x=('x', range(Stroke.WIDTH)),
                            y=('y', range(Stroke.HEIGHT)),
                            c=('c', ['R', 'G', 'B'])))  # yapf: disable

        return data

    @staticmethod
    def read_masks(mask_paths: list[Path]) -> np.ndarray:
        masks = {}
        for mask_path in tqdm(mask_paths):

            mask = Image.open(mask_path).convert('RGB')  # HWC

            if mask.size != (Stroke.WIDTH, Stroke.HEIGHT):
                # center crop
                w, h = mask.size
                left = (w - Stroke.WIDTH) / 2
                top = (h - Stroke.HEIGHT) / 2
                right = (w + Stroke.WIDTH) / 2
                bottom = (h + Stroke.HEIGHT) / 2

                mask = mask.crop((left, top, right, bottom))

            mask = np.transpose(mask, (2, 0, 1))  # CHW
            mask = mask.astype(np.uint8)
            masks[mask_path.stem] = mask
        return masks  # BCHW

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

        return images  # BCHW

    def initialize_undo_cropping(self, png_path):
        image_paths = list(png_path.glob('*.png'))

        crop_indexes = {}
        original_sizes = {}
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

                crop_indexes[img_id] = (left, top, right, bottom)
                original_sizes[img_id] = (w, h)

        self.crop_indexes = crop_indexes
        self.original_sizes = original_sizes

    def undo_cropping(self, images):

        for img_id, image in images.items():

            image = images[img_id]  # CHW

            if self.crop_indexes.get(
                    img_id) is None:  # We have already 512x512 image
                continue
            print(
                f'Undoing cropping because original shape was {self.original_sizes[img_id]} for {img_id}.'
            )

            crop_l, crop_t, crop_r, crop_b = self.crop_indexes[img_id]
            new_w, new_h = self.original_sizes[img_id]

            pil_image = Image.fromarray(np.transpose(image, (1, 2, 0)))

            new_pil_image = Image.new(pil_image.mode, (new_w, new_h),
                                      (0, 0, 0))

            new_pil_image.paste(pil_image, (int(crop_l), int(crop_t)))

            images[img_id] = np.transpose(np.asarray(new_pil_image), (2, 0, 1))
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
        image = self.images[ix]  # CHW
        label = self.targets[ix]
        # image = torch.as_tensor(image, dtype=torch.float32) / 255.  # delete
        # Transform images
        if self.transform is not None:
            # image = self.transform(image=image)['image']  # true
            image = Image.fromarray(np.transpose(image, (1, 2, 0)))  # HWC

            image = self.transform(image)  # delete

            image = np.asarray(image, dtype=np.float32)
            image = np.transpose(image, (2, 0, 1))  # CHW

        image = (image / 255.).astype(np.float32)

        label = torch.as_tensor(label, dtype=torch.long)

        return {'data': image, 'target': label}

    def __len__(self):
        return len(self.targets)


class StrokeSegmentationDataset:
    # KANAMA: channel Green - 1
    # ISKEMI: channel Blue  - 2
    def __init__(self,
                 root,
                 transform=None,
                 val_transform=None,
                 test_size=0.25,
                 debug=False):
        self.root = root  # Path('./input/stroke')

        dataset = xr.open_dataset(self.root / 'nc/stroke-segmentation.nc')

        if debug:
            _id = np.random.permutation(dataset.id.values)[:64]
            dataset = dataset.sel({'id': _id})
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
            transform=val_transform)


class StrokeSegmentationTorch(BaseTorchDataset):
    def __init__(self, images, masks, transform):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __getitem__(self, ix):
        image = self.images[ix]  # CHW (3, 512, 512)
        mask = self.masks[ix]  # CHW (3, 512, 512)

        C, H, W = image.shape

        image = np.transpose(image, axes=(1, 2, 0))  # HWC
        mask = np.transpose(mask, axes=(1, 2, 0))  # HWC
        mask = cv.cvtColor(mask, cv.COLOR_RGB2GRAY)  # HW
        # mask should be 012 image
        # this trick is necessary because of the way the model was trained
        _mask = mask.copy()
        mask[_mask == 1] = 2
        mask[_mask == 2] = 1

        if self.transform is not None:
            sample = self.transform(image=image, mask=mask)
            image = sample['image']
            mask = sample['mask']

            image = image / 255.
            mask = mask.long()

        return {'data': image, 'target': mask}

    def __len__(self):
        return len(self.images)


class StrokeSegmentationTorchFrozen(BaseTorchDataset):
    def __init__(self, images, transform):
        self.images = images
        self.transform = transform

    def __getitem__(self, ix):
        image = self.images[ix]  # CHW (3, 512, 512)

        C, H, W = image.shape
        image = np.transpose(image, axes=(1, 2, 0))  # HWC
        mask = np.zeros(shape=(H, W))

        if self.transform is not None:
            sample = self.transform(image=image, mask=mask)
            image = sample['image']
            mask = sample['mask']

            image = image / 255.
            mask = mask.long()

        return {'data': image, 'target': mask}

    def __len__(self):
        return len(self.images)


class StrokeDataset:
    CLASSES = {'no-stroke': 0, 'hemorrhage': 1, 'ischemia': 2}

    def __init__(self,
                 root=Path('./input/teknofest'),
                 transform=None,
                 val_transform=None,
                 test_size=0.25,
                 debug=False):

        self.root = root
        self.transform = transform
        self.val_transform = val_transform
        self.test_size = test_size

        no_stroke_paths = [{
            'name': image_path.stem,
            'image': image_path,
            'mask': None,
            'label': self.CLASSES['no-stroke']
        } for image_path in (self.root / 'raw/INMEYOK/DICOM').glob('*.dcm')]

        hemorrhage_paths = [{
            'name': image_path.stem,
            'image': image_path,
            'mask': mask_path,
            'label': self.CLASSES['hemorrhage']
        } for image_path, mask_path in zip((
            self.root / 'raw/KANAMA/DICOM').glob('*.dcm'), (
                self.root / 'raw/KANAMA/MASK').glob('*.png'))]

        ischemia_paths = [{
            'name': image_path.stem,
            'image': image_path,
            'mask': mask_path,
            'label': self.CLASSES['ischemia']
        } for image_path, mask_path in zip((
            self.root / 'raw/ISKEMI/DICOM').glob('*.dcm'), (
                self.root / 'raw/ISKEMI/MASK').glob('*.png'))]

        self.paths = [*no_stroke_paths, *hemorrhage_paths, *ischemia_paths]

        if debug:
            self.paths = np.random.permutation(self.paths)[:100].tolist()

        self.train_paths, self.test_paths = train_test_split(
            self.paths, test_size=test_size, random_state=42, shuffle=True)

        self.trainset = StrokeTorchDataset(paths=self.train_paths,
                                           transform=self.transform)
        self.testset = StrokeTorchDataset(paths=self.test_paths,
                                          transform=self.val_transform)

    @staticmethod
    def overlay(image: np.ndarray,
                mask: np.ndarray,
                label: int = None,
                name: str = None,
                alpha=0.8) -> np.ndarray:
        image_rgb = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        mask_rgb = StrokeDataset.mask_gray2channel(mask, only_red=True)

        image_patch_mask = np.where(mask_rgb != 0, image_rgb, 0)
        image_patch_bg = np.where(mask_rgb == 0, image_rgb, 0)
        overlay = image_patch_bg + np.uint8((1 - alpha) * (image_patch_mask) +
                                            (alpha) * (mask_rgb))

        overlay = overlay.astype(np.uint8)
        plt.imshow(overlay)
        if name is not None and label is not None:
            plt.suptitle(f'Name: {name}, Label: {label}')
        plt.axis('off')
        return overlay

    @staticmethod
    def mask_gray2channel(mask: np.ndarray, only_red=True) -> np.ndarray:
        mask_rgb = np.zeros((*mask.shape, 3))
        if only_red:
            mask_rgb[:, :,
                     0] = (((mask == 1) | (mask == 2)) * 255).astype(np.uint8)
            return mask_rgb

        mask_rgb[:, :, 1] = ((mask == 1) * 255).astype(np.uint8)
        mask_rgb[:, :, 2] = ((mask == 2) * 255).astype(np.uint8)
        return mask_rgb  # HWC


class StrokeTorchDataset(BaseTorchDataset):
    WIDTH, HEIGHT = (512, 512)

    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, ix):

        path = self.paths[ix]

        name = path['name']
        image = self.read_image(path['image'])
        mask = self.read_mask(
            path['mask']) if path['mask'] is not None else np.zeros_like(image)
        label = path['label']

        if self.transform is not None:
            sample = self.transform(image=image, mask=mask)
            image = sample['image']
            mask = sample['mask']

            image = image / 255.
            mask = mask.long()

        return {'image': image, 'mask': mask, 'label': label, 'name': name}

    def getitem_by_name(self, name):

        for ix, path in enumerate(self.paths):
            if path['name'] == name:
                return self.__getitem__(ix=ix)

    @staticmethod
    def read_mask(path: Path) -> np.ndarray:
        mask = cv.imread(str(path))
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        mask = mask.astype(np.uint8)
        return mask  # HW

    @staticmethod
    def read_scan(path: Path):
        return dicom.dcmread(path)

    def preprocess(self, path_dict, resample=True, center=True):
        scan_path = path_dict['image']
        mask_path = path_dict['mask']

        scan = self.read_scan(scan_path)
        image = self.scan2hu(scan=scan)

        if mask_path is None:
            mask = np.zeros_like(image)
        else:
            mask = self.read_mask(mask_path)

        image = self.normalize(image=image)

        if center:
            image = self.zero_center(image)

        if resample:
            image, spacing = self.resample(image=image,
                                           scan=scan,
                                           new_spacing=np.array([1., 1.]))
            mask, spacing = self.resample(image=mask,
                                          scan=scan,
                                          new_spacing=np.array([1., 1.]))

        return image, mask

    @staticmethod
    def scan2hu(scan):
        image = scan.pixel_array
        image = image.astype(np.int16)
        # Set outside-of-scan pixels to 0
        # The intercept is usually -1024, so air is approximately 0
        # image[image == -2000] = 0

        # Convert to Hounsfield units (HU)
        intercept = scan.RescaleIntercept
        slope = scan.RescaleSlope

        image = np.int16(
            image.astype(np.float64) * slope) + np.int16(intercept)
        return image

    @staticmethod
    def resample(image, scan, new_spacing=np.array([1., 1.])):
        # Determine current pixel spacing
        spacing = np.array(scan.PixelSpacing, dtype=np.float64)

        resize_factor = spacing / new_spacing
        new_real_shape = image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image.shape
        new_spacing = spacing / real_resize_factor

        image = scipy.ndimage.interpolation.zoom(image,
                                                 real_resize_factor,
                                                 mode='nearest')

        return image, new_spacing

    @staticmethod
    def normalize(image):
        # check this table for more information http://i.imgur.com/4rlyReh.png
        MIN_BOUND = 0.
        MAX_BOUND = 300.0
        image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        image[image > 1] = 1.
        image[image < 0] = 0.
        return image

    @staticmethod
    def zero_center(image, pixel_mean=0.1):
        image = image - pixel_mean
        return image

    @staticmethod
    def vcrop(img, vmean, vwidth, from_statistic=False):
        if from_statistic:
            vmean, vstd, _ = StrokeTorchDataset.vstatistic(img)
            vmin = vmean - vstd
            vmax = vmean + vstd
        else:
            vmin, vmax = (vmean - vwidth / 2), (vmean + vwidth / 2)
        img = (img - vmin) / vwidth
        img = np.clip(img, 0., 1.)
        img *= 255.

        img = img.astype(np.uint8)
        return img

    @staticmethod
    def vstatistic(img, vmin=-500, vmax=500):
        _mask = (img > vmin) & (img < vmax)
        _img = np.ma.masked_array(img, mask=~_mask)
        _valid_img_values = _img[~_img.mask].compressed()
        return (_valid_img_values.mean(), _valid_img_values.std(),
                _valid_img_values)


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
        Ap.ToTensorV2()
    ])

    d = StrokeSegmentationDataset(root=Path('./input/stroke'),
                                  test_size=0.2,
                                  transform=transform)

    sample = d.trainset[10]
    print()

    StrokeSegmentationDataset(root=Path('./input/stroke'), test_size=0.25)
