import xarray as xr
import numpy as np
from PIL import Image
from pathlib import Path
import os
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import albumentations as A
import albumentations.pytorch as Ap
from sklearn.model_selection import train_test_split


class StrokeAugmentation:
    def __init__(self, dataset, save_path, transforms, totaldata):
        self.dataset = dataset
        self.transforms = transforms
        self.data_path = os.path.join(save_path, 'data ccrop resize/')
        self.mask_path = os.path.join(save_path, 'masks ccrop resize/')
        self.totaldata = totaldata

    def transform_data(self, images, masks):
        im_data = np.transpose(images, axes=(1, 2, 0))
        mask_data = masks / 255
        mask_data[2, :, :] *= 2
        mask_data = mask_data[int(np.max(mask_data)), :, :]
        sample = self.transforms(image=im_data, mask=mask_data)

        image = sample['image']
        mask = sample['mask']

        return image, mask

    def get_data(self):
        a = 1
        for k in range(self.totaldata):
            ind = np.random.randint(low=0, high=len(self.dataset.image))
            image, mask = self.transform_data(images=self.dataset.image.values[ind],
                                              masks=self.dataset.mask.values[ind])
            msk = np.zeros((512, 512, 3))
            msk[mask == 1, 1] = 1
            msk[mask == 2, 2] = 1
            msk = np.uint8((msk * 255))
            im = Image.fromarray(image)
            msk = Image.fromarray(msk)
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)
            if not os.path.exists(self.mask_path):
                os.makedirs(self.mask_path)

            if np.sum(image - self.dataset.image.values[ind].transpose(1, 2, 0)) != 0:
                im.save(self.data_path + str(a + 1) + '.png')
                msk.save(self.mask_path + str(a + 1) + '.png')
                a += 1

    def get_ccropresizedata(self):
        for k in range(len(self.dataset.image.values)):
            image, mask = self.transform_data(images=self.dataset.image.values[k],
                                              masks=self.dataset.mask.values[k])
            msk = np.zeros((512, 512, 3))
            msk[mask == 1, 1] = 1
            msk[mask == 2, 2] = 1
            msk = np.uint8((msk * 255))
            im = Image.fromarray(image)
            msk = Image.fromarray(msk)
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)
            if not os.path.exists(self.mask_path):
                os.makedirs(self.mask_path)

            im.save(self.data_path + str(k + 1) + '.png')
            msk.save(self.mask_path + str(k + 1) + '.png')


if __name__ == '__main__':
    root = 'D:/Gorkem Can Ates/PycharmProjects/stroke-segmentation-challenge/input/stroke/'
    dataset = xr.open_dataset(root + 'nc/stroke-segmentation.nc')

    train_ids, test_ids = train_test_split(dataset.id.values,
                                           test_size=0.25,
                                           random_state=42,
                                           shuffle=True)

    train_dataset = dataset.sel({'id': train_ids})
    test_dataset = dataset.sel({'id': test_ids})

    transform = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.01,
                           scale_limit=0.01,
                           rotate_limit=180,
                           p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.OneOf([A.GridDistortion(p=0.5),
                 A.ElasticTransform(p=0.5),
                 A.OpticalDistortion(p=0.5, distort_limit=2, shift_limit=0.5)],
                p=0.2),

        A.GridDropout(p=0.2, random_offset=True, mask_fill_value=None),


        A.OneOf([A.GaussianBlur(p=0.5),
                 A.GlassBlur(p=0.5)], p=0.2),

        A.ColorJitter(p=0.2, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
        A.GaussNoise(p=0.2),

    ])
    # save_path = 'D:/Gorkem Can Ates/PycharmProjects/stroke-segmentation-challenge/input/stroke/Augmented raw/'
    # StrokeAugmentation(dataset=train_dataset,
    #                    transforms=transform,
    #                    save_path=save_path,
    #                    totaldata=20000).get_data()

    # StrokeAugmentation(dataset=train_dataset,
    #                    transforms=A.Sequential([A.CenterCrop(p=1, height=256, width=256),
    #                                             A.Resize(p=1, height=512, width=512)]),
    #                    save_path=save_path,
    #                    totaldata=1000).get_ccropresizedata()

