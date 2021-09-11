# -*- encoding: utf-8 -*-
# @File    :   atlas.py
# @Time    :   2021/06/03 01:07:11
# @Author  :   Ugur Gudelek
# @Contact :   ugurgudelek@gmail.com
# @Desc    :   None

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from skimage.transform import resize
import nibabel as nib
from pathlib import Path


class ATLASdataset(Dataset):
    def __init__(self, root=Path('./input/atlas'), augmentation=True):

        self.root = root

        list_path = list(self.root.glob('*/*/*/*t1w_deface_stx.nii.gz'))
        list_path.sort()

        self.augmentation = augmentation
        self.imglist = list_path

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        # path = os.path.join(self.imglist[index], 't01')
        # os.path.join(path, 'T1w_p.nii')
        path = self.imglist[index]
        tempimg = nib.load(path)
        B = np.flip(tempimg.get_data(), 1)
        sp_size = 64
        img = resize(B, (sp_size, sp_size, sp_size), mode='constant')
        img = 1.0 * img
        img = (img - np.min(img)) / (np.max(img) - np.min(img))

        if self.augmentation:
            random_n = torch.rand(1)
            if random_n[0] > 0.5:
                img = np.flip(img, 0)

        img = np.ascontiguousarray(img, dtype=np.float32)

        imageout = torch.from_numpy(img).float().view(1, sp_size, sp_size,
                                                      sp_size)
        imageout = 2 * imageout - 1

        return imageout