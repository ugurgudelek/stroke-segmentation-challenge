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


class Stroke:
    WIDTH, HEIGHT = (512, 512)
    CLASSES = {'no-stroke': 0, 'stroke': 1}

    def __init__(self, root: Path):
        self.root = root

        # Read no stroke images
        no_stroke_image_paths = list(
            (self.root / 'raw/INMEYOK/PNG').glob('*.png'))
        no_stroke_images = self.read_images(image_paths=no_stroke_image_paths)
        no_stroke_ids = list(map(lambda p: p.stem, no_stroke_image_paths))

        # Read stroke images
        stroke_image_paths = list((self.root / 'raw/KANAMA/PNG').glob('*.png'))
        stroke_image_paths.extend(
            list((self.root / 'raw/ISKEMI/PNG').glob('*.png')))
        stroke_images = self.read_images(image_paths=stroke_image_paths)
        stroke_ids = list(map(lambda p: p.stem, stroke_image_paths))

        no_stroke_data = xr.Dataset(data_vars=dict(
            image=(['id', 'c', 'x', 'y'], no_stroke_images),
            label=(['id'], [Stroke.CLASSES['no-stroke']] * len(no_stroke_images))),
                                    coords=dict(id=('id', no_stroke_ids),
                                                x=('x', range(Stroke.WIDTH)),
                                                y=('y', range(Stroke.HEIGHT)),
                                                c=('c', ['R', 'G', 'B'])))  # yapf: disable

        stroke_data = xr.Dataset(data_vars=dict(
            image=(['id', 'c', 'x', 'y'], stroke_images),
            label=(['id'], [Stroke.CLASSES['stroke']] * len(stroke_images))),
                                 coords=dict(id=('id', stroke_ids),
                                             x=('x', range(Stroke.WIDTH)),
                                             y=('y', range(Stroke.HEIGHT)),
                                             c=('c', ['R', 'G', 'B'])))  # yapf: disable

        self.dataset = xr.concat([no_stroke_data, stroke_data], dim='id')

        save_path = self.root / Path('nc')
        save_path.mkdir(exist_ok=True)
        self.dataset.to_netcdf(path=save_path / 'stroke.nc', engine='netcdf4')

    @staticmethod
    def read_images(image_paths):
        images = []
        for img_path in tqdm(image_paths):
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
            img = np.transpose(img, (2, 0, 1))
            images.append(np.asarray(img))

        return np.asarray(images)


if __name__ == '__main__':

    Stroke(root=Path('./input/stroke'))

