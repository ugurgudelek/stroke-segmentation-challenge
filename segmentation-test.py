import warnings

warnings.filterwarnings("ignore", module="matplotlib\..*")

from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms
from torch.optim import lr_scheduler, Adam

import albumentations as A
import albumentations.pytorch as Ap

from dataset.stroke import StrokeSegmentationDataset, Stroke, StrokeSegmentationTorch
from model.res_unet_plus import ResUnetPlus

from trainer.segmentation import SegmentationTrainer
from dataset.stroke import Stroke

from metric.segmentation import IoU, DiceScore, FBeta, PostIoU, MaskMeanMetric
import loss.loss as local_loss

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from PIL import Image

PNG_PATH = Path('input/teknofest/contest/input/segmentation/PNG')
PRETRAINED_MODEL_PATH = Path('./checkpoints/segmentation/620/model-optim.pth')
OUTPUT_PATH = Path('input/teknofest/contest/output/segmentation')
HOLD_PROBABILITY = 0.05

params = {
    'resume': False,
    'pretrained': True,
    'pretrained_path': PRETRAINED_MODEL_PATH,
    'device': 'cuda'
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = ResUnetPlus(in_features=3,
                    out_features=3,
                    k=0.5,
                    norm_type='gn',
                    squeeze=True,
                    upsample_type='bilinear')

stroke_dataset = Stroke(root=None)
xr_dataset = Stroke.from_path_segmentation(png_path=PNG_PATH,
                                           infer_mask=True,
                                           hold_prop=HOLD_PROBABILITY)
torch_dataset = StrokeSegmentationTorch(images=xr_dataset.image.values,
                                        masks=xr_dataset.mask.values,
                                        transform=Ap.ToTensorV2())

criterion = local_loss.CombinedLoss(
    main_criterion=local_loss.IoULoss(reduction='none'),
    combined_criterion=local_loss.VGGLoss(
        extractor=local_loss.VGGExtractor(device=device),
        criterion=nn.MSELoss(reduction='none'),
        reduction='none',
        device=device),
    weight=[1, 0.1],
    reduction='none')

trainer = SegmentationTrainer(
    model=model,
    criterion=criterion,
    metrics=[
        IoU(),
        PostIoU(),
        MaskMeanMetric()
        # local_loss.IoULoss(reduction='mean'),
        # local_loss.BoundaryLoss(device='cpu', reduction='mean')
    ],
    # logger=self.logger,
    params=params,
    hyperparams=dict()
    )  # yapf:disable


def im012torgb(im012):
    H, W = im012.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[im012 == 1, 1] = 255
    rgb[im012 == 2, 2] = 255
    return rgb  # HWC


def rgbtoim012_channel_change(rgb):
    C, H, W = rgb.shape
    im012 = np.zeros((H, W), dtype=np.uint8)
    im012[rgb[1] == 255] = 2
    im012[rgb[2] == 255] = 1
    return im012  # HW


def plot(image, prediction, id):
    image = np.moveaxis(image, 0, -1)
    pred_rgb = im012torgb(prediction)

    try:
        mask_path = [
            p for p in Path('input/teknofest/raw').glob(f'**/{id}.png')
            if p.parent.stem == 'MASK'
        ][0]
        masks = Stroke.read_images(image_paths=[mask_path])
        mask = masks[id][0]
        mask_rgb = im012torgb(mask)

    except:
        H, W, C = pred_rgb.shape
        mask_rgb = np.zeros((H, W, 3), dtype=np.uint8)

    fig, axes = plt.subplots(ncols=3, figsize=(10, 5), sharey=True)
    axes[0].imshow(image)
    axes[0].axis("off")
    axes[1].imshow(pred_rgb)
    axes[1].axis("off")
    axes[2].imshow(mask_rgb)
    axes[2].axis("off")

    fig.suptitle(f"{id}")


# Make predictions
images = xr_dataset.image.values  #BCHW
masks = xr_dataset.mask.values  #BCHW
ids = xr_dataset.id.values  # B

OUTPUT_PATH.mkdir(exist_ok=True, parents=True)
(OUTPUT_PATH / '012').mkdir(exist_ok=True, parents=True)
(OUTPUT_PATH / 'rgb').mkdir(exist_ok=True, parents=True)

# score, (predictions, targets) = trainer.score(dataset=torch_dataset,
#                                               classification=False,
#                                               batch_size=4)
# print(score)

stroke_dataset.initialize_undo_cropping(png_path=PNG_PATH)

item_cnt = 0
batch_size = 4
for predictions, targets in trainer.transform_iter(dataset=torch_dataset,
                                                   batch_size=batch_size,
                                                   classification=True):

    # Save predictions
    item_cnt += len(predictions)
    for prediction, mask, id in zip(predictions,
                                    masks[(item_cnt - batch_size):(item_cnt)],
                                    ids[(item_cnt - batch_size):(item_cnt)]):
        # Contest mode prediction image save
        prediction_img = Image.fromarray(prediction.astype(np.uint8),
                                         mode='P')  # HW
        prediction_img = im012torgb(np.array(prediction_img))  # HWC
        prediction_img = np.transpose(prediction_img, (2, 0, 1))  # CHW

        # undo cropping
        prediction_img = stroke_dataset.undo_cropping(
            images=dict(zip([id], [prediction_img])))[id]
        mask = stroke_dataset.undo_cropping(images=dict(zip([id], [mask])))[id]

        prediction_img = rgbtoim012_channel_change(prediction_img)  # HW
        prediction_img = Image.fromarray(prediction_img)
        prediction_img.save(OUTPUT_PATH / f'012/{id}.png')

        # Debug mode prediction image save
        prediction_img_rgb = Image.open(OUTPUT_PATH /
                                        f'012/{id}.png').convert('RGB')
        prediction_img_rgb = np.asarray(prediction_img_rgb)  # HWC
        prediction_img_rgb = np.transpose(prediction_img_rgb, (2, 0, 1))  # CHW
        prediction_img_rgb = im012torgb(prediction_img_rgb[0])
        Image.fromarray(prediction_img_rgb).save(OUTPUT_PATH / f'rgb/{id}.png')

        mask_img_rgb = im012torgb(mask[0])
        Image.fromarray(mask_img_rgb).save(OUTPUT_PATH / f'rgb/{id}_mask.png')
