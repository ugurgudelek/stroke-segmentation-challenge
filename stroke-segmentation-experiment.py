# -*- encoding: utf-8 -*-
# @File    :   stroke-experiment.py
# @Time    :   2021/05/15 16:45:06
# @Author  :   Ugur Gudelek, Gorkem Can Ates
# @Contact :   ugurgudelek@gmail.com
# @Desc    :   None

from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms
from torch.optim import lr_scheduler, Adam

import albumentations as A
import albumentations.pytorch as Ap

from berries.experiments.experiment import Experiment
from berries.logger import MultiLogger
from berries import loss as berries_loss

from dataset.stroke import StrokeSegmentationDataset
from model.res_unet_plus import ResUnetPlus
from model.res_unet import ResUnet
from model.xnet import XNET

from trainer.segmentation import SegmentationTrainer
from dataset.stroke import Stroke

from metric.segmentation import IoU, DiceScore, FBeta
import loss.loss as local_loss

import torchmetrics


class StrokeExperiment(Experiment):
    def __init__(self):
        self.params = {
            'project_name': 'stroke',
            'experiment_name':
            'Rec1-ResUnetPlus-gn-k05-CombinedIoU-BD-balanced-weight-lr1e-4-bsize-4',

            # 'project_name': 'debug',
            # 'experiment_name': 'stroke',
            'seed': 42,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'resume': False,
            'pretrained': False,
            'checkpoint': {
                'on_epoch': 1000,
                'metric': IoU.__name__.lower(),
                'trigger': lambda new, old: new > old
            },
            'log': {
                'on_epoch': 1,
            },
            'stdout': {
                'verbose': True,
                'on_epoch': 1,
                'on_batch': 1000
            },
            'root': Path('./'),
            'neptune': {
                # 'id': 'STROK-389',
                'workspace': 'machining',
                'project': 'stroke',
                'tags': [
                    'StrokeSeg',
                    'Recursive:1',
                    'ResUnetPlus(gn, k=0.5)',
                    'CombinedVgg', 'IoULoss', 'BoundaryLoss-balanced-weight'
                                              'lr:1e-4',
                    'bsize:4'
                ],

                # 'tags': [' '
                #
                #          ],
                'source_files': ['./stroke-segmentation-experiment.py']
            }
        }  # yapf: disable

        self.hyperparams = {
            'lr': 0.0001,
            'weight_decay': 0.,
            'epoch': 500,
            'batch_size': 4,
            'validation_batch_size': 4,
            # 'recursive': {
            #     'K': 3
            # }
        }  # yapf: disable

        # Create netcdf4 file for faster reading
        # Stroke(root=Path('./input/stroke')).for_classification()
        # Stroke(root=Path('./input/stroke')).for_segmentation()

        # self.model = XNET(in_channels=3,
        #                   out_channels=3,
        #                   device=self.params['device'],
        #                   k=0.25,
        #                   norm_type='gn',
        #                   upsample_type='bilinear')

        self.model = ResUnetPlus(in_features=3 *
                                             (2 if 'recursive' in self.hyperparams else 1),
                                 out_features=3,
                                 k=0.5,
                                 norm_type='gn',
                                 upsample_type='bilinear')

        print(self.model)

        self.transform = A.Compose([
            # A.Resize(256, 256),
            # A.CenterCrop(224, 224),
            A.ShiftScaleRotate(shift_limit=0.01,
                               scale_limit=0.01,
                               rotate_limit=180,
                               p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.OneOf([
                A.GridDistortion(p=0.5),
                A.ElasticTransform(p=0.5),
                A.OpticalDistortion(p=0.5, distort_limit=2, shift_limit=0.5)
            ],
                p=0.2),
            A.GridDropout(p=0.2, random_offset=True, mask_fill_value=None),
            A.OneOf([A.GaussianBlur(p=0.5),
                     A.GlassBlur(p=0.5)], p=0.2),
            A.ColorJitter(p=0.2,
                          brightness=0.4,
                          contrast=0.4,
                          saturation=0.4,
                          hue=0.4),
            A.GaussNoise(p=0.2),
            Ap.ToTensorV2()
        ])

        self.val_transform = A.Compose([
            # A.Resize(256, 256),
            # A.CenterCrop(224, 224),
            Ap.ToTensorV2()
        ])

        self.dataset = StrokeSegmentationDataset(
            Path('./input/teknofest'),
            transform=self.transform,
            val_transform=self.val_transform,
            test_size=0.25,
            debug=False)

        self.logger = MultiLogger(
            root=self.params['root'],
            project_name=self.params['project_name'],
            experiment_name=self.params['experiment_name'],
            params=self.params,
            hyperparams=self.hyperparams)

        # self.criterion = local_loss.CombinedLoss(
        #     main_criterion=local_loss.IoULoss(reduction='none'),
        #     combined_criterion=local_loss.VGGLoss(
        #         extractor=local_loss.VGGExtractor(
        #             device=self.params['device']),
        #         criterion=nn.MSELoss(reduction='none'),
        #         reduction='none',
        #         device=self.params['device']),
        #     weight=[1, 0.1],
        #     reduction='none')
        self.criterion = local_loss.CombinedLoss(main_criterion=local_loss.IoULoss(reduction='none'),
                                                 combined_criterion=local_loss.BoundaryLoss(reduction='none',
                                                                                            device=self.params[
                                                                                                'device']),
                                                 weight=[1, 0.01],
                                                 balance=True,
                                                 adopt_weight=True,
                                                 reduction='none'
                                                 )

        self.optimizer = Adam(params=self.model.parameters(),
                              lr=self.hyperparams['lr'],
                              weight_decay=self.hyperparams['weight_decay'])

        # Decay LR by a factor of 0.1 every 7 epochs
        # self.exp_lr_scheduler = lr_scheduler.StepLR(self.optimizer,
        #                                             step_size=7,
        #                                             gamma=0.1,
        #                                             verbose=False)

        self.trainer = SegmentationTrainer(
            model=self.model,
            criterion=self.criterion,
            metrics=[
                IoU(),
                DiceScore(),
                local_loss.IoULoss(reduction='mean'),
                local_loss.BoundaryLoss(device='cpu', reduction='mean')
            ],
            hyperparams=self.hyperparams,
            params=self.params,
            logger=self.logger)  # yapf:disable

    def run(self):
        self.trainer.fit(dataset=self.dataset.trainset,
                         validation_dataset=self.dataset.testset)

        # Log final model
        self.logger.log_model(path=self.trainer._get_last_checkpoint_path())

        # Log prediction dataframe
        prediction_dataframe = self.trainer.to_prediction_dataframe(dataset=self.dataset.testset,
                                                                    classification=True,
                                                                    save=True)  # yapf:disable

        self.logger.log_dataframe(key='prediction/validation',
                                  dataframe=prediction_dataframe)

        # Log image
        # Example can be found at trainer.py


if __name__ == '__main__':
    experiment = StrokeExperiment()
    experiment.run()
