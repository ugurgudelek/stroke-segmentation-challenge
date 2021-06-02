# -*- encoding: utf-8 -*-
# @File    :   stroke-experiment.py
# @Time    :   2021/05/15 16:45:06
# @Author  :   Ugur Gudelek, Gorkem Can Ates
# @Contact :   ugurgudelek@gmail.com
# @Desc    :   None

from pathlib import Path
import torch
from torchvision import transforms
from torch.optim import lr_scheduler, Adam
import albumentations as A
import albumentations.pytorch as Ap

from berries.experiments.experiment import Experiment
from berries.metric import metrics
from berries.logger import MultiLogger

from dataset.stroke import StrokeSegmentationDataset
from model.ResUnetPlus import ResUnetPlus
from model.ResUnet import ResUnet
from model.XNET import XNET
from trainer.demo import DemoTrainer
from metric import seg_metrics as local_metrics
from dataset.stroke import Stroke
from loss.losses import DiceLoss, IoULoss, TrevskyLoss, FocalLoss, EnhancedMixingLoss


class StrokeExperiment(Experiment):
    def __init__(self):
        self.params = {
            'project_name': 'stroke',
            'experiment_name': 'ResUnet-gn-IoU-lr1e-4-bsize-4-pretrained-0-dataaug-1-TL-0',

            # 'project_name': 'debug',
            # 'experiment_name': 'process',
            'seed': 42,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',

            'resume': False,
            'pretrained': False,
            'checkpoint': {
                'on_epoch': 1000,
                'metric': local_metrics.Accuracy.__name__.lower(),
                'trigger': lambda new, old: new > old
            },
            'log': {
                'on_epoch': 1,
            },
            'stdout': {
                'verbose': True,
                'on_batch': 20,
                'on_epoch': 1
            },
            'root': Path('./'),
            'neptune': {
                # 'id': 'STROK-267',
                'workspace': 'machining',
                'project': 'stroke',
                'tags': ['StrokeSeg', 'ResUnet-gn-IoU-lr1e-4-bsize-4-pretrained-0-dataaug-1-TL-0'],
                'source_files': ['./stroke-experiment.py']
            }
        }

        self.hyperparams = {
            'lr': 0.0001,
            'weight_decay': 0.,
            'epoch': 500,
            'batch_size': 4,
            'validation_batch_size': 4,
        }  # yapf: disable

        self.alpha = 0.333
        # Create netcdf4 file for faster reading
        # Stroke(root=Path('./input/stroke')).for_classification()
        # Stroke(root=Path('./input/stroke')).for_segmentation()

        # self.model = XNET(in_channels=3,
        #                   out_channels=3,
        #                   device=self.params['device'],
        #                   k=0.25,
        #                   norm_type='gn',
        #                   upsample_type='bilinear')

        # self.model = ResUnetPlus(in_features=3,
        #                          out_features=3,
        #                          k=0.25,
        #                          norm_type='gn',
        #                          upsample_type='bilinear')

        self.model = ResUnet(in_features=3,
                             out_features=3,
                             k=0.25,
                             norm_type='gn')
        print(self.model)

        self.transform = A.Compose([
            A.ShiftScaleRotate(shift_limit=0.01,
                               scale_limit=0.01,
                               rotate_limit=180,
                               p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # A.ColorJitter(p=0.2),
            # A.GaussianBlur(p=0.2),
            # A.GaussNoise(p=0.2),
            # A.GlassBlur(p=0.2),
            Ap.ToTensorV2()
        ])
        self.val_transform = A.Compose([Ap.ToTensorV2()])
        self.dataset = StrokeSegmentationDataset(Path('../stroke-segmentation-challenge/input/stroke'),
                                                 transform=self.transform,
                                                 val_transform=self.val_transform,
                                                 test_size=0.25)

        self.logger = MultiLogger(
            root=self.params['root'],
            project_name=self.params['project_name'],
            experiment_name=self.params['experiment_name'],
            params=self.params,
            hyperparams=self.hyperparams)

        self.optimizer = Adam(params=self.model.parameters(),
                              lr=self.hyperparams.get('lr', 0.001),
                              weight_decay=self.hyperparams.get(
                                  'weight_decay', 0))

        # Decay LR by a factor of 0.1 every 7 epochs
        self.exp_lr_scheduler = lr_scheduler.StepLR(self.optimizer,
                                                    step_size=7,
                                                    gamma=0.1,
                                                    verbose=False)

        self.trainer = DemoTrainer(
            model=self.model,
            criterion=IoULoss(),
            # criterion=torch.nn.CrossEntropyLoss(),
            #
            # metrics=[local_metrics.Accuracy,
            #          local_metrics.MeanIoU,
            #          local_metrics.IoU_class1,
            #          local_metrics.IoU_class2,
            #          local_metrics.Recall_class1,
            #          local_metrics.Recall_class2,
            #          local_metrics.Precision_class1,
            #          local_metrics.Precision_class2],
            metrics=[local_metrics.Accuracy],
            hyperparams=self.hyperparams,
            params=self.params,
            logger=self.logger
        )

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
