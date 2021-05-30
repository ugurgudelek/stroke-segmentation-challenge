# -*- encoding: utf-8 -*-
# @File    :   stroke-experiment.py
# @Time    :   2021/05/15 16:45:06
# @Author  :   Ugur Gudelek, Gorkem Can Ates
# @Contact :   ugurgudelek@gmail.com
# @Desc    :   None

from pathlib import Path
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF

from berries.experiments.experiment import Experiment
from berries.metric import metrics
from berries.logger import MultiLogger

from dataset.stroke import StrokeClassificationDataset
from model.cnn import VGG16, VGG19, CNN, DenseNet, ResNet, CustomCNN
from trainer.demo import DemoTrainer
from metric import metrics as local_metrics
from dataset.stroke import Stroke
from loss.losses import WeightedBCELoss


class StrokeExperiment(Experiment):
    def __init__(self):
        self.params = {
            'project_name': 'stroke',
            # 'experiment_name': 'ResNet-152-WBCE-lr1e-4-bsize-8-pretrained-0-dataaug-2-TL-0',
            'experiment_name': 'VGG19_bn-WBCE-lr1e-4-bsize-8-pretrained-0-dataaug-2-TL-0',

            # 'project_name': 'debug',
            # 'experiment_name': 'process',
            'seed': 42,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',

            'resume': True,
            'pretrained': False,
            'checkpoint': {
                'on_epoch': 2,
            },
            'log': {
                'on_epoch': 1,
            },
            'stdout': {
                'verbose': True,
                'on_batch': 0,
                'on_epoch': 1
            },
            'root': Path('./'),
            'neptune': {
                'id': 'STROK-184',
                'workspace': 'machining',
                'project': 'stroke',
                'tags': ['StrokeSeg', 'VGG19_bn-WBCE-lr1e-4-bsize-8-pretrained-0-dataaug-2-TL-0'],
                'source_files': ['./stroke-experiment.py']
            }
        }

        self.hyperparams = {
            'lr': 0.0001,
            'weight_decay': 0.,
            'epoch': 100,
            'batch_size': 8,
            'validation_batch_size': 8,
        }  # yapf: disable

        self.alpha = 0.333
        # Create netcdf4 file for faster reading
        # Stroke(root=Path('./input/stroke')).for_classification()
        # Stroke(root=Path('./input/stroke')).for_segmentation()

        self.model = VGG19(pre_trained=False,
                           req_grad=True,
                           bn=True,
                           out_channels=2,
                           input_dim=(3, 512, 512))

        # self.model = ResNet(net_type='ResNet-152',
        #                     pre_trained=False,
        #                     req_grad=True,
        #                     out_channels=2,
        #                     input_dim=(3, 512, 512))

        print(self.model)

        self.dataset = StrokeSegmentationDataset(Path('../stroke-segmentation-challenge/input/stroke'),
                                                 transform=transforms.Compose([transforms.RandomApply(
                                                     [transforms.RandomRotation((-180, 180)),
                                                      transforms.ColorJitter(),
                                                      transforms.RandomHorizontalFlip(p=0.5),
                                                      transforms.RandomVerticalFlip(p=0.5),
                                                      transforms.GaussianBlur(kernel_size=51)], p=0.5)]),
                                                 test_size=0.25)

        self.logger = MultiLogger(
            root=self.params['root'],
            project_name=self.params['project_name'],
            experiment_name=self.params['experiment_name'],
            params=self.params,
            hyperparams=self.hyperparams)

        self.trainer = DemoTrainer(
            model=self.model,
            criterion=torch.nn.CrossEntropyLoss(
                weight=torch.tensor([self.alpha, 1 - self.alpha]).to(self.params['device'])),
            # criterion=WeightedBCELoss([self.alpha, 1-self.alpha]),
            metrics=[local_metrics.Accuracy,
                     local_metrics.MeanMetric,
                     local_metrics.Recall,
                     local_metrics.Specificity],
            # metrics=[metrics.Accuracy],

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
