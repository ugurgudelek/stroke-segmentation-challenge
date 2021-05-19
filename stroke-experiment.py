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

from dataset.stroke_classification import StrokeClassificationDataset
from model.cnn import VGG16, CNN, DenseNet, ResNet
from trainer.demo import DemoTrainer
from metric import metrics as local_metrics
from dataset.stroke import Stroke
from loss.losses import WeightedBCELoss


class StrokeExperiment(Experiment):
    def __init__(self):
        self.params = {
            # 'project_name': 'stroke',
            # 'experiment_name': 'ResNet-161-WBCE-lr1e-4-bsize-8-pretrained-0-dataaug-1-TL-0',
            'project_name': 'debug',
            'experiment_name': 'stroke',
            'seed': 42,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',

            'resume': False,
            'pretrained': False,
            'log_interval': 1,
            'stdout_interval': 1,
            'save_checkpoint': True,
            'root': Path('.'),
            'verbose': 1,
            'neptune_project_name': 'machining/stroke',
        }  # yapf: disable

        self.hyperparams = {
            'lr': 0.0001,
            'weight_decay': 0.,
            'epoch': 100,
            'batch_size': 64,
            'validation_batch_size': 64,
        }  # yapf: disable

        self.alpha = 0.333
        # Create netcdf4 file for faster reading
        # Stroke(root=Path('./input/stroke'))

        self.model = CNN(in_channels=3, out_channels=2, input_dim=(3, 512, 512))
        # self.model = VGG16(pre_trained=False, req_grad=True, bn=False, out_channels=1, input_dim=(3, 512, 512))
        # self.model = ResNet(net_type='ResNet-152', pre_trained=False, req_grad=True, out_channels=1,
        #                     input_dim=(3, 512, 512))
        print(self.model)

        self.dataset = StrokeClassificationDataset(Path('../stroke-segmentation-challenge/input/stroke'),
                                                   transform=transforms.Compose([transforms.RandomRotation((-180, 180)),
                                                                                 transforms.RandomHorizontalFlip(p=0.5),
                                                                                 transforms.RandomVerticalFlip(p=0.5)]),
                                                   test_size=0.25)
        self.logger = MultiLogger(
            root=self.params['root'],
            project_name=self.params['project_name'],
            experiment_name=self.params['experiment_name'],
            params=self.params,
            hyperparams=self.hyperparams)

        self.trainer = DemoTrainer(
            model=self.model,
            criterion=torch.nn.CrossEntropyLoss(weight=torch.tensor([self.alpha, 1 - self.alpha]).to('cuda')),
            # criterion=WeightedBCELoss([self.alpha, 1-self.alpha]),
            # metrics=[metrics.Accuracy,
            #          local_metrics.MeanMetric,
            #          local_metrics.Recall,
            #          local_metrics.FPR],
            metrics=[metrics.Accuracy],

            hyperparams=self.hyperparams,
            params=self.params,
            logger=self.logger
        )

    def run(self):
        self.trainer.fit(dataset=self.dataset.trainset,
                         validation_dataset=self.dataset.testset)

        self.trainer.to_prediction_dataframe(dataset=self.dataset.testset,
                                             classification=True)


if __name__ == '__main__':
    experiment = StrokeExperiment()
    experiment.run()


