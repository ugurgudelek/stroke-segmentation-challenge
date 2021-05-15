# -*- encoding: utf-8 -*-
# @File    :   stroke-experiment.py
# @Time    :   2021/05/15 16:45:06
# @Author  :   Ugur Gudelek, Gorkem Can Ates
# @Contact :   ugurgudelek@gmail.com
# @Desc    :   None

from pathlib import Path
import torch
from torchvision import transforms
from berries.experiments.experiment import Experiment
from berries.metric import metrics

from dataset.mnist import MNIST
from model.cnn import CNN
from trainer.demo import DemoTrainer


class StrokeExperiment(Experiment):
    def __init__(self):
        self.params = {
            'project_name': 'debug',
            'experiment_name': 'stroke',
            'seed': 42,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',

            'resume': False,
            'pretrained': False,
            'log_interval': 1,
            'stdout_interval': 256,
            'save_checkpoint': True,
            'root': Path('.'),
            'verbose': 1
        }  # yapf: disable

        self.hyperparams = {
            'lr': 0.001,
            'weight_decay': 0.,
            'epoch': 11,
            'batch_size': 256,
            'validation_batch_size': 256,
        }  # yapf: disable

        self.model = CNN(in_channels=1, out_channels=10, input_dim=(1, 28, 28))
        print(self.model)

        self.dataset = MNIST(root='./input/',
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ]))

        self.trainer = DemoTrainer(
            model=self.model,
            criterion=torch.nn.CrossEntropyLoss(),
            metrics=[metrics.Accuracy],
            hyperparams=self.hyperparams,
            params=self.params,
        )

    def run(self):
        self.trainer.fit(dataset=self.dataset.trainset,
                         validation_dataset=self.dataset.testset)

        self.trainer.to_prediction_dataframe(dataset=self.dataset.testset,
                                             classification=True)


if __name__ == '__main__':
    experiment = StrokeExperiment()
    experiment.run()
