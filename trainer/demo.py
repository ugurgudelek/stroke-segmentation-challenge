# -*- encoding: utf-8 -*-
# @File    :   demo.py
# @Time    :   2021/05/15 06:14:04
# @Author  :   Ugur Gudelek
# @Contact :   ugurgudelek@gmail.com
# @Desc    :   None

from berries.trainer.base import BaseTrainer


class DemoTrainer(BaseTrainer):
    def __init__(self,
                 model,
                 metrics,
                 hyperparams,
                 params,
                 optimizer=None,
                 scheduler=None,
                 criterion=None,
                 logger=None):
        super().__init__(model, metrics, hyperparams, params, optimizer, scheduler,
                         criterion, logger)

    def after_fit_one_epoch(self, history_container, metric_container):
        import matplotlib.pyplot as plt

        samples = self.validation_dataset.get_random_sample(n=3)
        predictions, targets = self.transform(dataset=samples,
                                              classification=True)
        for i in range(len(samples)):
            data = samples[i]['data'].squeeze()
            target = targets[i]
            prediction = predictions[i]

            fig = plt.figure()
            plt.imshow(data)
            plt.axis("off")
            plt.suptitle(
                f'Epoch:{self.epoch}    Target:{target}    Prediction:{prediction}'
            )
            self.logger.log_image(fig)
            plt.close()
