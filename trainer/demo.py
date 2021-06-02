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
        import numpy as np

        samples = self.validation_dataset.get_random_sample(n=3)
        predictions, targets = self.transform(dataset=samples,
                                              classification=True)
        for i in range(len(samples)):
            # data = samples[i]['data'].squeeze()
            target = targets[i]
            prediction = predictions[i]
            target_image = np.zeros((512, 512, 3))
            target_image[:, :, np.max(target)] = target
            target_image[target_image != 0] = 255

            pred_image = np.zeros((512, 512, 3))
            pred_image[prediction == 1, 1] = 255
            pred_image[prediction == 2, 2] = 255
            pred_image.astype(np.uint8)
            fig1 = plt.figure()
            plt.imshow(target_image.astype(np.uint8))
            plt.axis("off")
            plt.suptitle(
                f'Epoch:{self.epoch} Ex:{i+1} Target Mask'
            )
            fig2 = plt.figure()
            plt.imshow(pred_image.astype(np.uint8))
            plt.axis("off")
            plt.suptitle(
                f'Epoch:{self.epoch} Ex:{i+1} Prediction Mask'
            )
            self.logger.log_image(fig1)
            self.logger.log_image(fig2)
            plt.close('all')
