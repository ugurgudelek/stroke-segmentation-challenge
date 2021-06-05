# -*- encoding: utf-8 -*-
# @File    :   demo.py
# @Time    :   2021/05/15 06:14:04
# @Author  :   Ugur Gudelek
# @Contact :   ugurgudelek@gmail.com
# @Desc    :   None

from berries.trainer.base import BaseTrainer
import torch
import functools


def hook(before=None, after=None):
    def wrap(f):

        @functools.wraps(f)
        def wrapped_f(self, *args, **kwargs):

            if before:
                self.__getattribute__(before)(*args, **kwargs)

            returned_value = f(self, *args, **kwargs)

            if after:
                if returned_value is None:
                    self.__getattribute__(after)()
                elif isinstance(returned_value, tuple):
                    self.__getattribute__(after)(*returned_value)

            return returned_value

        return wrapped_f

    return wrap


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

    # @hook(before='before_fit_one_batch', after='after_fit_one_batch')
    # def _fit_one_batch(self, batch, train):
    #     """All training steps are implemented here.
    #     This function is the core of handling model - actual training loop.
    #
    #     Args:
    #         batch (dict): [description]
    #         train (bool): [description]
    #
    #     Returns:
    #         loss    (torch.Tensor): [description]
    #         output  (torch.Tensor): [description]
    #         data    (torch.Tensor): [description]
    #         target  (torch.Tensor): [description]
    #     """
    #
    #     self._set_grad_enabled(train)
    #
    #     data, target = self.handle_batch(batch)
    #
    #     # do not let pytorch accumulates gradient
    #     self.optimizer.zero_grad()
    #
    #     # track history if only in train
    #     with torch.set_grad_enabled(train):
    #         if self.criterion.K > 1:
    #             output = torch.zeros_like(data).to('cuda')
    #             main_losses = combined_losses = 0
    #             for i in range(self.criterion.K):
    #                 output = self.forward(torch.cat((data, output), dim=1))
    #                 init_main_loss, init_combined_loss = self.criterion(output, target)
    #
    #                 main_losses += (i + 1) * init_main_loss
    #                 combined_losses += (i + 1) * init_combined_loss
    #
    #             coeff = 0.5 * self.criterion.K * (self.criterion.K + 1)
    #
    #             main_loss = main_losses / coeff
    #             combined_loss = combined_losses / coeff
    #
    #             loss = self.criterion.balance[0] * main_loss + self.criterion.balance[1] * combined_loss
    #         else:
    #             output = self.forward(data)
    #             loss = self.compute_loss(output, target)
    #
    #         if train:
    #
    #             # calculate gradient with backpropagation
    #             if self.criterion.reduction == 'none':
    #                 loss.sum().backward()
    #             else:
    #                 loss.backward()
    #
    #             # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
    #             if self.hyperparams.get('clip', None):  # if clip is given
    #                 torch.nn.utils.clip_grad_norm_(
    #                     self.model.parameters(),
    #                     max_norm=self.hyperparams['clip'])
    #
    #             # distribute gradients to update weights
    #             self.optimizer.step()
    #
    #     return loss, output, data, target

    def after_fit_one_epoch(self, history_container, metric_container):
        import matplotlib.pyplot as plt
        import numpy as np

        samples = self.validation_dataset.get_random_sample(n=4)
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
                f'Epoch:{self.epoch} Ex:{i + 1} Target Mask'
            )
            fig2 = plt.figure()
            plt.imshow(pred_image.astype(np.uint8))
            plt.axis("off")
            plt.suptitle(
                f'Epoch:{self.epoch} Ex:{i + 1} Prediction Mask'
            )
            self.logger.log_image(fig1)
            self.logger.log_image(fig2)
            plt.close('all')
