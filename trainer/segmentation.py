# -*- encoding: utf-8 -*-
# @File    :   demo.py
# @Time    :   2021/05/15 06:14:04
# @Author  :   Ugur Gudelek
# @Contact :   ugurgudelek@gmail.com
# @Desc    :   None

from berries.trainer.base import BaseTrainer, hook
import torch
import functools


class SegmentationTrainer(BaseTrainer):
    def __init__(self,
                 model,
                 metrics,
                 hyperparams,
                 params,
                 optimizer=None,
                 scheduler=None,
                 criterion=None,
                 logger=None):
        super().__init__(model, metrics, hyperparams, params, optimizer,
                         scheduler, criterion, logger)

    @hook(before='before_fit_one_batch', after='after_fit_one_batch')
    def _fit_one_batch(self, batch, train):
        """All training steps are implemented here.
        This function is the core of handling model - actual training loop.
        """

        self._set_grad_enabled(train)

        data, target = self.handle_batch(batch)

        # do not let pytorch accumulates gradient
        self.optimizer.zero_grad()

        # track history if only in train
        with torch.set_grad_enabled(train):

            recursive = 'recursive' in self.hyperparams
            if not recursive:

                output = self.forward(data)
                if self.criterion._get_name() == 'CombinedLoss':
                    if self.criterion.adopt_weight:

                        self.criterion.epoch = self.epoch

                loss = self.compute_loss(output, target)

                if train:

                    # calculate gradient with backpropagation
                    if self.criterion.reduction == 'none':
                        loss.mean().backward()
                    else:
                        loss.backward()

            else:  # recursive
                K = self.hyperparams['recursive']['K']
                coeff = 0.5 * K * (K + 1)

                loss = 0.
                output = torch.zeros_like(data, device=self.device)
                for i in range(K):
                    output = self.forward(
                        torch.cat((data.detach(), output.detach()), dim=1))
                    if self.criterion._get_name() == 'CombinedLoss':
                        if self.criterion.adopt_weight:
                            self.criterion.epoch = self.epoch

                    _loss = self.compute_loss(output, target.detach())
                    _loss *= ((i + 1) / coeff)

                    if train:
                        if self.criterion.reduction == 'none':
                            _loss.mean().backward()
                        else:
                            _loss.backward()

                    _loss = _loss.detach()
                    loss += _loss

            if train:

                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                if self.hyperparams.get('clip', None):  # if clip is given
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.hyperparams['clip'])

                # distribute gradients to update weights
                self.optimizer.step()

        return loss, output, data, target

    def after_fit_one_epoch(self, history_container, metric_container):
        import matplotlib.pyplot as plt
        import numpy as np

        samples = self.validation_dataset.get_random_sample(n=10)
        predictions, targets = self.transform(dataset=samples,
                                              classification=True)
        for i, sample in enumerate(samples):
            image = sample['data'].permute((1, 2, 0))  # CHW
            target = sample['target']  # HW
            pred = predictions[i]

            H, W = target.shape

            target_rgb = np.zeros((H, W, 3), dtype=np.uint8)
            target_rgb[target == 1, 1] = 255
            target_rgb[target == 2, 2] = 255

            pred_rgb = np.zeros((H, W, 3), dtype=np.uint8)
            pred_rgb[pred == 1, 1] = 255
            pred_rgb[pred == 2, 2] = 255

            fig, axes = plt.subplots(ncols=3)

            # axes[0].imshow(image, alpha=0.3)
            axes[0].imshow(image)
            axes[0].axis("off")
            axes[1].imshow(target_rgb)
            axes[1].axis("off")
            axes[2].imshow(pred_rgb)
            axes[2].axis("off")

            fig.suptitle(f'Epoch:{self.epoch} Ex:{i + 1}')
            self.logger.log_image(fig)
            plt.close('all')
