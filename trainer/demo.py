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