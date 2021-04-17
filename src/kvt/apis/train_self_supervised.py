import copy
import math
import os
from collections import defaultdict

import hydra
import kvt.utils
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from easydict import EasyDict as edict
from kvt.builder import (
    build_callbacks,
    build_dataloaders,
    build_hooks,
    build_lightning_module,
    build_logger,
    build_loss,
    build_metrics,
    build_model,
    build_optimizer,
    build_scheduler,
)
from torch.nn import Conv2d, GroupNorm, Linear
from torch.nn.modules.batchnorm import _BatchNorm
from tqdm import tqdm


class DataModule(pl.LightningDataModule):
    def __init__(self, train_dataloader, val_dataloader):
        super().__init__()
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

    def train_dataloader(self):
        return self.train_dataloader

    def val_dataloader(self):
        return self.val_dataloader


def run(config):
    # build model
    # model = build_model(config)

    # build logger
    # logger = build_logger(config)

    # # logging for wandb or mlflow
    # if hasattr(logger, "log_hyperparams"):
    #     logger.log_hyperparams(params=config.trainer)
    #     logger.log_hyperparams(params=config.dataset)
    #     logger.log_hyperparams(params=config.augmentation)

    # build callbacks
    # callbacks = build_callbacks(config)

    # initialize model
    # model, params = kvt.utils.initialize_model(config, model)

    # build optimizer
    # optimizer = build_optimizer(config, params=params)

    # build scheduler
    # scheduler = build_scheduler(config, optimizer=optimizer)

    # build datasets
    dataloaders = build_dataloaders(config)

    # build lightning module
    # lightning_module = build_lightning_module(config)
    # lightning_module.set_contents(
    #     model, optimizer, scheduler, hooks, dataloaders,
    # )
    from pl_bolts.models.self_supervised import MocoV2

    model = MocoV2(num_negatives=65536)
    assert 65536 % 256 == 0
    model.use_ddp = False
    model.use_ddp2 = False

    print(model.hparams.num_negatives)

    train_dataloader = dataloaders[0]["dataloader"]
    # train_dataloader.drop_last = True
    val_dataloader = dataloaders[1]["dataloader"]
    # val_dataloader.drop_last = True

    datamodule = DataModule(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )
    trainer = pl.Trainer(**config.trainer.trainer)
    trainer.fit(model, datamodule)

    # # train loop
    # trainer = pl.Trainer(logger=logger, callbacks=callbacks, **config.trainer.trainer)
    # trainer.fit(lightning_module)

    # # log best model
    # if hasattr(logger, "log_hyperparams"):
    #     logger.log_hyperparams(
    #         params={"best_model_path": trainer.checkpoint_callback.best_model_path}
    #     )

    # # load best checkpoint
    # state_dict = torch.load(trainer.checkpoint_callback.best_model_path)["state_dict"]

    # # if using dp, it is necessary to fix state dict keys
    # if len(config.trainer.trainer.gpus) >= 2:
    #     state_dict = kvt.utils.fix_dp_model_state_dict(state_dict)

    # lightning_module.model.load_state_dict(state_dict)

    # # evaluate
    # metric_dict = evaluate(lightning_module, hooks, config)

    # if hasattr(logger, "log_metrics"):
    #     logger.log_metrics(metric_dict)
