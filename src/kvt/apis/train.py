import copy
import os
import subprocess
from collections import defaultdict
from subprocess import PIPE

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
    build_strong_transform,
)
from kvt.evaluate import evaluate
from omegaconf import OmegaConf, open_dict
from pytorch_lightning.plugins import DDPPlugin
from tqdm import tqdm


def train_last_linear(config, model, hooks, logger):
    print("---------------------------------------------------------------")
    print("Last linear training")

    # build datasets
    dataloaders = build_dataloaders(config)

    # initialize model
    model, params = kvt.utils.initialize_model(config, model, backbone_lr_ratio=0)

    # build optimizer
    optimizer = torch.optim.Adam(params, lr=0.001)

    # build lightning module
    lightning_module = build_lightning_module(
        config,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        hooks=hooks,
        dataloaders=dataloaders,
    )

    # train loop
    trainer_params = copy.deepcopy(config.trainer.trainer)
    # overwrite
    for key, value in config.trainer.model.last_linear.params.items():
        trainer_params[key] = value

    trainer = pl.Trainer(logger=logger, **trainer_params)
    trainer.fit(lightning_module)

    # re-initialize model (set trainable parameters)
    model, _ = kvt.utils.reinitialize_model(config, model)

    return model


def run(config):
    # build hooks
    loss_fn = build_loss(config)
    metric_fn = build_metrics(config)
    hooks = build_hooks(config)
    hooks.update({"loss_fn": loss_fn, "metric_fn": metric_fn})

    # build model
    model = build_model(config)

    # build callbacks
    callbacks = build_callbacks(config)

    # build logger
    logger = build_logger(config)

    # debug
    if config.debug:
        logger = None
        OmegaConf.set_struct(config, True)
        with open_dict(config):
            config.trainer.trainer.max_epochs = None
            config.trainer.trainer.max_steps = 100

    # logging for wandb or mlflow
    if hasattr(logger, "log_hyperparams"):
        for k, v in config.trainer.items():
            if not k in ("metrics", "inference"):
                logger.log_hyperparams(params=v)
        logger.log_hyperparams(params=config.dataset)
        logger.log_hyperparams(params=config.augmentation)

    # last linear training
    if (
        hasattr(config.trainer.model, "last_linear")
        and (config.trainer.model.last_linear.training)
        and (config.trainer.model.params.pretrained)
    ):
        model = train_last_linear(config, model, hooks, logger)

    # initialize model
    model, params = kvt.utils.initialize_model(config, model)

    # build optimizer
    optimizer = build_optimizer(config, model=model, params=params)

    # build scheduler
    scheduler = build_scheduler(config, optimizer=optimizer)

    # build datasets
    dataloaders = build_dataloaders(config)

    # build strong transform
    strong_transform, storong_transform_p = build_strong_transform(config)

    # build lightning module
    lightning_module = build_lightning_module(
        config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        hooks=hooks,
        dataloaders=dataloaders,
        strong_transform=strong_transform,
        storong_transform_p=storong_transform_p,
    )

    # build plugins
    # fix this issue
    # https://github.com/PyTorchLightning/pytorch-lightning/discussions/6219
    plugins = []
    if hasattr(config.trainer.trainer, "accelerator") and (
        config.trainer.trainer.accelerator in ("ddp", "ddp2")
    ):
        plugins.append(
            DDPPlugin(find_unused_parameters=False),
        )

    # best model path
    dir_path = config.trainer.callbacks.ModelCheckpoint.dirpath
    filename = f"fold_{config.dataset.dataset.params.idx_fold}_best.ckpt"
    best_model_path = os.path.join(dir_path, filename)

    # train loop
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        plugins=plugins,
        **config.trainer.trainer,
    )
    if not config.trainer.skip_training:
        trainer.fit(lightning_module)
        path = trainer.checkpoint_callback.best_model_path
        # copy best model
        subprocess.run(
            f"mv {path} {best_model_path}", shell=True, stdout=PIPE, stderr=PIPE
        )

    # log best model
    if hasattr(logger, "log_hyperparams"):
        logger.log_hyperparams(params={"best_model_path": best_model_path})

    # load best checkpoint
    print(f"Loading best model: {best_model_path}")
    state_dict = torch.load(best_model_path)["state_dict"]

    # if using dp, it is necessary to fix state dict keys
    if (
        hasattr(config.trainer.trainer, "sync_batchnorm")
        and config.trainer.trainer.sync_batchnorm
    ):
        state_dict = kvt.utils.fix_dp_model_state_dict(state_dict)

    lightning_module.model.load_state_dict(state_dict)

    # evaluate
    metric_dict = evaluate(lightning_module, hooks, config, mode=["validation"])

    if hasattr(logger, "log_metrics"):
        logger.log_metrics(metric_dict)
