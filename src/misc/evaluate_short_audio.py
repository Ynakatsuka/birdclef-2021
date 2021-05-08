import glob
import os
import sys

import hydra
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset

sys.path.append("src/")
import custom  # import all custom modules for registering objects.
import kvt
import kvt.augmentation
import kvt.utils
from kvt.builder import (
    build_dataloaders,
    build_hooks,
    build_lightning_module,
    build_logger,
    build_model,
)

# from kvt.evaluate import evaluate
from kvt.initialization import initialize
from kvt.registry import TRANSFORMS
from kvt.utils import build_from_config

from .tools import evaluate


def run(config):
    # overwrite path
    OmegaConf.set_struct(config, True)
    with open_dict(config):
        config.trainer.model.params.backbone.params.pretrained = False

    # build logger
    logger = build_logger(config)

    # logging for wandb or mlflow
    if hasattr(logger, "log_hyperparams"):
        for k, v in config.trainer.items():
            if not k in ("metrics", "inference"):
                logger.log_hyperparams(params=v)
        logger.log_hyperparams(params=config.dataset)
        logger.log_hyperparams(params=config.augmentation)

    # build dataloaders
    dataloaders = build_dataloaders(config)

    # build model
    model = build_model(config)

    # build hooks
    hooks = build_hooks(config)

    # build lightning module
    lightning_module = build_lightning_module(
        config,
        model=model,
        optimizer=None,
        scheduler=None,
        hooks=hooks,
        dataloaders=dataloaders,
        strong_transform=None,
    )

    # load best checkpoint
    dir_path = config.trainer.callbacks.ModelCheckpoint.dirpath
    filename = f"fold_{config.dataset.dataset.params.idx_fold}_best.ckpt"
    best_model_path = os.path.join(dir_path, filename)

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
    print("Result:")
    print(metric_dict)


@hydra.main(config_path="../../config", config_name="default")
def main(config: DictConfig) -> None:
    run(config)


if __name__ == "__main__":
    initialize()
    main()
