import os

import kvt.utils
import torch
from kvt.builder import (
    build_dataloaders,
    build_hooks,
    build_lightning_module,
    build_model,
)
from kvt.evaluate import evaluate


def run(config):
    # build hooks
    hooks = build_hooks(config)

    # build model
    model = build_model(config)

    # build datasets
    dataloaders = build_dataloaders(config)

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
    filename = f"{config.experiment_name}_fold_{config.dataset.dataset.params.idx_fold}_best.ckpt"
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
    evaluate(lightning_module, hooks, config, mode="test")
