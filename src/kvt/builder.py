from __future__ import absolute_import, division, print_function

from functools import partial

import hydra
import torch
from easydict import EasyDict as edict
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

import kvt
import kvt.augmentation
from kvt.registry import (
    DATASETS,
    HOOKS,
    LIGHTNING_MODULES,
    LOSSES,
    METRICS,
    OPTIMIZERS,
    SCHEDULERS,
    TRANSFORMS,
)
from kvt.utils import build_from_config


def build_dataloaders(config):
    dataloaders = []
    datasets_config = config.dataset.dataset

    if isinstance(datasets_config, DictConfig):
        datasets_config = OmegaConf.to_container(datasets_config, resolve=True)

    if not isinstance(datasets_config, list):
        datasets_config = [datasets_config]

    for dataset_config in datasets_config:
        dataset_config = edict(dataset_config)
        for split_config in dataset_config.splits:
            # if isinstance(split_config, DictConfig):
            # split_config = OmegaConf.to_container(split_config, resolve=True)

            if isinstance(dataset_config, DictConfig):
                cfg = edict(
                    {
                        "name": dataset_config.name,
                        "params": OmegaConf.to_container(
                            dataset_config.params, resolve=True
                        ),
                    }
                )
            else:
                cfg = edict(
                    {
                        "name": dataset_config.name,
                        "params": dataset_config.params,
                    }
                )

            cfg.params.update(split_config)

            if config.print_config:
                print("---------------------------------------------------------------")
                print(f"dataset config: \n {cfg}")

            split = cfg.params.split
            is_train = cfg.params.mode == "train"
            if is_train:
                batch_size = config.trainer.train.batch_size
            else:
                batch_size = config.trainer.evaluation.batch_size

            # build transform
            transform_configs = {
                "split": split,
                "aug_cfg": config.augmentation.get(split),
            }
            for p in ["height", "width"]:
                if hasattr(config.augmentation, p):
                    transform_configs[p] = getattr(config.augmentation, p)
                else:
                    print(f"{p} is not in augmentation config")
            transform = build_from_config(
                config.dataset.transform,
                TRANSFORMS,
                default_args=transform_configs,
            )

            # build dataset
            dataset = build_from_config(
                cfg,
                DATASETS,
                default_args={"transform": transform, "batch_size": batch_size},
            )

            dataloader = DataLoader(
                dataset,
                shuffle=is_train,
                batch_size=batch_size,
                drop_last=False,
                num_workers=config.dataset.transform.num_preprocessor,
                pin_memory=True,
            )

            dataloaders.append(
                {
                    "split": cfg.params.split,
                    "mode": cfg.params.mode,
                    "dataloader": dataloader,
                }
            )
    return dataloaders


def build_model(config):
    build_model_hook_config = {"name": "DefaultModelBuilderHook"}
    hooks = config.trainer.hooks
    if (hooks is not None) and ("build_model" in hooks):
        build_model_hook_config.update(hooks.build_model)

    build_model_fn = build_from_config(build_model_hook_config, HOOKS)
    return build_model_fn(config.trainer.model)


def build_optimizer(config, model=None, **kwargs):
    # for specific optimizers that needs "base optimizer"
    if config.trainer.optimizer.name == "AGC":
        # optimizer: instance
        base_optimizer = build_from_config(
            config.trainer.optimizer.params.base, OPTIMIZERS, default_args=kwargs
        )
        optimizer = getattr(kvt.optimizers, config.trainer.optimizer.name)(
            model.parameters(),
            base_optimizer,
            model=model,
            **{k: v for k, v in config.trainer.optimizer.params.items() if k != "base"},
        )
    elif config.trainer.optimizer.name == "SAM":
        # optimizer: class
        base_optimizer = OPTIMIZERS.get(config.trainer.optimizer.params.base.name)
        optimizer = getattr(kvt.optimizers, config.trainer.optimizer.name)(
            model.parameters(),
            base_optimizer,
            **{k: v for k, v in config.trainer.optimizer.params.items() if k != "base"},
        )
    else:
        optimizer = build_from_config(
            config.trainer.optimizer, OPTIMIZERS, default_args=kwargs
        )

    return optimizer


def build_scheduler(config, **kwargs):
    if config.trainer.scheduler is None:
        return None

    scheduler = build_from_config(
        config.trainer.scheduler, SCHEDULERS, default_args=kwargs
    )

    return scheduler


def build_loss(config, **kwargs):
    return build_from_config(config.trainer.loss, LOSSES, default_args=kwargs)


def build_lightning_module(config, **kwargs):
    return build_from_config(
        config.trainer.lightning_module, LIGHTNING_MODULES, default_args=kwargs
    )


def build_callbacks(config):
    """pytorch_lightning callbacks"""
    callbacks = []
    if config.trainer.callbacks is not None:
        for callback_name in config.trainer.callbacks:
            cfg = config.trainer.callbacks.get(callback_name)
            callback = hydra.utils.instantiate(cfg)
            callbacks.append(callback)
    return callbacks


def build_logger(config):
    """pytorch_lightning logger"""
    logger = None
    cfg = config.trainer.logger
    if cfg is not None:
        logger = hydra.utils.instantiate(cfg)
    return logger


def build_metrics(config):
    """pytorch_lightning metrics"""
    metrics = {}
    if config.trainer.metrics is not None:
        for name, cfg in config.trainer.metrics.items():
            metrics[name] = build_from_config(cfg, METRICS)
    return edict(metrics)


def build_hooks(config):
    # build default hooks
    post_forward_hook_config = {"name": "DefaultPostForwardHook"}

    if "hooks" in config.trainer:
        hooks = config.trainer.hooks
        if ("post_forward" in hooks) and (hooks.post_forward is not None):
            post_forward_hook_config.update(hooks.post_forward)

    hooks_dict = {}
    hooks_dict["post_forward_fn"] = build_from_config(post_forward_hook_config, HOOKS)
    hooks = edict(hooks_dict)

    return hooks


def build_strong_transform(config):
    strong_transform, p = None, None
    if hasattr(config.augmentation, "strong_transform"):
        strong_cfg = config.augmentation.get("strong_transform")
        p = strong_cfg.p
        if hasattr(kvt.augmentation, strong_cfg.name):
            strong_transform = partial(
                getattr(kvt.augmentation, strong_cfg.name), **strong_cfg.params
            )
        else:
            raise ValueError(f"kvt.augmentation does not contain {strong_cfg.name}")
    return strong_transform, p
