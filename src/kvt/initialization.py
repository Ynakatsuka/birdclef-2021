from __future__ import absolute_import, division, print_function

import pkgutil

import pretrainedmodels
import pytorch_lightning as pl
import resnest.torch
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models

import kvt.hooks
import kvt.lightning_modules
import kvt.losses
import kvt.models.backbones
import kvt.models.segmentations
import kvt.models.sound_event_detections
import kvt.optimizers
from kvt.registry import (
    BACKBONES,
    DATASETS,
    HOOKS,
    LIGHTNING_MODULES,
    LOSSES,
    METRICS,
    MODELS,
    OPTIMIZERS,
    SCHEDULERS,
    TRANSFORMS,
)

try:
    import torch_optimizer
except ImportError:
    torch_optimizer = None


def register_torch_modules():
    # register backbones
    for name, cls in kvt.models.backbones.__dict__.items():
        if not callable(cls):
            continue
        BACKBONES.register(cls)

    # pretrained models
    for name, cls in pretrainedmodels.__dict__.items():
        if not callable(cls):
            continue
        BACKBONES.register(cls)

    # timm
    for name in timm.list_models():
        cls = timm.model_entrypoint(name)
        if not callable(cls):
            continue
        BACKBONES.register(cls)

    # torchvision models
    for name, cls in torchvision.models.__dict__.items():
        if not callable(cls):
            continue
        BACKBONES.register(cls)

    # resnest
    for name, cls in resnest.torch.__dict__.items():
        if not callable(cls):
            continue
        BACKBONES.register(cls)

    # segmentation models
    for name, cls in kvt.models.segmentations.__dict__.items():
        if not callable(cls):
            continue
        MODELS.register(cls)

    # sound event detection models
    for name, cls in kvt.models.sound_event_detections.__dict__.items():
        if not callable(cls):
            continue
        MODELS.register(cls)

    # register losses
    losses = [
        nn.L1Loss,
        nn.MSELoss,
        nn.CrossEntropyLoss,
        nn.CTCLoss,
        nn.NLLLoss,
        nn.PoissonNLLLoss,
        nn.KLDivLoss,
        nn.BCELoss,
        nn.BCEWithLogitsLoss,
        nn.MarginRankingLoss,
        nn.HingeEmbeddingLoss,
        nn.MultiLabelMarginLoss,
        nn.SmoothL1Loss,
        nn.SoftMarginLoss,
        nn.MultiLabelSoftMarginLoss,
        nn.CosineEmbeddingLoss,
        nn.MultiMarginLoss,
        nn.TripletMarginLoss,
        kvt.losses.DiceLoss,
        kvt.losses.FocalLoss,
        kvt.losses.BinaryFocalLoss,
        kvt.losses.LovaszSoftmaxLoss,
        kvt.losses.LovaszHingeLoss,
        kvt.losses.OUSMLoss,
        kvt.losses.IterativeSelfLearningLoss,
        kvt.losses.JointOptimizationLoss,
        kvt.losses.LabelSmoothingCrossEntropy,
        kvt.losses.OUSMLoss,
        kvt.losses.SymmetricCrossEntropy,
        kvt.losses.SymmetricBCELoss,
        kvt.losses.SymmetricBinaryFocalLoss,
    ]

    for loss in losses:
        LOSSES.register(loss)

    # register optimizers
    optimizers = [
        optim.Adadelta,
        optim.Adagrad,
        optim.Adam,
        optim.AdamW,
        optim.SparseAdam,
        optim.Adamax,
        optim.ASGD,
        optim.LBFGS,
        optim.RMSprop,
        optim.Rprop,
        optim.SGD,
        kvt.optimizers.AGC,
        kvt.optimizers.SGD_AGC,
        kvt.optimizers.SAM,
        kvt.optimizers.SM3,
        kvt.optimizers.RAdam,
    ]
    for optimizer in optimizers:
        OPTIMIZERS.register(optimizer)

    if torch_optimizer is not None:
        for name, cls in torch_optimizer.__dict__.items():
            if not callable(cls):
                continue
            if hasattr(cls, "__name__"):
                OPTIMIZERS.register(cls)

    # register schedulers
    schedulers = [
        optim.lr_scheduler.StepLR,
        optim.lr_scheduler.MultiStepLR,
        optim.lr_scheduler.ExponentialLR,
        optim.lr_scheduler.CosineAnnealingLR,
        optim.lr_scheduler.ReduceLROnPlateau,
        optim.lr_scheduler.CyclicLR,
        optim.lr_scheduler.OneCycleLR,
    ]
    for scheduler in schedulers:
        SCHEDULERS.register(scheduler)

    # register lightning module
    lightning_modules = [
        kvt.lightning_modules.LightningModuleBase,
        kvt.lightning_modules.LightningModuleSAM,
        kvt.lightning_modules.LightningModuleSpecMixUp,
    ]
    for lightning_module in lightning_modules:
        LIGHTNING_MODULES.register(lightning_module)

    # register metrics (functions)
    for name, cls in pl.metrics.functional.__dict__.items():
        if not callable(cls):
            continue
        cls.__name__ = name
        METRICS.register(cls)

    for name, cls in pl.metrics.functional.classification.__dict__.items():
        if not callable(cls):
            continue
        cls.__name__ = name
        METRICS.register(cls)


def register_default_hooks():
    HOOKS.register(kvt.hooks.DefaultModelBuilderHook)
    HOOKS.register(kvt.hooks.DefaultPostForwardHook)
    HOOKS.register(kvt.hooks.SigmoidPostForwardHook)


def initialize():
    register_torch_modules()
    register_default_hooks()
