from __future__ import absolute_import, division, print_function

from kvt.utils import Registry

BACKBONES = Registry("backbone")
MODELS = Registry("models")
LOSSES = Registry("losses")
OPTIMIZERS = Registry("optimizer")
SCHEDULERS = Registry("scheduler")

DATASETS = Registry("dataset")
TRANSFORMS = Registry("transform")
HOOKS = Registry("hook")
LIGHTNING_MODULES = Registry("lightning_module")
METRICS = Registry("metrics")
