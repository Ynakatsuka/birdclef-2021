from __future__ import absolute_import, division, print_function

import functools
import inspect

import kvt
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf


class Registry:
    def __init__(self, name):
        self._name = name
        self._obj_dict = dict()

    def __repr__(self):
        format_str = self.__class__.__name__
        format_str += f"(name={self._name}, items={list(self._obj_dict.keys())})"
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def obj_dict(self):
        return self._obj_dict

    def get(self, key: str):
        return self._obj_dict.get(key, None)

    def register(self, obj):
        """Register a callable object.

        Args:
            obj: callable object to be registered
        """
        if not callable(obj):
            raise ValueError(f"object must be callable")

        obj_name = obj.__name__
        if obj_name in self._obj_dict:
            pass
            # print(f"{obj_name} is already registered in {self.name}")
            # raise KeyError(f'{obj_name} is already registered in {self.name}')

        self._obj_dict[obj_name] = obj
        return obj


def build_from_config(config, registry, default_args=None):
    """Build a callable object from configuation dict.

    Args:
        config (dict or DictConfig): Configuration dict. It should contain the key "name".
        registry (:obj:`Registry`): The registry to search the name from.
        default_args (dict, optional): Default initialization argments.
    """
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)
    if isinstance(default_args, DictConfig):
        default_args = OmegaConf.to_container(default_args, resolve=True)

    assert isinstance(config, dict) and "name" in config
    assert isinstance(default_args, dict) or default_args is None

    name = config["name"]
    name = name.replace("-", "_")
    obj = registry.get(name)
    if obj is None:
        raise KeyError(f"{name} is not in the {registry.name} registry")

    print(f"Loaded object file: {inspect.getfile(obj)}")

    args = dict()
    if default_args is not None:
        args.update(default_args)
    if "params" in config:
        args.update(config["params"])

    if name in kvt.registry.METRICS._obj_dict.keys():
        o = functools.partial(obj, **args)
    else:
        o = obj(**args)

    return o
