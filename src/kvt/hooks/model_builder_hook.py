from __future__ import absolute_import, division, print_function

import abc
import copy
import os

import kvt.registry
import torch
import torch.nn as nn
from kvt.models.layers import (
    AdaptiveConcatPool2d,
    BlurPool,
    Flatten,
    GeM,
    NetVLAD,
    SEBlock,
    SoftPool,
)
from kvt.registry import BACKBONES, MODELS
from kvt.utils import build_from_config
from omegaconf import OmegaConf


def analyze_in_features(model):
    if hasattr(model, "classifier"):
        in_features = model.classifier.in_features
    elif hasattr(model, "fc"):
        in_features = model.fc.in_features
    elif hasattr(model, "last_linear"):
        in_features = model.last_linear.in_features
    elif hasattr(model, "head"):
        in_features = model.head.fc.in_features
    else:
        raise ValueError(f"Model has no last linear layer: {model}")

    return in_features


def replace_last_linear(
    model,
    num_classes,
    pool_type,
    dropout_rate,
    use_seblock,
):
    # default args
    if dropout_rate is None:
        dropout_rate = 0.5
    if use_seblock is None:
        use_seblock = False

    # replace pooling
    def replace_pooling_layer(original, layer_name):
        fc_input_shape_ratio = 1
        if pool_type == "concat":
            setattr(original, layer_name, AdaptiveConcatPool2d())
            fc_input_shape_ratio = 2
        elif pool_type == "avg":
            setattr(original, layer_name, nn.AdaptiveAvgPool2d((1, 1)))
        elif pool_type == "adaptive_avg":
            setattr(original, layer_name, nn.AdaptiveAvgPool2d((10, 10)))
            fc_input_shape_ratio = 100
        elif pool_type == "gem":
            setattr(original, layer_name, GeM())

        return fc_input_shape_ratio

    for layer_name in ["avgpool", "global_pool"]:
        if hasattr(model, layer_name):
            fc_input_shape_ratio = replace_pooling_layer(model, layer_name)
        elif hasattr(model, "head") and hasattr(model.head, layer_name):
            fc_input_shape_ratio = replace_pooling_layer(model.head, layer_name)
        else:
            fc_input_shape_ratio = 1

    in_features = analyze_in_features(model)
    in_features *= fc_input_shape_ratio

    # replace fc
    last_layers = [Flatten()]

    if use_seblock:
        last_layers.append(SEBlock(in_features))

    last_layers.extend(
        [
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes),
        ]
    )
    last_layers = nn.Sequential(*last_layers)

    if hasattr(model, "classifier"):
        model.classifier = last_layers
    elif hasattr(model, "fc"):
        in_features = model.fc.in_features * fc_input_shape_ratio
        model.fc = last_layers
    elif hasattr(model, "last_linear"):
        model.last_linear = last_layers
    elif hasattr(model, "head"):
        model.head.fc = last_layers

    return model


class ModelBuilderHookBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, config):
        pass


class DefaultModelBuilderHook(ModelBuilderHookBase):
    def __call__(self, config):
        #######################################################################
        # classification models
        #######################################################################
        if BACKBONES.get(config.name) is not None:
            model = self.build_classification_model(config)

        #######################################################################
        # sound event detection models
        #######################################################################
        if BACKBONES.get(config.name) == "SED":
            backbone = build_from_config(config.backbone, BACKBONES)
            in_features = analyze_in_features(backbone)
            args = {"backbone": backbone, "in_features": in_features}
            model = build_from_config(config, MODELS, args)

        #######################################################################
        # segmentation models
        #######################################################################
        else:
            model = build_from_config(config, MODELS)

        # load pretrained model trained on external data
        print(config)
        if isinstance(config.params.pretrained, str):
            path = config.params.pretrained
            print(f"Loading pretrained trained from: {path}")
            if os.path.exists(path):
                state_dict = torch.load(path)["state_dict"]
            else:
                state_dict = torch.hub.load_state_dict_from_url(path, progress=True)[
                    "state_dict"
                ]

            # fix state_dict
            if hasattr(config, "fix_state_dict"):
                if config.fix_state_dict == "mocov2":
                    state_dict = kvt.utils.fix_mocov2_state_dict(state_dict)
            else:
                # local model trained on dp
                state_dict = kvt.utils.fix_dp_model_state_dict(state_dict)

            model = kvt.utils.load_state_dict_on_same_size(model, state_dict)

        return model

    def build_classification_model(self, config):
        # build model
        if hasattr(config.params, "pretrained") and config.params.pretrained:
            # if pretrained is True and num_classes is not 1000,
            # loading pretraining model fails
            # To avoid this issue, load as default num_classes
            pretrained_config = copy.deepcopy(config)
            pretrained_config = OmegaConf.to_container(pretrained_config)
            del pretrained_config["params"]["num_classes"]
            model = build_from_config(pretrained_config, BACKBONES)
        else:
            model = build_from_config(config, BACKBONES)

        # replace last linear
        if config.last_linear.replace:
            model = replace_last_linear(
                model,
                config.params.num_classes,
                config.last_linear.pool_type,
                config.last_linear.dropout,
                config.last_linear.use_seblock,
            )

        return model
