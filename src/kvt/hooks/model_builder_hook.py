from __future__ import absolute_import, division, print_function

import abc
import copy
import os

import kvt.registry
import torch
import torch.nn as nn
from kvt.models.layers import AdaptiveConcatPool2d, Flatten, GeM, SEBlock
from kvt.registry import BACKBONES, MODELS
from kvt.utils import build_from_config
from omegaconf import OmegaConf


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def analyze_in_features(model):
    if hasattr(model, "classifier"):
        in_features = model.classifier.in_features
    elif hasattr(model, "fc"):
        in_features = model.fc.in_features
    elif hasattr(model, "last_linear"):
        in_features = model.last_linear.in_features
    elif hasattr(model, "head"):
        if hasattr(model.head, "fc"):
            in_features = model.head.fc.in_features
        else:
            in_features = model.head.in_features
    else:
        raise ValueError(f"Model has no last linear layer: {model}")

    return in_features


def replace_last_linear(
    model,
    num_classes,
    pool_type="gem",
    dropout_rate=0,
    use_seblock=False,
    use_identity_as_last_layer=False,
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
        elif pool_type == "identity":
            setattr(original, layer_name, Identity())

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

    if use_identity_as_last_layer:
        last_layers = Identity()

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


def update_input_layer(model, in_channels):
    for l in model.children():
        if isinstance(l, nn.Sequential):
            for ll in l.children():
                assert ll.bias is None
                data = torch.mean(ll.weight, axis=1).unsqueeze(1)
                data = data.repeat((1, in_channels, 1, 1))
                ll.weight.data = data
                break
        else:
            assert l.bias is None
            data = torch.mean(l.weight, axis=1).unsqueeze(1)
            data = data.repeat((1, in_channels, 1, 1))
            l.weight.data = data
        break
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
        if "SED" in config.name:
            model = self.build_sound_event_detection_model(config)

        #######################################################################
        # segmentation models
        #######################################################################
        else:
            model = build_from_config(config, MODELS)

        # load pretrained model trained on external data
        if hasattr(config.params, "pretrained") and isinstance(
            config.params.pretrained, str
        ):
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
            pretrained_config = OmegaConf.to_container(pretrained_config, resolve=True)
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

    def build_sound_event_detection_model(self, config):
        # build model
        backbone_config = {"name": config.params.backbone.name}
        params = config.params.backbone.params

        # if in_chans is valid key
        backbone = build_from_config(backbone_config, BACKBONES, params)

        # params = {k: v for k, v in params.items() if k != "in_channels"}
        # backbone = build_from_config(backbone_config, BACKBONES, params)
        # backbone = update_input_layer(backbone, in_channels)

        in_features = analyze_in_features(backbone)
        backbone = replace_last_linear(
            backbone,
            num_classes=1,
            pool_type="identity",
            use_identity_as_last_layer=True,
        )

        args = {"encoder": backbone, "in_features": in_features}
        model = build_from_config(config, MODELS, default_args=args)
        return model
