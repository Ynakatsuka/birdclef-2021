from __future__ import absolute_import, division, print_function

import audiomentations as audi
import cv2
import kvt
import kvt.augmentation
import numpy as np


def get_training_augmentation():
    print(
        "[get_training_augmentation] crop_size:", crop_size, ", resize_to:", resize_to
    )

    train_transform = [
        audi.Normalize(),
    ]

    return audi.Compose(train_transform)


def get_test_augmentation():
    """Add paddings to make audio shape divisible by 32"""
    test_transform = [
        audi.Normalize(),
    ]
    return audi.Compose(test_transform)


def get_transform(cfg):
    def get_object(trans):
        params = trans.params if trans.params is not None else {}

        if trans.name in {"Compose", "OneOf"}:
            augs_tmp = [get_object(aug) for aug in trans.member]
            return getattr(audi, trans.name)(augs_tmp, **params)

        if hasattr(audi, trans.name):
            return getattr(audi, trans.name)(**params)
        elif hasattr(kvt.augmentation, trans.name):
            return getattr(kvt.augmentation, trans.name)(**params)
        else:
            return eval(trans.name)(**params)

    augs = [get_object(t) for t in cfg]

    return audi.Compose(augs)


@kvt.TRANSFORMS.register
def base_audio_transform(split, aug_cfg=None, **_):
    if aug_cfg is not None:
        aug = get_transform(aug_cfg)
    # use default transform
    elif split == "train":
        aug = get_training_augmentation()
    else:
        aug = get_test_augmentation()

    return aug
