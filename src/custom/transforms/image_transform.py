from __future__ import absolute_import, division, print_function

import albumentations as albu
import cv2
import kvt
import kvt.augmentation
import numpy as np


def get_training_augmentation(resize_to=(320, 640)):
    print(
        "[get_training_augmentation] crop_size:", crop_size, ", resize_to:", resize_to
    )

    train_transform = [
        albu.Resize(*resize_to),
        albu.Normalize(),
    ]

    return albu.Compose(train_transform)


def get_test_augmentation(resize_to=(320, 640)):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.Resize(*resize_to),
        albu.Normalize(),
    ]
    return albu.Compose(test_transform)


def get_transform(cfg):
    def get_object(trans):
        params = trans.params if trans.params is not None else {}

        if trans.name in {"Compose", "OneOf"}:
            augs_tmp = [get_object(aug) for aug in trans.member]
            return getattr(albu, trans.name)(augs_tmp, **params)

        if hasattr(albu, trans.name):
            return getattr(albu, trans.name)(**params)
        elif hasattr(kvt.augmentation, trans.name):
            return getattr(kvt.augmentation, trans.name)(**params)
        else:
            return eval(trans.name)(**params)

    augs = [get_object(t) for t in cfg]

    return albu.Compose(augs)


@kvt.TRANSFORMS.register
def base_image_transform(split, aug_cfg=None, height=256, width=256, tta=1, **_):
    resize_to = (height, width)

    print("resize_to:", resize_to)
    print("tta:", tta)

    if aug_cfg is not None:
        aug = get_transform(aug_cfg)
    # use default transform
    elif split == "train":
        aug = get_training_augmentation(resize_to)
    else:
        aug = get_test_augmentation(resize_to)

    def transform(image, mask=None):
        def _transform(image):
            if split == "train":
                augmented = aug(image=image)
            else:
                augmented = aug(image=image)

            if (split == "test") and (tta > 1):
                images = []
                images.append(augmented["image"])
                images.append(aug(image=np.fliplr(image))["image"])
                if tta > 2:
                    images.append(aug(image=np.flipud(image))["image"])
                if tta > 3:
                    images.append(aug(image=np.flipud(np.fliplr(image)))["image"])
                image = np.stack(images, axis=0)
                image = np.transpose(image, (0, 3, 1, 2))
            else:
                image = augmented["image"]
                image = np.transpose(image, (2, 0, 1))
            return image

        image = _transform(image)

        if mask is not None:
            mask = _transform(mask)
            return {"image": image, "mask": mask}

        return image

    return transform
