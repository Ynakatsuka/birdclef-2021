from __future__ import absolute_import, division, print_function

import os

import kvt
import kvt.losses
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@kvt.LOSSES.register
class BCEFocalLossHook(nn.Module):
    def __init__(self, weights=[1, 1], class_weights=None):
        super().__init__()

        self.focal = kvt.losses.BinaryFocalLoss(pos_weight=class_weights)
        self.weights = weights
        self.class_weights = class_weights

    def forward(self, input, target):
        target = target.float()

        framewise_output = input["framewise_logit"]
        clipwise_output_with_max, _ = framewise_output.max(dim=1)

        aux_loss = self.focal(clipwise_output_with_max, target)

        return aux_loss


@kvt.LOSSES.register
class BCEFocal2WayLossHook(nn.Module):
    def __init__(self, weights=[1, 1], class_weights=None):
        super().__init__()

        self.focal = kvt.losses.BinaryFocalLoss(pos_weight=class_weights)
        self.weights = weights

    def forward(self, input, target):
        input_ = input["logit"]
        target = target.float()

        framewise_output = input["framewise_logit"]
        clipwise_output_with_max, _ = framewise_output.max(dim=1)

        loss = self.focal(input_, target)
        aux_loss = self.focal(clipwise_output_with_max, target)

        return self.weights[0] * loss + self.weights[1] * aux_loss


@kvt.LOSSES.register
class BCE2WayLossHook(BCEFocal2WayLossHook):
    def __init__(self, weights=[1, 1], class_weights=None):
        super().__init__()

        self.focal = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        self.weights = weights


@kvt.LOSSES.register
class ArcFace2WayLossHook(BCEFocal2WayLossHook):
    def __init__(self, num_classes, weights=[1, 1], class_weights=None):
        super().__init__()

        self.focal = kvt.losses.ArcFaceLoss(num_classes)
        self.weights = weights
        self.class_weights = class_weights


@kvt.LOSSES.register
class OHEM2WayLossHook(BCEFocal2WayLossHook):
    def __init__(self, weights=[1, 1], class_weights=None, **params):
        super().__init__()

        self.focal = kvt.losses.OHEMLossWithLogits(**params)
        self.weights = weights
        self.class_weights = class_weights


@kvt.LOSSES.register
class LabelSmoothingCrossEntropy2WayLossHook(BCEFocal2WayLossHook):
    def __init__(self, weights=[1, 1], class_weights=None):
        super().__init__()

        self.focal = kvt.losses.LabelSmoothingCrossEntropy()
        self.weights = weights
        self.class_weights = class_weights


@kvt.LOSSES.register
class OUSM2WayLossHook(BCEFocal2WayLossHook):
    def __init__(self, weights=[1, 1], class_weights=None):
        super().__init__()

        self.focal = kvt.losses.OUSMLoss(loss="BCEWithLogitsLoss")
        self.weights = weights
        self.class_weights = class_weights


@kvt.LOSSES.register
class SymmetricBinaryFocalLoss2WayLossHook(BCEFocal2WayLossHook):
    def __init__(self, weights=[1, 1], class_weights=None):
        super().__init__()

        self.focal = kvt.losses.SymmetricBinaryFocalLoss()
        self.weights = weights
        self.class_weights = class_weights


@kvt.LOSSES.register
class IterativeSelfLearningLoss2WayLossHook(BCEFocal2WayLossHook):
    def __init__(self, weights=[1, 1], class_weights=None):
        super().__init__()

        self.focal = kvt.losses.IterativeSelfLearningLoss(loss="BCEWithLogitsLoss")
        self.weights = weights
        self.class_weights = class_weights


@kvt.LOSSES.register
class BinaryReducedFocalLoss2WayLossHook(BCEFocal2WayLossHook):
    def __init__(self, weights=[1, 1], class_weights=None):
        super().__init__()

        self.focal = kvt.losses.BinaryReducedFocalLoss(pos_weight=class_weights)
        self.weights = weights


@kvt.LOSSES.register
class BCEFocalLossHook(nn.Module):
    def __init__(self, weights=[1, 1], class_weights=None):
        super().__init__()

        self.focal = kvt.losses.BinaryFocalLoss(pos_weight=class_weights)
        self.weights = weights

    def forward(self, input, target):
        target = target.float()

        framewise_output = input["framewise_logit"]
        clipwise_output_with_max, _ = framewise_output.max(dim=1)

        aux_loss = self.focal(clipwise_output_with_max, target)

        return aux_loss


@kvt.LOSSES.register
class BCEFocal3WayLossHook(nn.Module):
    def __init__(self, weights=[1, 1, 0.1], class_weights=None):
        super().__init__()

        self.focal = kvt.losses.BinaryFocalLoss(pos_weight=class_weights)
        self.weights = weights

    def forward(self, input, target, y_type):
        type_logit = input["type_logit"]

        input_ = input["logit"]
        target = target.float()

        framewise_output = input["framewise_logit"]
        clipwise_output_with_max, _ = framewise_output.max(dim=1)

        loss = self.focal(input_, target)
        type_loss = self.focal(type_logit, y_type)
        aux_loss = self.focal(clipwise_output_with_max, target)

        return (
            self.weights[0] * loss
            + self.weights[1] * aux_loss
            + self.weights[2] * type_loss
        )


# @kvt.LOSSES.register
# class SedScaledPosNegFocalLoss(nn.Module):
#     """
#     My loss function looked something like the below.
#     I wanted to experiment with different parameters
#     but in the end I mainly used the default values, which was just BCELoss.
#     I used a different loss function for 2 of the non-mixup models
#     and it was based on randomly removing the primary label predictions
#     from the loss function, to try increase the secondary_label predictions
#     but I gave up on the approach for the rest of the models
#     since I was running out of time and resources.
#     """

#     def __init__(self, gamma=0.0, alpha_1=1.0, alpha_0=1.0, secondary_factor=1.0):
#         super().__init__()

#         self.loss_fn = nn.BCELoss(reduction="none")
#         self.secondary_factor = secondary_factor
#         self.gamma = gamma
#         self.alpha_1 = alpha_1
#         self.alpha_0 = alpha_0
#         self.loss_keys = ["bce_loss", "F_loss", "FScaled_loss", "F_loss_0", "F_loss_1"]

#     def forward(self, y_pred, y_target):
#         y_true = y_target["all_labels"]
#         y_sec_true = y_target["secondary_labels"]
#         bs, s, o = y_true.shape

#         # Sigmoid has already been applied in the model
#         y_pred = torch.clamp(y_pred, min=EPSILON_FP16, max=1.0 - EPSILON_FP16)
#         y_pred = y_pred.reshape(bs * s, o)
#         y_true = y_true.reshape(bs * s, o)
#         y_sec_true = y_sec_true.reshape(bs * s, o)

#         with torch.no_grad():
#             y_all_ones_mask = torch.ones_like(y_true, requires_grad=False)
#             y_all_zeros_mask = torch.zeros_like(y_true, requires_grad=False)
#             y_all_mask = torch.where(y_true > 0.0, y_all_ones_mask, y_all_zeros_mask)
#             y_ones_mask = torch.ones_like(y_sec_true, requires_grad=False)
#             y_zeros_mask = (
#                 torch.ones_like(y_sec_true, requires_grad=False) * self.secondary_factor
#             )
#             y_secondary_mask = torch.where(y_sec_true > 0.0, y_zeros_mask, y_ones_mask)
#         bce_loss = self.loss_fn(y_pred, y_true)
#         pt = torch.exp(-bce_loss)
#         F_loss_0 = (self.alpha_0 * (1 - y_all_mask)) * (1 - pt) ** self.gamma * bce_loss
#         F_loss_1 = (self.alpha_1 * y_all_mask) * (1 - pt) ** self.gamma * bce_loss

#         F_loss = F_loss_0 + F_loss_1

#         FScaled_loss = y_secondary_mask * F_loss
#         FScaled_loss = FScaled_loss.mean()

#         return FScaled_loss, {
#             "bce_loss": bce_loss.mean(),
#             "F_loss_1": F_loss_1.mean(),
#             "F_loss_0": F_loss_0.mean(),
#             "F_loss": F_loss.mean(),
#             "FScaled_loss": FScaled_loss,
#         }
