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

        self.focal = kvt.losses.BinaryFocalLoss()
        self.weights = weights

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

        self.focal = kvt.losses.BinaryFocalLoss()
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

        self.focal = nn.CrossEntropyLoss()
        self.weights = weights


@kvt.LOSSES.register
class ArcFace2WayLossHook(BCEFocal2WayLossHook):
    def __init__(self, num_classes, weights=[1, 1], class_weights=None):
        super().__init__()

        self.focal = kvt.losses.ArcFaceLoss(num_classes)
        self.weights = weights


@kvt.LOSSES.register
class OHEM2WayLossHook(BCEFocal2WayLossHook):
    def __init__(self, weights=[1, 1], class_weights=None):
        super().__init__()

        self.focal = kvt.losses.OHEMLoss()
        self.weights = weights


@kvt.LOSSES.register
class LabelSmoothingCrossEntropy2WayLossHook(BCEFocal2WayLossHook):
    def __init__(self, weights=[1, 1], class_weights=None):
        super().__init__()

        self.focal = kvt.losses.LabelSmoothingCrossEntropy()
        self.weights = weights


@kvt.LOSSES.register
class OUSM2WayLossHook(BCEFocal2WayLossHook):
    def __init__(self, weights=[1, 1], class_weights=None):
        super().__init__()

        self.focal = kvt.losses.OUSMLoss(loss="BCEWithLogitsLoss")
        self.weights = weights


@kvt.LOSSES.register
class SymmetricBinaryFocalLoss2WayLossHook(BCEFocal2WayLossHook):
    def __init__(self, weights=[1, 1], class_weights=None):
        super().__init__()

        self.focal = kvt.losses.SymmetricBinaryFocalLoss()
        self.weights = weights


@kvt.LOSSES.register
class IterativeSelfLearningLoss2WayLossHook(BCEFocal2WayLossHook):
    def __init__(self, weights=[1, 1], class_weights=None):
        super().__init__()

        self.focal = kvt.losses.IterativeSelfLearningLoss(loss="BCEWithLogitsLoss")
        self.weights = weights
