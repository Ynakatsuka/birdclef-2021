from typing import Callable, List, Optional, Sequence

import kvt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)


@kvt.METRICS.register
def rowwise_micro_f1(
    pred,
    target,
    threshold=0.5,
):
    pred = pred > threshold
    score = f1_score(target, pred, average="samples")
    return score


@kvt.METRICS.register
def rowwise_micro_precision(
    pred,
    target,
    threshold=0.5,
):
    pred = pred > threshold
    score = precision_score(target, pred, average="samples")
    return score


@kvt.METRICS.register
def rowwise_micro_recall(
    pred,
    target,
    threshold=0.5,
):
    pred = pred > threshold
    score = recall_score(target, pred, average="samples")
    return score


@kvt.METRICS.register
def mAP(
    pred,
    target,
):
    score = average_precision_score(target, pred, average=None)
    score = np.nan_to_num(score).mean()
    return score
