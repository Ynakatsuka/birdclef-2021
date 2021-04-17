from typing import Callable, List, Optional, Sequence

import kvt
import pytorch_lightning as pl
import torch


@kvt.METRICS.register
def multilabel_auroc(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_weight: Optional[Sequence] = None,
    num_classes: Optional[int] = None,
    class_names: List[str] = None,
) -> torch.Tensor:
    num_classes = pl.metrics.utils.get_num_classes(pred, target, num_classes)

    result = {}
    class_auroc_vals = []
    for c in range(num_classes):
        pred_c = pred[:, c]
        target_c = target[:, c].int()

        if target_c.sum() == 0:
            continue
        if target_c.min() == 1:
            continue

        score = pl.metrics.functional.classification.auroc(
            pred_c, target_c, sample_weight
        )
        class_auroc_vals.append(score)

        if class_names is not None:
            result[class_names[c]] = score

    score = torch.mean(torch.stack(class_auroc_vals))

    return score, result
