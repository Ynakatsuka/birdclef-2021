import torch
import torch.nn as nn


class BCEWithLogitsLossAndIgnoreIndex(nn.Module):
    def __init__(self, ignore_index=-100, **kwargs):
        super().__init__()
        self.ignore_index = ignore_index
        self.bce = torch.nn.BCEWithLogitsLoss(**kwargs)

    def forward(self, input, target):
        target = target * (target != self.ignore_index).float()
        return self.bce(input, target)
