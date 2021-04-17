import torch
from torch import nn


class L2Distance(nn.Module):
    def forward(self, input, input2):
        diff = input - input2
        dist_sq = torch.mean(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)
        return dist