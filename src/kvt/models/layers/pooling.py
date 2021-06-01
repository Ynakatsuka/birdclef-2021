import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1, 1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, inference_p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.inference_p = inference_p
        self.eps = eps

    def forward(self, x):
        if self.training:
            return gem(x, p=self.p, eps=self.eps)
        else:
            return gem(x, p=self.inference_p, eps=self.eps)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )
