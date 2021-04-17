import torch
import torch.nn as nn
import torch.nn.functional as F


def onehot_encoding(label, n_classes):
    return (
        torch.zeros(label.size(0), n_classes)
        .to(label.device)
        .scatter_(1, label.view(-1, 1), 1)
    )


def oh_cross_entropy_loss(input, target, reduction):
    logp = F.log_softmax(input, dim=1)
    loss = torch.sum(-logp * target, dim=1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError("`reduction` must be one of 'none', 'mean', or 'sum'.")


class OHEMLoss(nn.Module):
    def __init__(self, rate=0.7, loss=None, weight=None, params={}, smoothing_eps=0):
        super().__init__()
        self.rate = rate
        self.smoothing_eps = smoothing_eps
        params["weight"] = weight
        self.loss = loss

    def forward(self, pred, target):
        batch_size = pred.size(0)
        if self.loss is not None:
            if self.smoothing_eps > 0:
                onehot = onehot_encoding(target, pred.size(1)).float()
                target = onehot * (1 - self.smoothing_eps) + torch.ones_like(
                    onehot
                ) * self.smoothing_eps / pred.size(1)
            ohem_cls_loss = self.loss(pred, target)
        else:
            if self.smoothing_eps > 0:
                onehot = onehot_encoding(target, pred.size(1)).float()
                target = onehot * (1 - self.smoothing_eps) + torch.ones_like(
                    onehot
                ) * self.smoothing_eps / pred.size(1)
                ohem_cls_loss = oh_cross_entropy_loss(
                    pred, target, reduction="none", ignore_index=-1
                )
            else:
                ohem_cls_loss = F.cross_entropy(pred, target, reduction="none")

        sorted_ohem_loss, idx = torch.sort(ohem_cls_loss, descending=True)
        keep_num = min(sorted_ohem_loss.size()[0], int(batch_size * self.rate))
        if keep_num < sorted_ohem_loss.size()[0]:
            keep_idx_cuda = idx[:keep_num]
            ohem_cls_loss = ohem_cls_loss[keep_idx_cuda]
        cls_loss = ohem_cls_loss.sum() / keep_num

        return cls_loss
