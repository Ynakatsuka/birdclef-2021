import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

from .base import LightningModuleBase


def disable_bn(model):
    for module in model.modules():
        if (
            isinstance(module, nn.BatchNorm1d)
            or isinstance(module, nn.BatchNorm2d)
            or isinstance(module, nn.BatchNorm3d)
            or isinstance(module, nn.SyncBatchNorm)
        ):
            module.eval()


def enable_bn(model):
    model.train()


class LightningModuleSAM(LightningModuleBase):
    automatic_optimization = False

    def training_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        aux_x = {k: v for k, v in batch.items() if (k != "x") and (k[0] == "x")}
        aux_y = {k: v for k, v in batch.items() if (k != "y") and (k[0] == "y")}

        if self.transform is not None:
            x = self.transform(x)

        if (
            (self.strong_transform is not None)
            and (self.max_epochs is not None)
            and (
                self.current_epoch
                <= self.max_epochs - self.disable_strong_transform_in_last_epochs
            )
            and (np.random.rand() < self.storong_transform_p)
        ):
            x, y_a, y_b, lam, idx = self.strong_transform(x, y)
            y_hat = self.forward(x, **aux_x)
            loss = lam * self.hooks.loss_fn(y_hat, y_a, **aux_y) + (
                1 - lam
            ) * self.hooks.loss_fn(y_hat, y_b, **aux_y)
        else:
            y_hat = self.forward(x, **aux_x)
            loss = self.hooks.loss_fn(y_hat, y, **aux_y)

        self.manual_backward(loss)

        def closure():
            if self.strong_transform is not None:
                y_hat = self.forward(x, **aux_x)
                loss = lam * self.hooks.loss_fn(y_hat, y_a, **aux_y) + (
                    1 - lam
                ) * self.hooks.loss_fn(y_hat, y_b, **aux_y)
            else:
                y_hat = self.forward(x)
                loss = self.hooks.loss_fn(y_hat, y, **aux_y)

            self.manual_backward(loss)
            return loss

        disable_bn(self.model)
        optimizer = self.optimizers()
        optimizer.step(closure=closure)
        optimizer.zero_grad()
        enable_bn(self.model)

        return {"loss": loss}
