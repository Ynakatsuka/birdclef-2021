import pytorch_lightning as pl
import torch

from .base import LightningModuleBase


class LightningModuleSAM(LightningModuleBase):
    automatic_optimization = False

    def training_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        if self.transform is not None:
            x = self.aumgmentation(x)

        if self.strong_transform is not None:
            x, y_a, y_b, lam = self.strong_transform(x, y)
            y_hat = self.forward(x)
            loss = lam * self.hooks.loss_fn(y_hat, y_a) + (
                1 - lam
            ) * self.hooks.loss_fn(y_hat, y_b)
        else:
            y_hat = self.forward(x)
            loss = self.hooks.loss_fn(y_hat, y)

        self.manual_backward(loss)

        def closure():
            if self.strong_transform is not None:
                y_hat = self.forward(x)
                loss = lam * self.hooks.loss_fn(y_hat, y_a) + (
                    1 - lam
                ) * self.hooks.loss_fn(y_hat, y_b)
            else:
                y_hat = self.forward(x)
                loss = self.hooks.loss_fn(y_hat, y)

            self.manual_backward(loss)
            return loss

        optimizer = self.optimizers()
        optimizer.step(closure=closure)
        optimizer.zero_grad()

        return {"loss": loss}
