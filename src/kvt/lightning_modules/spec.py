import numpy as np

from .base import LightningModuleBase


class LightningModuleSpecMixUp(LightningModuleBase):
    def training_step(self, batch, batch_nb):
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
            _, y_a, y_b, lam, idx = self.strong_transform(x, y)
            y_hat = self.forward(x, mixup_lambda=lam, mixup_index=idx, **aux_x)
            loss = lam * self.hooks.loss_fn(y_hat, y_a, **aux_y) + (
                1 - lam
            ) * self.hooks.loss_fn(y_hat, y_b, **aux_y)
        else:
            y_hat = self.forward(x, **aux_x)
            loss = self.hooks.loss_fn(y_hat, y, **aux_y)

        return {"loss": loss}
