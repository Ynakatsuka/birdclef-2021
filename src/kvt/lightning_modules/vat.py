import pytorch_lightning as pl
import torch

from .base import LightningModuleBase


class LightningModuleVAT(LightningModuleBase):
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        hooks,
        dataloaders,
        transform=None,
        strong_transform=None,
        validation_keys=["val_loss"],
    ):
        super().__init__(
            model,
            optimizer,
            scheduler,
            hooks,
            dataloaders,
            transform,
            strong_transform,
            validation_keys,
        )
        if not hasattr(hooks, "vat_loss_fn"):
            raise ValueError('hooks must have "vat_loss_fn".')

    def training_step(self, batch, batch_nb):
        x, y, unlabeled_x = batch["x"], batch["y"], batch["unlabeled_x"]

        if self.transform is not None:
            x = self.aumgmentation(x)
        if self.strong_transform is not None:
            x, y = self.strong_transform(x, y)

        vat_loss = self.hooks.vat_loss_fn(self.model, unlabeled_x)
        y_hat = self.forward(x)
        loss = self.hooks.loss_fn(y_hat, y)
        loss = loss + vat_loss

        return {"loss": loss}

    def validation_step(self, batch, batch_nb):
        x, y = batch["x"], batch["y"]
        y_hat = self.forward(x)

        val_loss = self.hooks.loss_fn(y_hat, y)
        outputs = {"val_loss": val_loss}

        if self.hooks.metric_fn is not None:
            y_hat = self.hooks.post_forward_fn(y_hat)

            for name, func in self.hooks.metric_fn.items():
                result = func(y_hat, y)
                if isinstance(result, tuple):
                    # discard detail score metrics
                    result, _ = result[0], result[1]
                outputs[f"val_{name}"] = result
                self.validation_keys.add(f"val_{name}")

        return outputs

