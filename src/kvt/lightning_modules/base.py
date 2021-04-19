import numpy as np
import pytorch_lightning as pl
import torch


class LightningModuleBase(pl.LightningModule):
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        hooks,
        dataloaders,
        transform=None,
        strong_transform=None,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.hooks = hooks
        self.dataloaders = dataloaders
        self.transform = transform
        self.strong_transform = strong_transform

        if (
            hasattr(self.hooks, "")
            and (self.hooks.metric_fn is not None)
            and (not isinstance(self.hooks.metric_fn, dict))
        ):
            raise ValueError("metric_fn must be dict.")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
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

        return {"loss": loss}

    # for dp, ddp
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/4073
    def training_step_end(self, training_step_outputs):
        outputs = {name: val.sum() for name, val in training_step_outputs.items()}

        if hasattr(self.logger, "log_metrics"):
            self.logger.log_metrics(outputs)

        return outputs

    def validation_step(self, batch, batch_nb):
        x, y = batch["x"], batch["y"]
        y_hat = self.forward(x)

        val_loss = self.hooks.loss_fn(y_hat, y)
        outputs = {"val_loss": val_loss}

        if self.hooks.metric_fn is not None:
            y_hat = self.hooks.post_forward_fn(y_hat)
            outputs["y_hat"] = y_hat.detach().cpu().numpy()
            outputs["y"] = y.detach().cpu().numpy()

        return outputs

    def validation_epoch_end(self, outputs):
        avg_outputs = {}
        avg_outputs["val_loss"] = torch.stack([o["val_loss"] for o in outputs]).mean()

        if self.hooks.metric_fn is not None:
            y_hat = np.concatenate([o["y_hat"] for o in outputs], axis=0)
            y = np.concatenate([o["y"] for o in outputs], axis=0)
            for name, func in self.hooks.metric_fn.items():
                result = func(y_hat, y)
                if isinstance(result, tuple):
                    # discard detail score metrics
                    result, _ = result[0], result[1]
                avg_outputs[f"val_{name}"] = result

        if not self.trainer.running_sanity_check:
            for key, value in avg_outputs.items():
                self.log(
                    key,
                    value,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )

        return avg_outputs

    def configure_optimizers(self):
        if self.scheduler is not None:
            return [self.optimizer], [self.scheduler]
        else:
            return [self.optimizer]

    def train_dataloader(self):
        dls = [d["dataloader"] for d in self.dataloaders if d["mode"] == "train"]
        assert len(dls) == 1
        return dls[0]

    def val_dataloader(self):
        dls = [d["dataloader"] for d in self.dataloaders if d["mode"] == "validation"]
        assert len(dls) == 1
        return dls[0]

    def test_dataloader(self):
        dls = [d["dataloader"] for d in self.dataloaders if d["mode"] == "test"]
        assert len(dls) == 1
        return dls[0]
