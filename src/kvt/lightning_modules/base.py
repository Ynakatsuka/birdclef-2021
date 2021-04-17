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
        validation_keys=["val_loss"],
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.hooks = hooks
        self.dataloaders = dataloaders
        self.transform = transform
        self.strong_transform = strong_transform
        self.validation_keys = set(validation_keys)

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

            for name, func in self.hooks.metric_fn.items():
                result = func(y_hat, y)
                if isinstance(result, tuple):
                    # discard detail score metrics
                    result, _ = result[0], result[1]
                outputs[f"val_{name}"] = result
                self.validation_keys.add(f"val_{name}")

        return outputs

    def validation_epoch_end(self, outputs):
        avg_outputs = {}
        for key in self.validation_keys:
            avg_outputs[key] = torch.stack([o[key] for o in outputs]).mean()

        if hasattr(self.logger, "log_metrics") and (
            not self.trainer.running_sanity_check
        ):
            self.logger.log_metrics(avg_outputs)

        return avg_outputs

    def configure_optimizers(self):
        if self.scheduler is not None:
            return [self.optimizer], [self.scheduler]
        else:
            return [self.optimizer]

    def train_dataloader(self):
        return [d["dataloader"] for d in self.dataloaders if d["mode"] == "train"]

    def val_dataloader(self):
        return [d["dataloader"] for d in self.dataloaders if d["mode"] == "validation"]

    def test_dataloader(self):
        return [d["dataloader"] for d in self.dataloaders if d["mode"] == "test"]
