import numpy as np
import pytorch_lightning as pl
import torch


class AutoClip:
    def __init__(self, percentile=0.25):
        self.grad_history = []
        self.percentile = percentile

    def compute_grad_norm(self, model):
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)

        return total_norm

    def __call__(self, model):
        grad_norm = self.compute_grad_norm(model)
        self.grad_history.append(grad_norm)
        clip_value = np.percentile(self.grad_history, self.percentile)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)


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
        storong_transform_p=None,
        disable_strong_transform_in_last_epochs=5,
        max_epochs=None,
        enable_autoclip=False,
        autoclip_percentile=0.25,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.hooks = hooks
        self.dataloaders = dataloaders
        self.transform = transform
        self.strong_transform = strong_transform
        self.storong_transform_p = storong_transform_p
        self.disable_strong_transform_in_last_epochs = (
            disable_strong_transform_in_last_epochs
        )
        self.max_epochs = max_epochs
        self.enable_autoclip = enable_autoclip

        if self.enable_autoclip:
            self.autoclip = AutoClip(percentile=autoclip_percentile)

        if (
            hasattr(self.hooks, "")
            and (self.hooks.metric_fn is not None)
            and (not isinstance(self.hooks.metric_fn, dict))
        ):
            raise ValueError("metric_fn must be dict.")

    def forward(self, x, **kwargs):
        return self.model(x, **kwargs)

    def backward(self, loss, optimizer, optimizer_idx, *args, **kwargs):
        if self.automatic_optimization or self._running_manual_backward:
            loss.backward(*args, **kwargs)
            if self.enable_autoclip:
                self.autoclip(self.model)

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
            x, y_a, y_b, lam, idx = self.strong_transform(x, y)
            y_hat = self.forward(x, **aux_x)
            loss = lam * self.hooks.loss_fn(y_hat, y_a, **aux_y) + (
                1 - lam
            ) * self.hooks.loss_fn(y_hat, y_b, **aux_y)
        else:
            y_hat = self.forward(x, **aux_x)
            loss = self.hooks.loss_fn(y_hat, y, **aux_y)

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
        aux_x = {k: v for k, v in batch.items() if (k != "x") and (k[0] == "x")}
        aux_y = {k: v for k, v in batch.items() if (k != "y") and (k[0] == "y")}

        y_hat = self.forward(x, **aux_x)

        val_loss = self.hooks.loss_fn(y_hat, y, **aux_y)
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
        assert len(dls) <= 1
        return dls[0]

    def val_dataloader(self):
        dls = [d["dataloader"] for d in self.dataloaders if d["mode"] == "validation"]
        assert len(dls) <= 1
        if len(dls):
            return dls[0]
        else:
            return None

    def test_dataloader(self):
        dls = [d["dataloader"] for d in self.dataloaders if d["mode"] == "test"]
        assert len(dls) <= 1
        if len(dls):
            return dls[0]
        else:
            return None
