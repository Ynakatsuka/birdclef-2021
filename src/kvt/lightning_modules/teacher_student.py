import pytorch_lightning as pl
import torch

from .base import LightningModuleBase


class LightningModuleTeacherStudent(LightningModuleBase):
    def __init__(
        self,
        teacher_model,
        model,
        optimizer,
        scheduler,
        hooks,
        dataloaders,
        transform=None,
        strong_transform=None,
        teacher_transforms=None,
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
        self.teacher_model = teacher_model
        self.teacher_transforms = teacher_transforms

    def training_step(self, batch, batch_nb):
        x, y, unlabeled_x = batch["x"], batch["y"], batch["unlabeled_x"]

        if self.teacher_transforms is not None:
            # list of augmentated images
            xs = self.teacher_transforms(unlabeled_x)

        with torch.no_grad():
            pseudo_label = self.teacher_model(xs)
            pseudo_label = torch.mean(pseudo_label, axis=0)

        if self.transform is not None:
            x = self.aumgmentation(x)
        if self.strong_transform is not None:
            x, y = self.strong_transform(x, y)

        unlabeled_y_hat = self.forward(unlabeled_x)
        y_hat = self.forward(x)
        loss = self.hooks.loss_fn(y_hat, y, unlabeled_y_hat, pseudo_label)
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

