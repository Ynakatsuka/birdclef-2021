import glob
import math
import os
import sys

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import soundfile as sf
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append("src/")
import custom  # import all custom modules for registering objects.
import kvt
import kvt.augmentation
import kvt.utils
from kvt.builder import build_hooks, build_lightning_module, build_model

# from kvt.evaluate import evaluate
from kvt.initialization import initialize
from kvt.registry import TRANSFORMS
from kvt.utils import build_from_config

from tools import evaluate

CLIP_SECONDS = 5
DATA_DIR = "../data/input/"
SR = 32000


class ClippedTrainDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transforms=None, idx_fold=0):
        self.transforms = transforms
        self.period = CLIP_SECONDS
        self.idx_fold = idx_fold
        df = df[df.Fold == self.idx_fold]
        self.image_filenames = df["filename"].values
        self.paths = df["path"].values
        self.seconds = df["second"].values

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx: int):
        wav_name = self.image_filenames[idx]
        path = self.paths[idx]
        second = self.seconds[idx]

        x, _ = sf.read(path)

        len_x = len(x)
        effective_length = SR * self.period
        if len_x < effective_length:
            new_x = np.zeros(effective_length, dtype=x.dtype)
            new_x[:len_x] = x
            x = new_x

        x = np.nan_to_num(x.astype(np.float32))

        if self.transforms:
            x = self.transforms(x, SR)

        x = np.nan_to_num(x)

        return {"x": x, "filename": wav_name, "second": second}


def build_short_audio_dataloaders(config):
    split = "validation"
    batch_size = config.trainer.evaluation.batch_size

    # build dataframe
    paths = glob.glob(DATA_DIR + "train_short_audio_clipped/" + "*.ogg")
    df = pd.DataFrame(paths, columns=["path"])
    df["filename_with_seconds"] = df["path"].apply(lambda x: x.split("/")[-1])
    df["second"] = df["filename_with_seconds"].apply(lambda x: int(x.split("_")[1]))
    df["filename"] = df["filename_with_seconds"].apply(
        lambda x: x.split("_")[0] + ".ogg"
    )

    meta = pd.read_csv(DATA_DIR + config.dataset.dataset.params.csv_filename)
    df = df.merge(meta, how="left")

    # build transform
    transform_configs = {
        "split": split,
        "aug_cfg": config.augmentation.get(split),
    }
    transform = build_from_config(
        config.dataset.transform,
        TRANSFORMS,
        default_args=transform_configs,
    )

    # build dataset
    dataset = ClippedTrainDataset(
        df,
        transforms=transform,
        idx_fold=config.dataset.dataset.params.idx_fold,
    )
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        drop_last=False,
        num_workers=24,
    )

    result = [{"dataloader": dataloader, "split": split, "mode": split}]
    return result


def run(config):
    pl.seed_everything(config.seed)

    # overwrite path
    OmegaConf.set_struct(config, True)
    with open_dict(config):
        config.trainer.model.params.backbone.params.pretrained = False

    # build dataloaders
    dataloaders = build_short_audio_dataloaders(config)

    # build model
    model = build_model(config)

    # build hooks
    hooks = build_hooks(config)

    # build lightning module
    lightning_module = build_lightning_module(
        config,
        model=model,
        optimizer=None,
        scheduler=None,
        hooks=hooks,
        dataloaders=dataloaders,
        strong_transform=None,
    )

    # load best checkpoint
    dir_path = config.trainer.callbacks.ModelCheckpoint.dirpath
    filename = f"fold_{config.dataset.dataset.params.idx_fold}_best.ckpt"
    best_model_path = os.path.join(dir_path, filename)

    state_dict = torch.load(best_model_path)["state_dict"]

    # if using dp, it is necessary to fix state dict keys
    if (
        hasattr(config.trainer.trainer, "sync_batchnorm")
        and config.trainer.trainer.sync_batchnorm
    ):
        state_dict = kvt.utils.fix_dp_model_state_dict(state_dict)

    lightning_module.model.load_state_dict(state_dict)

    # inference
    print("---------------------------------------------------------------")
    print("Inference")
    lightning_module.eval()
    lightning_module.cuda()

    secondwise_dirpath = os.path.join(config.trainer.evaluation.dirpath, "secondwise")
    clipwise_dirpath = os.path.join(config.trainer.evaluation.dirpath, "clipwise")

    with torch.no_grad():
        for dl_dict in lightning_module.dataloaders:
            dataloader, split = dl_dict["dataloader"], dl_dict["split"]
            batch_size = dataloader.batch_size
            total_size = len(dataloader.dataset)
            total_step = math.ceil(total_size / batch_size)

            tbar = tqdm(enumerate(dataloader), total=total_step)
            for i, data in tbar:
                x = data["x"].cuda()
                filenames = data["filename"]
                seconds = data["second"]

                outputs = lightning_module(x)

                kernel_size = outputs["framewise_logit"].shape[1] // CLIP_SECONDS

                clip_wise_predictions = (
                    F.sigmoid(outputs["logit"]).detach().cpu().numpy()
                )
                second_wise_predictions = (
                    F.sigmoid(
                        F.max_pool1d(
                            outputs["framewise_logit"].transpose(1, 2),
                            kernel_size=kernel_size,
                        )
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )

                if not os.path.exists(secondwise_dirpath):
                    os.makedirs(secondwise_dirpath)
                if not os.path.exists(clipwise_dirpath):
                    os.makedirs(clipwise_dirpath)

                for filename, second, c_pred, s_pred in zip(
                    filenames, seconds, clip_wise_predictions, second_wise_predictions
                ):
                    c_path = os.path.join(
                        clipwise_dirpath,
                        f"{config.experiment_name}_{filename}_{second:0>5}.npy",
                    )
                    s_path = os.path.join(
                        secondwise_dirpath,
                        f"{config.experiment_name}_{filename}_{second:0>5}.npy",
                    )
                    np.save(c_path, c_pred)
                    np.save(s_path, s_pred)


@hydra.main(config_path="../../config", config_name="default")
def main(config: DictConfig) -> None:
    run(config)


if __name__ == "__main__":
    initialize()
    main()
