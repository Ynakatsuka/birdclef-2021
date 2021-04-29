import glob
import os
import sys

import hydra
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

sys.path.append("src/")
import custom  # import all custom modules for registering objects.
import kvt
import kvt.augmentation
import kvt.utils
from kvt.builder import build_hooks, build_lightning_module, build_model
from kvt.evaluate import evaluate
from kvt.initialization import initialize
from kvt.registry import TRANSFORMS
from kvt.utils import build_from_config

if "kaggle" in os.listdir("./"):
    DATADIR = "../input/birdclef-2021/test_soundscapes/"
else:
    DATADIR = "data/input/test_soundscapes/"


class TestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, clip: np.ndarray, transforms=None):
        self.df = df
        self.clip = clip
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        SR = 32000
        sample = self.df.loc[idx, :]
        row_id = sample.row_id

        end_seconds = int(sample.seconds)
        start_seconds = int(end_seconds - 5)

        start_index = SR * start_seconds
        end_index = SR * end_seconds

        x = self.clip[start_index:end_index].astype(np.float32)
        x = np.nan_to_num(x)

        if self.transforms:
            x = self.transforms(x, SR)

        x = np.nan_to_num(x)

        return {"x": x}


def build_test_dataloaders(config):
    split = "test"
    batch_size = config.trainer.evaluation.batch_size

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

    dataloaders = []

    data_dir = os.path.join(hydra.utils.get_original_cwd(), DATADIR, "*.ogg")
    all_audios = list(glob.glob(data_dir))
    for audio_path in all_audios:
        clip, _ = sf.read(audio_path)
        name = audio_path.split("/")[-1]

        seconds = []
        row_ids = []
        for second in range(5, 605, 5):
            row_id = "_".join(name.split("_")[:2]) + f"_{second}"
            seconds.append(second)
            row_ids.append(row_id)

        df = pd.DataFrame({"row_id": row_ids, "seconds": seconds})

        # build dataset
        dataset = TestDataset(df, clip, transforms=transform)
        dataloader = DataLoader(
            dataset,
            shuffle=False,
            batch_size=batch_size,
            drop_last=False,
            num_workers=4,
            pin_memory=True,
        )

        result = {"dataloader": dataloader, "split": "test", "mode": "test"}
        dataloaders.append(result)

    return dataloaders


def run(config):
    # build model
    model = build_model(config)

    # build hooks
    hooks = build_hooks(config)

    # build datasets
    dataloaders = build_test_dataloaders(config)

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

    # evaluate
    evaluate(lightning_module, hooks, config, mode="test")


@hydra.main(config_path="../../config", config_name="default")
def main(config: DictConfig) -> None:
    run(config)


if __name__ == "__main__":
    initialize()
    main()
