import glob
import os
import sys

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import soundfile as sf
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset

sys.path.append("src/")
import custom  # import all custom modules for registering objects.
import kvt
import kvt.augmentation
import kvt.utils
from kvt.builder import build_hooks, build_lightning_module, build_logger, build_model

# from kvt.evaluate import evaluate
from kvt.initialization import initialize
from kvt.registry import TRANSFORMS
from kvt.utils import build_from_config

from tools import evaluate

LAT_DICT = {"COL": 5.57, "COR": 10.12, "SNE": 38.49, "SSW": 42.47}
LON_DICT = {"COL": -75.85, "COR": -84.51, "SNE": -119.95, "SSW": -76.45}
NUM_CLASSES = 397


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
    ids = []

    data_dir = os.path.join(
        config.trainer.inference.input_dir,
        "*.ogg",
    )
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
        df["site"] = df["row_id"].apply(lambda x: x.split("_")[1])
        df["latitude"] = df["site"].map(LAT_DICT)
        df["longitude"] = df["site"].map(LON_DICT)
        df["date"] = df["row_id"].apply(lambda x: x.split("_")[2].replace("_", "-"))

        # build dataset
        dataset = TestDataset(df, clip, transforms=transform)
        dataloader = DataLoader(
            dataset,
            shuffle=False,
            batch_size=batch_size,
            drop_last=False,
            num_workers=4,
        )

        result = [{"dataloader": dataloader, "split": "test", "mode": "test"}]
        yield result, row_ids


def preprocess_target(x, target_unique_values):
    labels = np.zeros(len(target_unique_values), dtype="float32")
    for y in x:
        if y == "nocall":
            continue
        labels[target_unique_values.index(y)] = 1.0
    return pd.Series(labels)


def run(config):
    pl.seed_everything(config.seed)

    # overwrite path
    OmegaConf.set_struct(config, True)
    with open_dict(config):
        config.trainer.model.params.backbone.params.pretrained = False

    # build logger
    logger = build_logger(config)

    # logging for wandb or mlflow
    if hasattr(logger, "log_hyperparams"):
        for k, v in config.trainer.items():
            if not k in ("metrics", "inference"):
                logger.log_hyperparams(params=v)
        logger.log_hyperparams(params=config.dataset)
        logger.log_hyperparams(params=config.augmentation)

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
        dataloaders=None,
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
    columns = [f"pred_{i:03}" for i in range(config.trainer.model.params.num_classes)]
    outputs = []
    for i, (dataloaders, row_ids) in enumerate(build_test_dataloaders(config)):
        lightning_module.dataloaders = dataloaders
        _, output = evaluate(
            lightning_module,
            hooks,
            config,
            mode="test",
            return_predictions=True,
            tta=config.augmentation.tta,
        )

        output = pd.DataFrame(output[0], columns=columns)
        output["row_id"] = row_ids
        outputs.append(output)
    outputs = pd.concat(outputs).reset_index(drop=True)

    # save predictions dataframe
    path = os.path.join(
        config.trainer.inference.dirpath,
        config.trainer.inference.filename,
    )
    outputs.to_pickle(path)

    # merge labels
    labels = pd.read_csv(f"{config.input_dir}/train_soundscape_labels.csv")
    labels["birds"] = labels["birds"].apply(lambda x: x.split())
    df = labels.merge(outputs)

    # target
    target = (
        df["birds"]
        .apply(lambda x: preprocess_target(x, config.competition.target_unique_values))
        .values
    )

    # evaluate
    result = {}

    pred = df[columns].values[:, :NUM_CLASSES] > 0.5

    for func in [f1_score, precision_score, recall_score]:
        result[f"train_soundscapes_{func.__name__}_50"] = func(
            target, pred, average="samples", zero_division=1
        )

    pred = df[columns].values[:, :NUM_CLASSES] > 0.25

    for func in [f1_score, precision_score, recall_score]:
        result[f"train_soundscapes_{func.__name__}_25"] = func(
            target, pred, average="samples", zero_division=1
        )

    print("Result:")
    print(result)

    # log
    if logger is not None:
        logger.log_metrics(result)


@hydra.main(config_path="../../config", config_name="default")
def main(config: DictConfig) -> None:
    run(config)


if __name__ == "__main__":
    initialize()
    main()
