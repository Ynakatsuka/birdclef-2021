import os
import random

import kvt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy import interpolate


@kvt.DATASETS.register
class WaveformDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_filename,
        image_column,
        target_column,
        target_unique_values,
        input_dir,
        split="train",
        transform=None,
        sample_rate=32000,
        period=20,
        num_fold=5,
        idx_fold=0,
        secondary_target_column=None,
        secondary_coef=0.1,
        addtional_numerical_columns=None,
        **params,
    ):
        self.image_column = image_column
        self.target_column = target_column
        self.target_unique_values = target_unique_values
        self.input_dir = input_dir
        self.split = split
        self.transform = transform
        self.sample_rate = sample_rate
        self.period = period
        self.num_fold = num_fold
        self.idx_fold = idx_fold
        self.secondary_target_column = secondary_target_column
        self.secondary_coef = secondary_coef
        self.addtional_numerical_columns = addtional_numerical_columns

        # load
        df = pd.read_csv(os.path.join(input_dir, csv_filename))

        if self.split == "validation":
            df = df[df.Fold == self.idx_fold]
        elif self.split == "train":
            df = df[df.Fold != self.idx_fold]

        self.image_filenames = df[self.image_column].tolist()
        self.targets = df[self.target_column].tolist()
        if self.secondary_target_column is not None:
            self.secondary_targets = (
                df[self.secondary_target_column].apply(eval).tolist()
            )  # lisf of list

        if self.addtional_numerical_columns is not None:
            self.addtional_numerical_features = (
                df[self.addtional_numerical_columns].values.astype("float32") / 200
            )

        # image dir
        if self.split == "test":
            self.images_dir = "train_soundscapes_clipped"
        else:
            self.images_dir = "train_short_audio"

    def __len__(self):
        return len(self.targets)

    def _preprocess_input(self, x, sr):
        len_x = len(x)
        effective_length = sr * self.period
        if len_x < effective_length:
            new_x = np.zeros(effective_length, dtype=x.dtype)
            if self.split == "train":
                start = np.random.randint(effective_length - len_x)
            else:
                start = 0
            new_x[start : start + len_x] = x
            x = new_x.astype(np.float32)
        elif len_x > effective_length:
            if self.split == "train":
                start = np.random.randint(len_x - effective_length)
            else:
                start = 0
            x = x[start : start + effective_length].astype(np.float32)
        else:
            x = x.astype(np.float32)

        x = np.nan_to_num(x)
        return x

    def _preprocess_target(self, y, secondary_y=None):
        labels = np.zeros(len(self.target_unique_values), dtype="float32")
        labels[self.target_unique_values.index(y)] = 1.0
        if (secondary_y is not None) and (self.split == "train"):
            for sy in secondary_y:
                if sy in self.target_unique_values:
                    labels[self.target_unique_values.index(sy)] = (
                        1.0 * self.secondary_coef
                    )

        return labels

    def __getitem__(self, idx):
        wav_name = self.image_filenames[idx]
        ebird_code = self.targets[idx]

        x, sr = sf.read(
            os.path.join(self.input_dir, self.images_dir, ebird_code, wav_name)
        )
        x = self._preprocess_input(x, sr)

        if self.transform is not None:
            x = self.transform(x, self.sample_rate)
        x = np.nan_to_num(x)

        secondary_ebird_code = None
        if self.secondary_target_column is not None:
            secondary_ebird_code = self.secondary_targets[idx]  # list

        y = self._preprocess_target(ebird_code, secondary_ebird_code)

        input_ = {"x": x, "y": y}

        if self.addtional_numerical_columns is not None:
            input_["additional_x"] = self.addtional_numerical_features[idx]

        return input_
