import glob
import math
import os

import kvt
import librosa
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
        secondary_coef=1,
        addtional_numerical_columns=None,
        addtional_target_columns=None,
        use_head_or_tail=False,
        use_external_data_as_nocall=False,
        external_datadir=None,
        nocall_replace_probability=0,
        label_smoothing=0,
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
        self.addtional_target_columns = addtional_target_columns
        self.use_head_or_tail = use_head_or_tail
        self.use_external_data_as_nocall = use_external_data_as_nocall
        self.external_datadir = external_datadir
        self.nocall_replace_probability = nocall_replace_probability
        self.label_smoothing = label_smoothing

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

        if self.addtional_target_columns is not None:
            if (len(addtional_target_columns) == 1) and (
                "type" in addtional_target_columns
            ):
                df["__target_song__"] = (
                    df["type"]
                    .apply(eval)
                    .apply(lambda x: int(bool(sum([1 for w in x if "song" in w]))))
                )
                df["__target_call__"] = (
                    df["type"]
                    .apply(eval)
                    .apply(lambda x: int(bool(sum([1 for w in x if "call" in w]))))
                )
                self.addtional_targets = df[
                    ["__target_song__", "__target_call__"]
                ].values
            else:
                raise ValueError

        if self.use_external_data_as_nocall:
            self.target_unique_values += ["nocall"]
            self.nocall_paths = glob.glob(f"{external_datadir}/*.wav")
            assert len(self.nocall_paths) > 0

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
            start = 0
            if self.split == "train":
                if self.use_head_or_tail:
                    if np.random.rand() < 0.5:
                        start = len(x) - effective_length
                else:
                    start = np.random.randint(len_x - effective_length)
            x = x[start : start + effective_length].astype(np.float32)
        else:
            x = x.astype(np.float32)

        x = np.nan_to_num(x)
        return x, start

    def _preprocess_target(self, y, secondary_y=None):
        if self.split == "train":
            smoothing = self.label_smoothing
        else:
            smoothing = 0

        labels = np.zeros(len(self.target_unique_values), dtype="float32") + smoothing
        labels[self.target_unique_values.index(y)] = 1.0 - smoothing
        if (secondary_y is not None) and (self.split == "train"):
            for sy in secondary_y:
                if sy in self.target_unique_values:
                    if self.secondary_coef > 0:
                        labels[self.target_unique_values.index(sy)] = (
                            1.0 - smoothing
                        ) * self.secondary_coef
                    else:
                        labels[
                            self.target_unique_values.index(sy)
                        ] = self.secondary_coef
        return labels

    def __getitem__(self, idx):
        if (
            self.use_external_data_as_nocall
            and (np.random.rand() <= self.nocall_replace_probability)
            and (self.split == "train")
        ):
            i = np.random.randint(len(self.nocall_paths))
            ebird_code = "nocall"
            x, sr = librosa.load(self.nocall_paths[i], sr=self.sample_rate)
        else:
            ebird_code = self.targets[idx]
            x, sr = sf.read(
                os.path.join(
                    self.input_dir,
                    self.images_dir,
                    ebird_code,
                    self.image_filenames[idx],
                )
            )

        x, start = self._preprocess_input(x, sr)

        if self.transform is not None:
            x = self.transform(x, self.sample_rate)
        x = np.nan_to_num(x)

        secondary_ebird_code = None
        if self.secondary_target_column is not None:
            secondary_ebird_code = self.secondary_targets[idx]  # list

        y = self._preprocess_target(ebird_code, secondary_ebird_code)

        input_ = {"x": x, "y": y}

        if self.addtional_numerical_columns is not None:
            input_["x_additional"] = self.addtional_numerical_features[idx]

        if self.addtional_target_columns is not None:
            input_["y_type"] = self.addtional_targets[idx]

        return input_


@kvt.DATASETS.register
class WaveformDatasetWithMissingLabels(WaveformDataset):
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
        secondary_coef=0.5,
        addtional_numerical_columns=None,
        addtional_target_columns=None,
        use_head_or_tail=False,
        use_pred_missing_label=True,
        use_pred_secondary_label=False,
        apply_adaptive_sampleing=False,
        **params,
    ):
        super().__init__(
            csv_filename=csv_filename,
            image_column=image_column,
            target_column=target_column,
            target_unique_values=target_unique_values,
            input_dir=input_dir,
            split=split,
            transform=transform,
            sample_rate=sample_rate,
            period=period,
            num_fold=num_fold,
            idx_fold=idx_fold,
            secondary_target_column=secondary_target_column,
            secondary_coef=secondary_coef,
            addtional_numerical_columns=addtional_numerical_columns,
            addtional_target_columns=addtional_target_columns,
            use_head_or_tail=use_head_or_tail,
            **params,
        )
        self.use_pred_missing_label = use_pred_missing_label
        self.use_pred_secondary_label = use_pred_secondary_label
        self.apply_adaptive_sampleing = apply_adaptive_sampleing

    def _preprocess_input(self, x, sr, oof_preds, ebird_code):
        len_x = len(x)
        effective_length = sr * self.period
        start = 0
        if len_x < effective_length:
            new_x = np.zeros(effective_length, dtype=x.dtype)
            if self.split == "train":
                start = np.random.randint(effective_length - len_x)
            new_x[start : start + len_x] = x
            x = new_x.astype(np.float32)
        elif len_x > effective_length:
            if self.split == "train":
                if self.use_head_or_tail:
                    if np.random.rand() < 0.5:
                        start = len(x) - effective_length
                else:
                    if self.apply_adaptive_sampleing:
                        preds = (
                            oof_preds.raw_pred.apply(
                                lambda x: x[self.target_unique_values.index(ebird_code)]
                            )
                            .rolling(self.period)
                            .max()
                            .dropna()
                            .values
                        )
                        p = preds / preds.sum()
                        if len(p):
                            start = min(
                                sr * np.random.choice(range(len(p)), 1, p=p)[0],
                                len_x - effective_length,
                            )
                        else:
                            start = np.random.randint(len_x - effective_length)
                    else:
                        start = np.random.randint(len_x - effective_length)
            x = x[start : start + effective_length].astype(np.float32)
        else:
            x = x.astype(np.float32)

        x = np.nan_to_num(x)
        return x, start

    def __getitem__(self, idx):
        wav_name = self.image_filenames[idx]
        ebird_code = self.targets[idx]

        x, sr = sf.read(
            os.path.join(self.input_dir, self.images_dir, ebird_code, wav_name)
        )
        oof_preds = pd.read_pickle(
            os.path.join(self.input_dir, "pred_labels", wav_name + ".pkl")
        )[: math.ceil(len(x) / sr)]

        x, start = self._preprocess_input(x, sr, oof_preds, ebird_code)

        if self.transform is not None:
            x = self.transform(x, self.sample_rate)
        x = np.nan_to_num(x)

        secondary_ebird_code = set()
        if self.use_pred_missing_label:
            secondary_ebird_code |= set(
                oof_preds.loc[
                    oof_preds.second.between(start // sr, start // sr + self.period),
                    "pred_missing_labels",
                ]
                .explode()
                .dropna()
                .values
            )
        if self.use_pred_secondary_label:
            secondary_ebird_code |= set(
                oof_preds.loc[
                    oof_preds.second.between(start // sr, start // sr + self.period),
                    "pred_secondary_labels",
                ]
                .explode()
                .dropna()
                .values
            )
        secondary_ebird_code -= set(ebird_code)

        y = self._preprocess_target(ebird_code, secondary_ebird_code)

        input_ = {"x": x, "y": y}

        if self.addtional_numerical_columns is not None:
            input_["x_additional"] = self.addtional_numerical_features[idx]

        if self.addtional_target_columns is not None:
            input_["y_type"] = self.addtional_targets[idx]

        return input_


@kvt.DATASETS.register
class WaveformDatasetWith2020CompetitionData(WaveformDataset):
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
        secondary_coef=1,
        addtional_numerical_columns=None,
        addtional_target_columns=None,
        use_head_or_tail=False,
        use_external_data_as_nocall=False,
        external_datadir=None,
        nocall_replace_probability=0,
        label_smoothing=0,
        num_repeat_external_data=10,
        **params,
    ):
        super().__init__(
            csv_filename=csv_filename,
            image_column=image_column,
            target_column=target_column,
            target_unique_values=target_unique_values,
            input_dir=input_dir,
            split=split,
            transform=transform,
            sample_rate=sample_rate,
            period=period,
            num_fold=num_fold,
            idx_fold=idx_fold,
            secondary_target_column=secondary_target_column,
            secondary_coef=secondary_coef,
            addtional_numerical_columns=addtional_numerical_columns,
            addtional_target_columns=addtional_target_columns,
            use_head_or_tail=use_head_or_tail,
            **params,
        )

        self.num_repeat_external_data = num_repeat_external_data

        # load 2020 dataset
        paths = glob.glob(
            "../data/external/birdclef2020-validation-audio-and-ground-truth/gt/*"
        )
        self.dfs = []
        for path in paths:
            filename = path.split("/")[-1][:-4]
            df = pd.read_csv(path, header=None)
            df.columns = ["second", "bird"]
            df["second"] = pd.to_datetime(df["second"].apply(lambda x: x[:8]))
            df["second"] = df["second"].dt.minute * 60 + df["second"].dt.second
            df["filename"] = filename
            df = df[df.bird.isin(target_unique_values)].reset_index(drop=True)
            if len(df):
                self.dfs.append(df)

    def __len__(self):
        if self.split == "train":
            return len(self.targets) + len(self.dfs) * self.num_repeat_external_data
        else:
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
            start = 0
            if self.split == "train":
                if self.use_head_or_tail:
                    if np.random.rand() < 0.5:
                        start = len(x) - effective_length
                else:
                    start = np.random.randint(len_x - effective_length)
            x = x[start : start + effective_length].astype(np.float32)
        else:
            x = x.astype(np.float32)

        x = np.nan_to_num(x)
        return x, start

    def _preprocess_target(self, y, secondary_y=None):
        if self.split == "train":
            smoothing = self.label_smoothing
        else:
            smoothing = 0

        labels = np.zeros(len(self.target_unique_values), dtype="float32") + smoothing
        labels[self.target_unique_values.index(y)] = 1.0 - smoothing
        if (secondary_y is not None) and (self.split == "train"):
            for sy in secondary_y:
                if sy in self.target_unique_values:
                    if self.secondary_coef > 0:
                        labels[self.target_unique_values.index(sy)] = (
                            1.0 - smoothing
                        ) * self.secondary_coef
                    else:
                        labels[
                            self.target_unique_values.index(sy)
                        ] = self.secondary_coef

        return labels

    def __getitem__(self, idx):
        if idx < len(self.targets):
            if (
                self.use_external_data_as_nocall
                and (np.random.rand() <= self.nocall_replace_probability)
                and (self.split == "train")
            ):
                i = np.random.randint(len(self.nocall_paths))
                ebird_code = "nocall"
                x, sr = librosa.load(self.nocall_paths[i], sr=self.sample_rate)
            else:
                ebird_code = self.targets[idx]
                x, sr = sf.read(
                    os.path.join(
                        self.input_dir,
                        self.images_dir,
                        ebird_code,
                        self.image_filenames[idx],
                    )
                )

            x, start = self._preprocess_input(x, sr)

            if self.transform is not None:
                x = self.transform(x, self.sample_rate)
            x = np.nan_to_num(x)

            secondary_ebird_code = None
            if self.secondary_target_column is not None:
                secondary_ebird_code = self.secondary_targets[idx]  # list

            y = self._preprocess_target(ebird_code, secondary_ebird_code)
        else:
            i = (idx - len(self.targets)) // self.num_repeat_external_data
            df = self.dfs[i]
            filename = df.filename.values[0]
            x, sr = sf.read(
                f"../data/external/birdclef2020-validation-audio-and-ground-truth/audio/{filename}.wav"
            )
            assert sr == 32000

            start_second = df.second.values[np.random.randint(len(df))]
            start_second = max(start_second - np.random.randint(self.period // 2), 0)
            if start_second + self.period > 600:
                start_second = 600 - self.period
            start = start_second * sr

            x = x[start : start + sr * self.period]
            x = x.astype(np.float32)
            x = np.nan_to_num(x)
            if self.transform is not None:
                x = self.transform(x, self.sample_rate)
            x = np.nan_to_num(x)

            ebird_codes = list(
                set(
                    df.bird[df.second.between(start_second, start_second + self.period)]
                )
            )
            ebird_code = ebird_codes[0]
            secondary_ebird_code = ebird_codes[1:]
            y = self._preprocess_target(ebird_code, secondary_ebird_code)

        input_ = {"x": x, "y": y}

        return input_
