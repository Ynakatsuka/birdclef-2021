import os
import random

import kvt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy import interpolate

# @kvt.DATASETS.register
# class WaveformDataset(torch.utils.data.Dataset):
#     def __init__(
#         self,
#         df: pd.DataFrame,
#         datadir: Path,
#         img_size=224,
#         waveform_transforms=None,
#         period=20,
#         validation=False,
#     ):
#         self.df = df
#         self.datadir = datadir
#         self.img_size = img_size
#         self.waveform_transforms = waveform_transforms
#         self.period = period
#         self.validation = validation

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx: int):
#         sample = self.df.loc[idx, :]
#         wav_name = sample["filename"]
#         ebird_code = sample["primary_label"]

#         y, sr = sf.read(self.datadir / ebird_code / wav_name)

#         len_y = len(y)
#         effective_length = sr * self.period
#         if len_y < effective_length:
#             new_y = np.zeros(effective_length, dtype=y.dtype)
#             if not self.validation:
#                 start = np.random.randint(effective_length - len_y)
#             else:
#                 start = 0
#             new_y[start : start + len_y] = y
#             y = new_y.astype(np.float32)
#         elif len_y > effective_length:
#             if not self.validation:
#                 start = np.random.randint(len_y - effective_length)
#             else:
#                 start = 0
#             y = y[start : start + effective_length].astype(np.float32)
#         else:
#             y = y.astype(np.float32)

#         y = np.nan_to_num(y)

#         if self.waveform_transforms:
#             y = self.waveform_transforms(y)

#         y = np.nan_to_num(y)

#         labels = np.zeros(len(CFG.target_columns), dtype=float)
#         labels[CFG.target_columns.index(ebird_code)] = 1.0

#         return {"x": y, "y": labels}
