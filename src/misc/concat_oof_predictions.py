import glob

import IPython.display as play
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from pandarallel import pandarallel
from tqdm import tqdm_notebook as tqdm

pandarallel.initialize(progress_bar=True)
CLIP_SECONDS = 5
missing_label_threshold = 0.9
secondary_label_threshold = 0.7


def save_second_wise_predictions(filename):
    paths = [p for p in clip_paths if filename in p]
    preds = np.concatenate([np.load(p).reshape(-1, 1) for p in paths], axis=1).T
    preds = np.tile(preds, (1, CLIP_SECONDS)).reshape(len(preds) * CLIP_SECONDS, -1)

    label = meta.loc[meta.filename == filename, "primary_label"].values[0]
    secondary_labels = meta.loc[meta.filename == filename, "secondary_labels"].values[0]

    result = pd.DataFrame(
        {
            "filename": filename,
            "raw_pred": [p for p in preds],
            "pred_missing_labels": [
                unique_labels[p >= missing_label_threshold] for p in preds
            ],
            "pred_secondary_labels": [
                unique_labels[p >= secondary_label_threshold] for p in preds
            ],
            "second": range(len(preds)),
        }
    )
    result["pred_secondary_labels"] = result["pred_secondary_labels"].apply(
        lambda x: list(set(x) & set(secondary_labels))
    )
    result.to_pickle(f"data/input/pred_labels/{filename}.pkl")


meta = pd.read_csv("data/input/train_metadata.csv")
unique_labels = meta.primary_label.unique()
filenames = sorted(meta.filename.values)
meta["secondary_labels"] = meta["secondary_labels"].apply(eval)
meta.head(3)


clip_paths = sorted(glob.glob("data/output/predictions/oof/clipwise/*.npy"))
second_paths = sorted(glob.glob("data/output/predictions/oof/secondwise/*.npy"))
assert len(clip_paths) == len(second_paths)


filenames = pd.Series(filenames)
filenames.parallel_apply(save_second_wise_predictions)
