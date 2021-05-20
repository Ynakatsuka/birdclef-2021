import glob

import numpy as np
import pandas as pd
import soundfile as sf
from pandarallel import pandarallel
from tqdm import tqdm

pandarallel.initialize(progress_bar=True)
CLIP_SECONDS = 5
DATA_DIR = "data/input/"


def save_clip(path):
    filename = path.split("/")[-1][:-4]  # .ogg

    clip, sr = sf.read(path)
    assert sr == 32000
    total_seconds = len(clip) / sr
    for i in range(int(np.ceil(total_seconds / CLIP_SECONDS))):
        save_filename = f"{filename}_{(i+1)* CLIP_SECONDS:0>5}_seconds.ogg"
        start = i * CLIP_SECONDS
        end = start + CLIP_SECONDS
        c = clip[start * sr : int(min(end * sr, total_seconds * sr))].astype("float32")

        sf.write(DATA_DIR + "train_short_audio_clipped/" + save_filename, c, sr)


def main():
    all_audio_paths = list(glob.glob(DATA_DIR + "train_short_audio/*/*.ogg"))
    all_audio_paths = pd.Series(all_audio_paths)
    all_audio_paths.parallel_apply(save_clip)


if __name__ == "__main__":
    main()
