import glob

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

CLIP_SECONDS = 5
DATA_DIR = "data/input/"


def main():
    df = pd.read_csv(DATA_DIR + "train_soundscape_labels.csv")

    results = []
    all_audio_paths = list(glob.glob(DATA_DIR + "train_soundscapes/*.ogg"))
    for path in tqdm(all_audio_paths):
        filename = path.split("/")[-1][:-4]  # .ogg
        audio_id = int(filename.split("_")[0])
        site = filename.split("_")[1]

        clip, sr = sf.read(path)
        total_seconds = len(clip) / sr
        for i in range(int(np.ceil(total_seconds / CLIP_SECONDS))):
            save_filename = f"{filename}_{(i+1)* CLIP_SECONDS}_seconds.ogg"
            start = i * CLIP_SECONDS
            end = start + CLIP_SECONDS
            c = clip[start * sr : min(end * sr, total_seconds * sr)].astype("float32")

            sf.write(DATA_DIR + "train_soundscapes_clipped/" + save_filename, c, sr)

            results.append([audio_id, site, end, save_filename])

    results = pd.DataFrame(
        results,
        columns=["audio_id", "site", "seconds", "filename"],
    )
    results = results.merge(df, how="left")
    results.to_csv(DATA_DIR + "train_soundscape_labels_clip.csv", index=None)


if __name__ == "__main__":
    main()
