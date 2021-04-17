import os
import pprint
import sys

sys.path.append("src/")

import hydra
import numpy as np
import pandas as pd
import tqdm
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../../config", config_name="default")
def main(config: DictConfig) -> None:
    print("-------------------------------------------------------------------")
    pprint.PrettyPrinter(indent=2).pprint(OmegaConf.to_container(config))

    image_column = config.dataset.dataset.params.image_column
    target_column = config.dataset.dataset.params.target_column

    train = pd.read_csv(config.catalog.train_path)
    train[image_column] = train[image_column].apply(lambda x: f"{x}.jpg")

    group = train[image_column]
    X = train[image_column]
    y = train[target_column]

    # split
    train["Fold"] = 0
    kfold = instantiate(config.fold.fold)
    for f, (_, valid_index) in enumerate(kfold.split(X, y, group)):
        train.loc[valid_index, "Fold"] = f
    path = os.path.join(config.fold.data_dir, config.fold.csv_filename)
    train.to_csv(path, index=False)

    # train annotation
    train_annotation = pd.read_csv(config.catalog.train_annotation_path)
    train_annotation[image_column] = train_annotation[image_column].apply(
        lambda x: f"{x}.jpg"
    )
    train_annotation.to_csv(
        config.catalog.preprocessed_train_annotation_path, index=False
    )


if __name__ == "__main__":
    main()
