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
    pprint.PrettyPrinter(indent=2).pprint(OmegaConf.to_container(config, resolve=True))

    train = pd.read_csv(config.competition.train_path)
    y = train[config.competition.target_column]

    # split
    train["Fold"] = 0
    kfold = instantiate(config.fold.fold)
    for f, (_, valid_index) in enumerate(kfold.split(train, y)):
        train.loc[valid_index, "Fold"] = f
    path = os.path.join(config.input_dir, config.fold.csv_filename)
    train.to_csv(path, index=False)


if __name__ == "__main__":
    main()
