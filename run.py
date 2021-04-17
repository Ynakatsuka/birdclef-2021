from __future__ import absolute_import, division, print_function

import pprint
import sys

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

# local libraries
sys.path.append("src/")
import custom  # import all custom modules for registering objects.
from kvt.apis.inference import run as run_inference
from kvt.apis.swa import run as run_swa
from kvt.apis.train import run as run_train
from kvt.initialization import initialize


def train(config):
    print("---------------------------------------------------------------")
    print("train")
    run_train(config)


def inference(config):
    print("---------------------------------------------------------------")
    print("inference")
    run_inference(config)


def swa(config):
    print("---------------------------------------------------------------")
    print("swa")
    run_swa(config)


@hydra.main(config_path="config", config_name="default")
def main(config: DictConfig) -> None:
    if config.disable_warnings:
        import warnings

        warnings.filterwarnings("ignore")

    if config.print_config:
        pprint.PrettyPrinter(indent=2).pprint(OmegaConf.to_container(config))

    if config.run in ("train", "inference", "swa"):
        # initialize torch
        pl.seed_everything(config.seed)

        # run main function
        eval(config.run)(config)
    else:
        raise ValueError(f"Invalid run mode: {config.run}.")


if __name__ == "__main__":
    initialize()
    main()
