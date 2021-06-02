#!/bin/bash
python src/misc/download_resnest50.py

# training
./bin/train.sh augmentation=exp029 trainer=exp073
./bin/train.sh augmentation=exp100 dataset=exp100 trainer.model.params.backbone.name=densenet121 trainer=exp100
./bin/train.sh augmentation=exp100 dataset=exp100 trainer.model.params.backbone.name=nfnet_l0 trainer=exp100
./bin/train.sh augmentation=exp100 dataset.dataset.params.idx_fold=0 dataset=exp100 trainer=exp107
./bin/train.sh augmentation=exp103 dataset=exp100 trainer=exp100
