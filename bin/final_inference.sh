#!/bin/bash
python src/misc/inference.py augmentation=exp029 trainer=exp073
python src/misc/inference.py augmentation=exp100 dataset=exp100 trainer.model.params.backbone.name=densenet121 trainer=exp100
python src/misc/inference.py augmentation=exp100 dataset=exp100 trainer.model.params.backbone.name=nfnet_l0 trainer=exp100
python src/misc/inference.py augmentation=exp100 dataset.dataset.params.idx_fold=0 dataset=exp100 trainer=exp107
python src/misc/inference.py augmentation=exp103 dataset=exp100 trainer=exp100
