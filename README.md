# 21st Solution for BirdCLEF 2021 - Birdcall Identification
This repository is forked from [@pudae](https://www.kaggle.com/pudae81)'s 
repository [here](https://github.com/pudae/kaggle-understanding-clouds).

## How to Run
```
git clone https://github.com/Ynakatsuka/birdclef-2021
cd birdclef-2021

# credentials
echo -e "machine api.wandb.ai \n  login user \n  password XXXXXXXXXXXX" >> docker/.netrc
echo '{"username":"XXXXXXXXXXXX","key":"XXXXXXXXXXXX"}' >> docker/kaggle.json
chmod 600 docker/kaggle.json

# Create Docker container and execute
./bin/docker.sh

# --------- inside container ---------
# download data
cd data/input/ && kaggle competitions download -c birdclef-2021 && unzip birdclef-2021.zip  && cd ../..

# split fold
python src/misc/fold.py

# prepare for validation
python src/misc/split_train_soundscape.py

# train
./bin/final_train.sh

# inference
./bin/final_inference.sh
```
