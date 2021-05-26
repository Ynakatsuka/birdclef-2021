## Setup Docker
```
git clone https://github.com/Ynakatsuka/birdclef-2021
cd birdclef-2021
echo -e "machine api.wandb.ai \n  login user \n  password XXXXXXXXXXXX" >> docker/.netrc
chmod 600 docker/.netrc
echo '{"username":"XXXXXXXXXXXX","key":"XXXXXXXXXXXX"}' >> docker/kaggle.json
chmod 600 docker/kaggle.json

./bin/docker.sh
```
