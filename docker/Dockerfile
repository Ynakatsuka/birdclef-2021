ARG BASE_TAG=staging

FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu18.04 AS nvidia
FROM gcr.io/kaggle-images/python:${BASE_TAG}

ADD clean-layer.sh  /tmp/clean-layer.sh

# Cuda support
COPY --from=nvidia /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/
COPY --from=nvidia /etc/apt/sources.list.d/nvidia-ml.list /etc/apt/sources.list.d/
COPY --from=nvidia /etc/apt/trusted.gpg /etc/apt/trusted.gpg.d/cuda.gpg
# See b/142337634#comment28
RUN sed -i 's/deb https:\/\/developer.download.nvidia.com/deb http:\/\/developer.download.nvidia.com/' /etc/apt/sources.list.d/*.list

ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/opt/bin:${PATH}

# Using cuda11.0 instead of 11.1
RUN conda remove --force -y pytorch torchvision torchaudio cpuonly && \
    conda install pytorch==1.8.1 torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge && \
    /tmp/clean-layer.sh

# requirements
ADD requirements.txt  /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# timm
# RUN pip install git+https://github.com/rwightman/pytorch-image-models/ -b v0.1-rs-weights
RUN pip install git+https://github.com/rwightman/pytorch-image-models/ -b v0.1-effv2-weights

# nodejs, jupyterlab extention
RUN npm install n -g && \
    n 15.14.0 && \
    ln -sf /usr/local/bin/node /opt/conda/bin/node && \
    jupyter labextension install jupyterlab-plotly@4.14.3 && \
    jupyter lab build

# credentials
COPY .netrc /root/.netrc
RUN mkdir ./kaggle
COPY kaggle.json /root/.kaggle/kaggle.json
