# cuda11対応
ARG BASE_TAG=staging

FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu18.04 AS nvidia
FROM gcr.io/kaggle-images/python:${BASE_TAG}

ADD clean-layer.sh  /tmp/clean-layer.sh
ADD requirements.txt  /tmp/requirements.txt

# Cuda support
COPY --from=nvidia /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/
COPY --from=nvidia /etc/apt/sources.list.d/nvidia-ml.list /etc/apt/sources.list.d/
COPY --from=nvidia /etc/apt/trusted.gpg /etc/apt/trusted.gpg.d/cuda.gpg
# See b/142337634#comment28
RUN sed -i 's/deb https:\/\/developer.download.nvidia.com/deb http:\/\/developer.download.nvidia.com/' /etc/apt/sources.list.d/*.list

# Ensure the cuda libraries are compatible with the custom Tensorflow wheels.
# TODO(b/120050292): Use templating to keep in sync or COPY installed binaries from it.
ENV CUDA_MAJOR_VERSION=11
ENV CUDA_MINOR_VERSION=1
ENV CUDA_PATCH_VERSION=3
ENV CUDA_VERSION=$CUDA_MAJOR_VERSION.$CUDA_MINOR_VERSION.$CUDA_PATCH_VERSION
ENV CUDA_PKG_VERSION=$CUDA_MAJOR_VERSION-$CUDA_MINOR_VERSION=$CUDA_VERSION-1
LABEL com.nvidia.volumes.needed="nvidia_driver"
LABEL com.nvidia.cuda.version="${CUDA_VERSION}"
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/opt/bin:${PATH}

# Using cuda11.0 instead of 11.1
RUN conda remove --force -y pytorch torchvision torchaudio cpuonly && \
    conda install pytorch==1.8.1 torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge && \
    /tmp/clean-layer.sh

RUN pip install -r /tmp/requirements.txt

RUN pip install git+https://github.com/rwightman/pytorch-image-models/ -b v0.1-dnf-weights

# nodejs, jupyterlab extention
# apt install -y nodejs npm

# RUN conda install -c conda-forge nodejs==15.14.0 && \
#     jupyter labextension install @jupyterlab/plotly-extension
