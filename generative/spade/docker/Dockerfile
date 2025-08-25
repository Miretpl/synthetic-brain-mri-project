FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
LABEL authors="miretpl"

RUN apt-get update \
    && apt-get install -y git \
    && git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch

COPY docker/requirements.txt requirements.txt
RUN pip install -r requirements.txt \
    && rm requirements.txt

COPY src src
RUN cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm /workspace/src/models/networks/

WORKDIR /workspace