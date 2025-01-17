FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
LABEL authors="miretpl"

COPY docker/requirements.txt requirements.txt
RUN pip install -r requirements.txt \
    && rm requirements.txt

COPY code code
COPY docker/bash bash