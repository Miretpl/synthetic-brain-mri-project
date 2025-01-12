FROM nvcr.io/nvidia/pytorch:23.04-py3
LABEL authors="miretpl"

ENV MLFLOW_TRACKING_URI=file:/project/mlruns
ENV TORCH_HOME=/project/outputs/torch_home
ENV HF_HOME=/project/outputs/hf_home

COPY docker/requirements.txt .
RUN pip3 install -r requirements.txt \
    && pip3 uninstall -y accelerate \
    && rm requirements.txt

COPY src src