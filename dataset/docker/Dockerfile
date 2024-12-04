FROM python:3.10.14
LABEL authors="miretpl"

RUN apt-get update && \
    apt-get install -y ffmpeg libsm6 libxext6

COPY dataset/docker/requirements.txt .
RUN pip install -r requirements.txt && rm requirements.txt

COPY dataset/code code
COPY dataset/docker/run.sh run.sh
CMD ./run.sh