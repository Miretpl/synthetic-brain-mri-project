FROM python:3.13.0-slim
LABEL authors="miretpl"

RUN apt-get update && \
    apt-get install -y ffmpeg libsm6 libxext6

COPY docker/requirements.txt .
RUN pip install -r requirements.txt && rm requirements.txt

COPY code code
COPY docker/run.sh run.sh
CMD ./run.sh