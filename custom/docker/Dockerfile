FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

COPY docker/requirements.txt requirements.txt
RUN pip install -r requirements.txt \
    && rm requirements.txt

ENV COMET_WORKSPACE=''
ENV COMET_API_KEY=''

COPY src src
CMD python src/training.py