#!/bin/bash

python ./src/code/train.py \
  --name 03 \
  --dataset_mode custom \
  --data_dir /data/generation \
  --ids_path /data/ids/raw/train.tsv \
  --label_nc 4 \
  --no_instance \
  --preprocess_mode none \
  --output_nc 1 \
  --contain_dontcare_label \
  --aspect_ratio 1.4 \
  --crop_size 224 \
  --checkpoints_dir /models \
  --no_html \
  --display_freq 1000 \
  --print_freq 1000 \
  --batchSize 8 \
  --use_vae