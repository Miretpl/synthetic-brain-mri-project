#!/bin/bash

python ./src/code/test.py \
  --name 01 \
  --preprocess_mode none \
  --crop_size 224 \
  --aspect_ratio 1.4 \
  --label_nc 4 \
  --contain_dontcare_label \
  --output_nc 1 \
  --data_dir /data/generation \
  --ids_path /data/ids/raw/test.tsv \
  --dataset_mode custom \
  --no_instance \
  --results_dir /models/results \
  --which_epoch 50 \
  --checkpoints_dir /models \
  --no_save_samples \
  --gen_type reconstruction