#!/usr/bin/bash

python src/copy_data.py \
  --src_dir="/data/raw/extracted" \
  --dst_dir="/data/segmentation/custom" \
  --ids="/data/ids/raw/train.tsv" \
  --num_workers=8