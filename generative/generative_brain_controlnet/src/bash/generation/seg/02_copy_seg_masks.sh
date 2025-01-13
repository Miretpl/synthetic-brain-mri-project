#!/usr/bin/bash

python /workspace/src/python/generation/copy_data.py \
  --src_dir="/data/raw/extracted" \
  --dst_dir="/data/segmentation/controlnet" \
  --ids="/data/ids/raw/train.tsv" \
  --num_workers=8