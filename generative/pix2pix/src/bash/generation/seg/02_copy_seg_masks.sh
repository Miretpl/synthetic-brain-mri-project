#!/bin/bash

python src/code/copy_data.py \
  --src_dir="/data/raw/extracted" \
  --dst_dir="/data/segmentation/pix2pix" \
  --ids="/data/ids/raw/train.tsv" \
  --num_workers=8