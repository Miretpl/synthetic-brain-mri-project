#!/usr/bin/bash

dataset_quantity=( 8000 10000 12000 )
for number in "${dataset_quantity[@]}"
do
  python src/copy_data.py \
    --src_dir="/data/raw/extracted" \
    --dst_dir="/data/segmentation/big/mixed/custom/dataset_$number" \
    --ids="/data/ids/segmentation/big/mixed/exp_02/train_$number.tsv" \
    --num_workers=8
done