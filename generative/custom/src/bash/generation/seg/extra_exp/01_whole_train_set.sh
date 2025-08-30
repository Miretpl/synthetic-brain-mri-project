#!/bin/bash

dataset_quantity=( 8000 10000 12000 )
for number in "${dataset_quantity[@]}"
do
  python src/generation.py \
    --run_id=39 \
    --output_dir="/data/segmentation/big/mixed/custom/dataset_$number" \
    --ids_name="/data/ids/segmentation/big/mixed/exp_02/train_$number.tsv"
done