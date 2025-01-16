#!/usr/bin/bash

python ./code/01_extract_png_images_from_nii.py
python ./code/02_split_dataset_to_sets.py

# Generate ids files for 50/50 whole dataset
python ./code/03_generate_datalist_for_segmentation_model.py \
  --src_ids="/data/ids/raw/train.tsv" \
  --output_dir="/data/ids/segmentation/big/mixed" \
  --real_size=3000 \
  --no-only_real \
  --number_of_data_lists=6000

# Generate ids files for small dataset experiments - real
python ./code/03_generate_datalist_for_segmentation_model.py \
  --src_ids="/data/ids/raw/train.tsv" \
  --output_dir="/data/ids/segmentation/small/real" \
  --only_real \
  --number_of_data_lists="20, 40, 60, 80, 100, 200, 400, 600, 800"

# Generate ids files for small dataset experiments - mixed
python ./code/03_generate_datalist_for_segmentation_model.py \
  --src_ids="/data/ids/raw/train.tsv" \
  --output_dir="/data/ids/segmentation/small/mixed" \
  --real_size=10 \
  --no-only_real \
  --number_of_data_lists="20, 40, 60, 80, 100, 200, 400, 600, 800"