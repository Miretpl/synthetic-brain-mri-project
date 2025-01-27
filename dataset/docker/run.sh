#!/usr/bin/bash

python ./code/01_extract_png_images_from_nii.py
python ./code/02_split_dataset_to_sets.py

# Generate ids files for 50/50 whole dataset
python ./code/03_generate_datalist_for_segmentation_model.py \
  --src_ids="/data/ids/raw/train.tsv" \
  --output_dir="/data/ids/segmentation/big/mixed/exp_01" \
  --real_size=3000 \
  --number_of_data_lists="6000"

# Generate ids files for big dataset experiments - mixed
python ./code/03_generate_datalist_for_segmentation_model.py \
  --src_ids="/data/ids/raw/train.tsv" \
  --output_dir="/data/ids/segmentation/big/mixed/exp_02" \
  --real_size=2000 \
  --number_of_data_lists="4000, 6000, 8000, 10000, 12000" \
  --fake_img_per_patient="3, 4, 6, 8, 10"

# Generate ids files for small dataset experiments - real
python ./code/03_generate_datalist_for_segmentation_model.py \
  --src_ids="/data/ids/raw/train.tsv" \
  --output_dir="/data/ids/segmentation/small/real" \
  --number_of_data_lists="20, 40, 60, 80, 100, 200, 400, 600, 800" \
  --fake_img_per_patient="0, 0, 0, 0, 0, 0, 0, 0, 0"

# Generate ids files for small dataset experiments - mixed
python ./code/03_generate_datalist_for_segmentation_model.py \
  --src_ids="/data/ids/raw/train.tsv" \
  --output_dir="/data/ids/segmentation/small/mixed" \
  --real_size=10 \
  --number_of_data_lists="20, 40, 60, 80, 100, 200, 400, 600, 800"