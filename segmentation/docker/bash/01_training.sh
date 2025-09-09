#!/bin/bash

# Train segmentation model - BIG dataset, real, mixed, synthetic
python ./code/train.py \
  --output_dir="/models/exp_01/real/runs" \
  --train_ids="/data/ids/raw/train.tsv" \
  --real_data_path="/data/raw/extracted"

python ./code/train.py \
  --output_dir="/models/exp_01/mixed/custom/runs" \
  --train_ids="/data/ids/segmentation/big/mixed/exp_01/train.tsv" \
  --real_data_path="/data/raw/extracted" \
  --fake_data_path="/data/segmentation/custom"

python ./code/train.py \
  --output_dir="/models/exp_01/mixed/controlnet/runs" \
  --train_ids="/data/ids/segmentation/big/mixed/exp_01/train.tsv" \
  --real_data_path="/data/raw/extracted" \
  --fake_data_path="/data/segmentation/controlnet"

python ./code/train.py \
  --output_dir="/models/exp_01/mixed/spade/runs" \
  --train_ids="/data/ids/segmentation/big/mixed/exp_01/train.tsv" \
  --real_data_path="/data/raw/extracted" \
  --fake_data_path="/data/segmentation/spade"

python ./code/train.py \
  --output_dir="/models/exp_01/mixed/pix2pix/runs" \
  --train_ids="/data/ids/segmentation/big/mixed/exp_01/train.tsv" \
  --real_data_path="/data/raw/extracted" \
  --fake_data_path="/data/segmentation/pix2pix"

python ./code/train.py \
  --output_dir="/models/exp_01/synthetic/custom/runs" \
  --train_ids="/data/ids/raw/train.tsv" \
  --real_data_path="/data/segmentation/custom"

python ./code/train.py \
  --output_dir="/models/exp_01/synthetic/controlnet/runs" \
  --train_ids="/data/ids/raw/train.tsv" \
  --real_data_path="/data/segmentation/controlnet"

python ./code/train.py \
  --output_dir="/models/exp_01/synthetic/spade/runs" \
  --train_ids="/data/ids/raw/train.tsv" \
  --real_data_path="/data/segmentation/spade"

python ./code/train.py \
  --output_dir="/models/exp_01/synthetic/pix2pix/runs" \
  --train_ids="/data/ids/raw/train.tsv" \
  --real_data_path="/data/segmentation/pix2pix"

# Train segmentation model - BIG mixed datasets
#dataset_quantity=( 4000 6000 )
#for number in "${dataset_quantity[@]}"
#do
#  echo "Training model on big mixed datasets - custom - $number"
#  python ./code/train.py \
#    --output_dir="/models/exp_02/mixed/extra_exp/custom/ds_$number/runs" \
#    --train_ids="/data/ids/segmentation/big/mixed/exp_02/train_$number.tsv" \
#    --real_data_path="/data/raw/extracted" \
#    --fake_data_path="/data/segmentation/custom"
#
#  echo "Training model on big mixed datasets - controlnet - $number"
#  python ./code/train.py \
#    --output_dir="/models/exp_02/mixed/extra_exp/controlnet/ds_$number/runs" \
#    --train_ids="/data/ids/segmentation/big/mixed/exp_02/train_$number.tsv" \
#    --real_data_path="/data/raw/extracted" \
#    --fake_data_path="/data/segmentation/controlnet"
#done

#dataset_quantity=( 8000 10000 12000 )
#for number in "${dataset_quantity[@]}"
#do
#  echo "Training model on big mixed datasets - Custom - $number"
#  python ./code/train.py \
#    --output_dir="/models/exp_02/mixed/extra_exp/custom/ds_$number/runs" \
#    --train_ids="/data/ids/segmentation/big/mixed/exp_02/train_$number.tsv" \
#    --real_data_path="/data/raw/extracted" \
#    --fake_data_path="/data/segmentation/big/mixed/custom/dataset_$number"
#
#  echo "Training model on big mixed datasets - ControlNet - $number"
#  python ./code/train.py \
#    --output_dir="/models/exp_02/mixed/extra_exp/controlnet/ds_$number/runs" \
#    --train_ids="/data/ids/segmentation/big/mixed/exp_02/train_$number.tsv" \
#    --real_data_path="/data/raw/extracted" \
#    --fake_data_path="/data/segmentation/big/mixed/controlnet/dataset_$number"
#
#  echo "Training model on big mixed datasets - SPADE - $number"
#  python ./code/train.py \
#    --output_dir="/models/exp_02/mixed/extra_exp/spade/ds_$number/runs" \
#    --train_ids="/data/ids/segmentation/big/mixed/exp_02/train_$number.tsv" \
#    --real_data_path="/data/raw/extracted" \
#    --fake_data_path="/data/segmentation/big/mixed/spade/dataset_$number"
#
#  echo "Training model on big mixed datasets - Pix2Pix - $number"
#  python ./code/train.py \
#    --output_dir="/models/exp_02/mixed/extra_exp/pix2pix/ds_$number/runs" \
#    --train_ids="/data/ids/segmentation/big/mixed/exp_02/train_$number.tsv" \
#    --real_data_path="/data/raw/extracted" \
#    --fake_data_path="/data/segmentation/big/mixed/pix2pix/dataset_$number"
#done

# Train segmentation model - SMALL REAL datasets
dataset_quantity=( 20 40 60 80 100 200 400 600 800 )

for number in "${dataset_quantity[@]}"
do
  echo "Training model on small real datasets - $number"
  python ./code/train.py \
    --output_dir="/models/exp_02/real/ds_$number/runs" \
    --train_ids="/data/ids/segmentation/small/real/train_$number.tsv" \
    --real_data_path="/data/raw/extracted"
done

# Train segmentation model - SMALL MIXED datasets
dataset_quantity=( 20 40 60 80 100 200 400 600 800 )

for number in "${dataset_quantity[@]}"
do
  echo "Training model on small real datasets - $number - Custom model"

  python ./code/train.py \
    --output_dir="/models/exp_02/mixed/custom/ds_$number/runs" \
    --train_ids="/data/ids/segmentation/small/mixed/train_$number.tsv" \
    --real_data_path="/data/raw/extracted" \
    --fake_data_path="/data/segmentation/custom"

  echo "Training model on small real datasets - $number - ControlNet model"
  python ./code/train.py \
    --output_dir="/models/exp_02/mixed/controlnet/ds_$number/runs" \
    --train_ids="/data/ids/segmentation/small/mixed/train_$number.tsv" \
    --real_data_path="/data/raw/extracted" \
    --fake_data_path="/data/segmentation/controlnet"

  echo "Training model on small real datasets - $number - SPADE model"
  python ./code/train.py \
    --output_dir="/models/exp_02/mixed/spade/ds_$number/runs" \
    --train_ids="/data/ids/segmentation/small/mixed/train_$number.tsv" \
    --real_data_path="/data/raw/extracted" \
    --fake_data_path="/data/segmentation/spade"

    echo "Training model on small mixed datasets - $number - Pix2Pix model"
    python ./code/train.py \
      --output_dir="/models/exp_02/mixed/pix2pix/ds_$number/runs" \
      --train_ids="/data/ids/segmentation/small/mixed/train_$number.tsv" \
      --real_data_path="/data/raw/extracted" \
      --fake_data_path="/data/segmentation/pix2pix"
done