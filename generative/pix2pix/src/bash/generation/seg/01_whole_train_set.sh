#!/bin/bash

python ./src/code/test.py \
  --name 01 \
  --dataroot /data/generation \
  --ids_path /data/ids/raw/train.tsv \
  --model pix2pix \
  --results_dir /data/segmentation/pix2pix \
  --checkpoints_dir /models \
  --preprocess none \
  --output_nc 1 \
  --input_nc 1 \
  --netG unet_128 \
  --epoch 200 \
  --eval \
  --direction BtoA