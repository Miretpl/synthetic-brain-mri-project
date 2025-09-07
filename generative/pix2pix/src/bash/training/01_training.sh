#!/bin/bash

python ./src/code/train.py \
  --name 01 \
  --dataroot /data/generation \
  --ids_path /data/ids/raw/train.tsv \
  --model pix2pix \
  --direction BtoA \
  --output_nc 1 \
  --input_nc 1 \
  --preprocess none \
  --crop 224 \
  --no_html \
  --checkpoints_dir /models \
  --netG unet_128