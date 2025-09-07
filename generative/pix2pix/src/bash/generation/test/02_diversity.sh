#!/bin/bash

python ./src/code/test.py \
  --name 01 \
  --dataroot /data/generation \
  --model pix2pix \
  --results_dir /models/results \
  --checkpoints_dir /models \
  --preprocess none \
  --output_nc 1 \
  --input_nc 1 \
  --netG unet_128 \
  --epoch 200 \
  --eval \
  --gen_type diversity \
  --direction BtoA