#!/bin/bash

python /workspace/src/python/testing/compute_msssim_sample.py \
  --seed=42 \
  --sample_dir="/generation/custom/results/diversity/01045" \
  --num_workers=4 \
  --model="OurModel" \
  --access_mode="a"

python /workspace/src/python/testing/compute_msssim_sample.py \
  --seed=42 \
  --sample_dir="/generation/controlnet/results/diversity/01045" \
  --num_workers=4 \
  --model="ControlNet" \
  --access_mode="a"

python /workspace/src/python/testing/compute_msssim_sample.py \
  --seed=42 \
  --sample_dir="/generation/spade/results/diversity/01045" \
  --num_workers=4 \
  --model="SPADE" \
  --access_mode="a"

python /workspace/src/python/testing/compute_msssim_sample.py \
  --seed=42 \
  --sample_dir="/generation/pix2pix/results/diversity/01045" \
  --num_workers=4 \
  --model="Pix2Pix" \
  --access_mode="a"