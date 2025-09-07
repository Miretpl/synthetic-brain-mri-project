#!/bin/bash

python /workspace/src/python/testing/compute_msssim_sample.py \
  --seed=42 \
  --sample_dir="/generation/pix2pix/results/diversity/01045" \
  --num_workers=4