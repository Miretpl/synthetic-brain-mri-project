#!/bin/bash

python /workspace/src/python/testing/compute_msssim_reconstruction.py \
  --seed=42 \
  --real_dir="/data/generation" \
  --fake_dir="/generation/controlnet/results/reconstruction" \
  --test_ids="/data/ids/raw/test.tsv" \
  --batch_size=8 \
  --num_workers=4 \
  --model="ControlNet" \
  --access_mode="a"