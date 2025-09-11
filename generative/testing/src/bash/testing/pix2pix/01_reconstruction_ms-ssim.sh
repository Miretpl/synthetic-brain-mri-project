#!/bin/bash

python /workspace/src/python/testing/compute_msssim_reconstruction.py \
  --seed=42 \
  --real_dir="/data/generation" \
  --fake_dir="/generation/pix2pix/results/reconstruction" \
  --test_ids="/data/ids/raw/test.tsv" \
  --batch_size=8 \
  --num_workers=4 \
  --model="Pix2Pix" \
  --access_mode="a"