#!/usr/bin/bash

python /workspace/src/python/testing/compute_fid.py \
  --seed=42 \
  --sample_dir="/generation/controlnet/results/reconstruction" \
  --test_ids="/data/ids/raw/test.tsv" \
  --batch_size=8 \
  --num_workers=4