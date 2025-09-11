#!/bin/bash

python /workspace/src/python/testing/compute_fid.py \
  --seed=42 \
  --sample_dir="/generation/custom/results/reconstruction" \
  --test_ids="/data/ids/raw/test.tsv" \
  --batch_size=8 \
  --num_workers=4 \
  --model="OurModel" \
  --access_mode="a"

python /workspace/src/python/testing/compute_fid.py \
  --seed=42 \
  --sample_dir="/generation/controlnet/results/reconstruction" \
  --test_ids="/data/ids/raw/test.tsv" \
  --batch_size=8 \
  --num_workers=4 \
  --model="ControlNet" \
  --access_mode="a"

python /workspace/src/python/testing/compute_fid.py \
  --seed=42 \
  --sample_dir="/generation/spade/results/reconstruction" \
  --test_ids="/data/ids/raw/test.tsv" \
  --batch_size=8 \
  --num_workers=4 \
  --model="SPADE" \
  --access_mode="a"

python /workspace/src/python/testing/compute_fid.py \
  --seed=42 \
  --sample_dir="/generation/pix2pix/results/reconstruction" \
  --test_ids="/data/ids/raw/test.tsv" \
  --batch_size=8 \
  --num_workers=4 \
  --model="Pix2Pix" \
  --access_mode="a"