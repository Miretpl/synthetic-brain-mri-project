#!/bin/bash

python /workspace/src/python/testing/compute_fid.py \
  --sample_dir="/generation/custom/results/reconstruction" \
  --test_ids="/data/ids/raw/test.tsv" \
  --model="OurModel"

python /workspace/src/python/testing/compute_fid.py \
  --sample_dir="/generation/controlnet/results/reconstruction" \
  --test_ids="/data/ids/raw/test.tsv" \
  --model="ControlNet"

python /workspace/src/python/testing/compute_fid.py \
  --sample_dir="/generation/spade/results/reconstruction" \
  --test_ids="/data/ids/raw/test.tsv" \
  --model="SPADE"

python /workspace/src/python/testing/compute_fid.py \
  --sample_dir="/generation/pix2pix/results/reconstruction" \
  --test_ids="/data/ids/raw/test.tsv" \
  --model="Pix2Pix"