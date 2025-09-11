#!/bin/bash

python /workspace/src/python/testing/compute_msssim_sample.py \
  --sample_dir="/generation/custom/results/diversity/01045" \
  --model="OurModel"

python /workspace/src/python/testing/compute_msssim_sample.py \
  --sample_dir="/generation/controlnet/results/diversity/01045" \
  --model="ControlNet"

python /workspace/src/python/testing/compute_msssim_sample.py \
  --sample_dir="/generation/spade/results/diversity/01045" \
  --model="SPADE"

python /workspace/src/python/testing/compute_msssim_sample.py \
  --sample_dir="/generation/pix2pix/results/diversity/01045" \
  --model="Pix2Pix"