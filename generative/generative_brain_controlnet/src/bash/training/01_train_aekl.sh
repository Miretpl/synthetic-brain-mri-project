#!/usr/bin/bash

python /workspace/src/python/training/train_aekl.py \
  --seed=42 \
  --run_dir="aekl" \
  --training_ids="/data/ids/raw/train.tsv" \
  --validation_ids="/data/ids/raw/validation.tsv" \
  --config_file="/config/stage1/aekl_v0.yaml" \
  --batch_size=8 \
  --n_epochs=100 \
  --adv_start=10 \
  --eval_freq=5 \
  --num_workers=2 \
  --experiment="AEKL" \
  --data_root_path="/data/generation"