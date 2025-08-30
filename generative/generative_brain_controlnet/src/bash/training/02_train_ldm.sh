#!/bin/bash

mlrun_id="129642340378682336/38b64d11530d4eac80e48f718bb2333a"

python /workspace/src/python/training/train_ldm.py \
  --seed=42 \
  --run_dir="ldm" \
  --training_ids="/data/ids/raw/train.tsv" \
  --validation_ids="/data/ids/raw/validation.tsv" \
  --stage1_uri="/project/mlruns/${mlrun_id}/artifacts/final_model" \
  --config_file="/config/ldm/ldm_v0.yaml" \
  --scale_factor=0.3 \
  --batch_size=8 \
  --n_epochs=150 \
  --eval_freq=5 \
  --num_workers=4 \
  --experiment="LDM" \
  --data_root_path="/data/generation"