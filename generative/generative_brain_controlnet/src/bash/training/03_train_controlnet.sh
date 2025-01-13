#!/usr/bin/bash

stage1_mlrun_id="129642340378682336/38b64d11530d4eac80e48f718bb2333a"
ldm_mlrun_id="122096036595437994/1262545b2fa845c380f0f8d77ba6c90b"

python /workspace/src/python/training/train_controlnet.py \
  --seed=42 \
  --run_dir="controlnet" \
  --training_ids="/data/ids/raw/train.tsv" \
  --validation_ids="/data/ids/raw/validation.tsv" \
  --stage1_uri="/project/mlruns/${stage1_mlrun_id}/artifacts/final_model" \
  --ddpm_uri="/project/mlruns/${ldm_mlrun_id}/artifacts/final_model" \
  --config_file="/config/controlnet/controlnet_v0.yaml" \
  --scale_factor=0.3 \
  --batch_size=8 \
  --n_epochs=150 \
  --eval_freq=5 \
  --num_workers=4 \
  --experiment="CONTROLNET" \
  --data_root_path="/data/generation"