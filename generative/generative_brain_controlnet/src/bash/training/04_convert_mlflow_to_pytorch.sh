#!/bin/bash

stage1_mlrun_id="129642340378682336/38b64d11530d4eac80e48f718bb2333a"
ldm_mlrun_id="122096036595437994/1262545b2fa845c380f0f8d77ba6c90b"
controlnet_mlrun_id="604380746799830390/c4c5815efd4d4994a9db212e47558ea9"

python /workspace/src/python/training/convert_mlflow_to_pytorch.py \
  --stage1_mlflow_path="/project/mlruns/${stage1_mlrun_id}/artifacts/final_model" \
  --diffusion_mlflow_path="/project/mlruns/${ldm_mlrun_id}/artifacts/final_model" \
  --controlnet_mlflow_path="/project/mlruns/${controlnet_mlrun_id}/artifacts/final_model" \
  --output_dir="/project/outputs/runs/final_models"