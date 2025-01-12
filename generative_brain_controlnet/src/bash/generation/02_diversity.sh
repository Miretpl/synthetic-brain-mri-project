#!/usr/bin/bash

python /workspace/src/python/generation/dataset.py \
  --output_dir="/results/diversity" \
  --stage1_path="/project/outputs/runs/final_models/autoencoder.pth" \
  --diffusion_path="/project/outputs/runs/final_models/diffusion_model.pth" \
  --controlnet_path="/project/outputs/runs/final_models/controlnet_model.pth" \
  --stage1_config_file_path="/config/stage1/aekl_v0.yaml" \
  --diffusion_config_file_path="/config/ldm/ldm_v0.yaml" \
  --controlnet_config_file_path="/config/controlnet/controlnet_v0.yaml" \
  --controlnet_scale=1.0 \
  --guidance_scale=7.0 \
  --x_size=20 \
  --y_size=28 \
  --num_workers=8 \
  --scale_factor=0.3 \
  --img_to_gen_per_seg_map=1000