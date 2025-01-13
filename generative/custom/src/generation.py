import argparse
from os import listdir
from os.path import join

import torch

from utils.data.generation import generate_img, generate_img_from_one_img
from utils.data.dataset import get_result_datasets
from accelerate import Accelerator
from accelerate.utils import PrecisionType
from utils.experiment.pipeline import create_pipeline
from base import config, model, noise_scheduler


parser = argparse.ArgumentParser()
parser.add_argument("--run_id", type=int, help="Run ID")
parser.add_argument("--img_to_gen_per_seg_map", type=int, default=1, help="")
parser.add_argument("--output_dir", help="Path to output directory")
args = parser.parse_args()

accelerator = Accelerator(
    mixed_precision=PrecisionType.FP16,
    gradient_accumulation_steps=config.gradient_accumulation_steps
)

model_dir_abs_path = join(config.results_dir_root, f'{args.run_id:02d}')
latest_model_trail = max(int(model_trail.split('_')[1]) for model_trail in listdir(model_dir_abs_path))

model = create_pipeline(
    model=accelerator.unwrap_model(model),
    noise_scheduler=noise_scheduler
).from_pretrained(
    pretrained_model_name_or_path=str(join(model_dir_abs_path, f'epoch_{latest_model_trail:04d}', 'model')),
).unet

model.eval()
model.enable_xformers_memory_efficient_attention()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_loader = get_result_datasets(config=config, img_per_seg_map=args.img_to_gen_per_seg_map)

model, data_loader = accelerator.prepare(model, data_loader, device_placement=[device, device])
pipeline = create_pipeline(
    model=accelerator.unwrap_model(model),
    noise_scheduler=noise_scheduler
)

if len(data_loader) == 1:
    generate_img_from_one_img(
        pipeline=pipeline,
        seed=config.seed,
        data_loader=data_loader,
        img_per_seg=args.img_to_gen_per_seg_map,
        output_dir=args.output_dir,
        batch_size=config.gen_batch_size
    )
else:
    generate_img(pipeline=pipeline, seed=config.seed, data_loader=data_loader, output_dir=args.output_dir)
