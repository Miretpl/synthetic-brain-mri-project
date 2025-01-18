from os.path import join
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL.Image import fromarray
from tqdm import tqdm

from utils.experiment.custom_ddpm_pipeline import CustomDDPMPipeline


def generate_and_save_img(pipeline: CustomDDPMPipeline, seed: int, data_loader: DataLoader, output_dir: str) -> None:
    tqdm_ = tqdm(enumerate(data_loader), total=len(data_loader), desc='Generating images')
    for idx, batch in tqdm_:
        images = pipeline(
            batch_size=batch['seg'].shape[0],
            control=batch['seg'],
            generator=torch.manual_seed(seed + idx)
        ).images

        result = np.array([
            (np.interp(img, (img.min(), img.max()), (0, +1)) * 255).astype(np.uint8)
            for img in images
        ])

        for i, img in enumerate(result):
            path = join(output_dir, '/'.join(batch['flair'][i].split('/')[-2:]))
            Path(str(join(output_dir, batch['flair'][i].split('/')[-2]))).mkdir(parents=True, exist_ok=True)
            fromarray(img[0]).save(path)
