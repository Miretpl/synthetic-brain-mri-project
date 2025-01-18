from os.path import join
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL.Image import fromarray
from tqdm import tqdm

from utils.experiment.custom_ddpm_pipeline import CustomDDPMPipeline


def generate_img(pipeline: CustomDDPMPipeline, seed: int, data_loader: DataLoader, output_dir: str) -> None:
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

        tqdm_.update(batch['seg'].shape[0])


def generate_img_from_one_img(
        pipeline: CustomDDPMPipeline, seed: int, data_loader: DataLoader, img_per_seg: int, output_dir: str,
        batch_size: int
) -> None:
    counter = 0
    for batch_idx, batch in enumerate(data_loader):
        tqdm_ = tqdm(range(img_per_seg), total=img_per_seg, desc='Generating images')
        for img_idx in tqdm_:
            repeat_val = batch_size
            if counter + batch_size > img_per_seg:
                repeat_val = img_per_seg - counter

            counter += repeat_val

            images = pipeline(
                batch_size=repeat_val,
                control=batch['seg'].repeat(repeat_val, 1, 1, 1),
                generator=torch.manual_seed(seed + img_idx)
            ).images

            result = []
            for img in images:
                result.append((np.interp(img, (img.min(), img.max()), (0, +1)) * 255).astype(np.uint8))

            result = np.array(result)

            for i, img in enumerate(result):
                idx = img_idx * batch_size + i
                path = join(
                    output_dir,
                    '/'.join(batch['flair'][0].split('/')[-2:]).replace('.png', f'_{idx}.png')
                )
                Path(str(join(output_dir, batch['flair'][0].split('/')[-2]))).mkdir(parents=True, exist_ok=True)
                print(path)
                fromarray(img[0]).save(path)

            tqdm_.update(repeat_val)
            if counter == img_per_seg:
                break
