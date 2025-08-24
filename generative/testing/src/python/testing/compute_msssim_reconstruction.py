""" Script to compute the MS-SSIM score of the reconstructions of the Autoencoder.

Here we compute the MS-SSIM score between the images of the test set of the MIMIC-CXR dataset and the reconstructions
created byt the AutoencoderKL.
"""
import argparse

import pandas as pd
import torch
from generative.metrics import MultiScaleSSIMMetric
from generative.networks.nets import AutoencoderKL
from monai.config import print_config
from monai.utils import set_determinism
from omegaconf import OmegaConf
from tqdm import tqdm
from util import get_test_dataloader


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2, help="Random seed to use.")
    parser.add_argument("--real_dir", help="Location of real data.")
    parser.add_argument("--fake_dir", help="Location of generated data.")
    parser.add_argument("--test_ids", help="Location of file with test ids.")
    parser.add_argument("--batch_size", type=int, default=256, help="Testing batch size.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of loader workers")

    args = parser.parse_args()
    return args


def main(args):
    set_determinism(seed=args.seed)
    print_config()

    print("Getting data...")
    test_loader = get_test_dataloader(
        batch_size=args.batch_size,
        test_ids=args.test_ids,
        fake_root=args.fake_dir,
        real_root=args.real_dir,
        num_workers=args.num_workers
    )

    device = torch.device("cuda")
    ms_ssim = MultiScaleSSIMMetric(spatial_dims=2, data_range=1.0, kernel_size=7)

    print("Computing MS-SSIM...")
    ms_ssim_list = []
    for batch in tqdm(test_loader):
        ms_ssim_list.append(ms_ssim(
            y_pred=batch["real"].to(device),
            y=batch["fake"].to(device)
        ))

    ms_ssim_list = torch.cat(ms_ssim_list, dim=0)
    print(f"Mean MS-SSIM: {ms_ssim_list.mean():.6f}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
