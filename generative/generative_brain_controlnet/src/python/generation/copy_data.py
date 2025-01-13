from __future__ import annotations

import argparse
from pathlib import Path
from shutil import copy2

from tqdm import tqdm
from util import get_raw_dataloader


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src_dir", help="Data path where source data is located.")
    parser.add_argument("--dst_dir", help="Data path where source data should be copied.")
    parser.add_argument("--ids", help="Path to ids tsv file.")
    parser.add_argument("--num_workers", type=int, help="")

    return parser.parse_args()


def main(args):
    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(exist_ok=True, parents=True)

    data_loader = get_raw_dataloader(
        batch_size=1,
        ids=args.ids,
        num_workers=args.num_workers
    )

    for batch in tqdm(data_loader, desc='Coping data'):
        src_path = f'{args.src_dir}/{batch["seg"][0]}'
        dst_path = f'{args.dst_dir}/{batch["seg"][0]}'

        copy2(src_path, dst_path)


if __name__ == "__main__":
    main(parse_args())
