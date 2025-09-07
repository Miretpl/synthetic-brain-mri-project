from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from shutil import copy2

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src_dir", help="Data path where source data is located.")
    parser.add_argument("--dst_dir", help="Data path where source data should be copied.")
    parser.add_argument("--ids_path", help="Location of ids tsv file.")
    parser.add_argument("--num_workers", type=int, help="")

    return parser.parse_args()


def main(args):
    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(args.ids_path, sep='\t')

    data_dicts = [
        {
            'flair': row['flair'],
            'seg': row['seg']
        }
        for index, row in df.iterrows()
    ]

    print(f'Found {len(data_dicts)} subjects')

    data_list = sorted(data_dicts, key=lambda x: x['flair'])

    for batch in tqdm(data_list, desc='Coping data'):
        src_path = f'{args.src_dir}/{batch["seg"]}'
        dst_path = f'{args.dst_dir}/{batch["seg"]}'

        copy2(src_path, dst_path)


if __name__ == "__main__":
    main(parse_args())
