import argparse
from collections import defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np


NUMBER_OF_IMAGES_PER_PATIENT = 6
MIN_NUMBER_OF_REAL_IMAGES_PER_PATIENT = 2


def __get_data_dict(path: str) -> list:
    return [
        {'flair': row['flair'], 'seg': row['seg'], 'status': row['status']}
        for _, row in pd.read_csv(path, sep='\t').iterrows()
    ]


def __seg_data_per_patient(data: list) -> defaultdict:
    result = defaultdict(list)

    for img_set in data:
        result[img_set['flair'].split('/')[0]].append({
            'flair': img_set['flair'],
            'seg': img_set['seg'],
            'status': img_set['status']
        })

    return result


def __get_patient_data_row(data: list, idx: int, is_real: bool, gen_idx: Optional[int] = None) -> dict:
    return {
        'flair': data[idx]['flair'] if gen_idx is None else data[idx]['flair'].replace('.png', f'_{gen_idx}.png'),
        'seg': data[idx]['seg'],
        'status': data[idx]['status'],
        'is_real': is_real
    }


def __generate_raw_gen_datalist(data: defaultdict, quantity_of_gen_img: int = 0) -> pd.DataFrame:
    result = []
    remaining = quantity_of_gen_img - (NUMBER_OF_IMAGES_PER_PATIENT - MIN_NUMBER_OF_REAL_IMAGES_PER_PATIENT)

    for _, p_data in data.items():
        idxes = np.random.choice(
            a=np.array(range(len(p_data))),
            size=max(NUMBER_OF_IMAGES_PER_PATIENT - quantity_of_gen_img, MIN_NUMBER_OF_REAL_IMAGES_PER_PATIENT),
            replace=False
        )

        for idx in idxes:
            result.append(__get_patient_data_row(data=p_data, idx=idx, is_real=True))

        for idx in (x for x in range(NUMBER_OF_IMAGES_PER_PATIENT) if x not in idxes):
            result.append(__get_patient_data_row(data=p_data, idx=idx, is_real=False))

        if remaining > 0:
            for idx in np.random.choice(a=np.array(range(len(p_data))), size=remaining, replace=False):
                result.append(__get_patient_data_row(data=p_data, idx=idx, is_real=False, gen_idx=1))

    return pd.DataFrame(result)


def __show_stats(dataset: pd.DataFrame, only_real: bool) -> None:
    status = datalist.groupby('status').size()
    is_real = datalist.groupby('is_real').size()

    if only_real:
        print(
            f'Train stats: total {dataset.shape[0]} | '
            f'Healthy percentage {round(status.healthy / (status.healthy + status.unhealthy) * 100, 2)}%'
        )
    else:
        print(
            f'Train stats: total {dataset.shape[0]} | '
            f'Healthy percentage {round(status.healthy / (status.healthy + status.unhealthy) * 100, 2)}% | '
            f'Fake percentage {round(is_real[False] / (is_real[True] + is_real[False]) * 100, 2)}%'
        )


parser = argparse.ArgumentParser()

parser.add_argument("--src_ids", help="Path to source ids tsv file.")
parser.add_argument("--output_dir", help="Path to output directory where new ids files will be places.")
parser.add_argument("--real_size", type=int, default=0, help="Number of real samples in data list.")
parser.add_argument(
    "--only_real",
    default=True,
    action=argparse.BooleanOptionalAction,
    help="If true the datalist will only contain real samples. True -> --only_real, False -> --no-only_real"
)
parser.add_argument(
    "--number_of_data_lists", help="Number of data in each data list in \"20, 40\" format."
)
parser.add_argument(
    "--number_of_fake_data_per_patient",
    type=int,
    default=NUMBER_OF_IMAGES_PER_PATIENT // 2,
    help="Number of fake images per patient."
)

args = parser.parse_args()

output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)

train = __get_data_dict(path=args.src_ids)
train = __seg_data_per_patient(train)

for gen_img_number in (int(x) for x in args.number_of_data_lists.replace(' ', '').split(',')):
    print(f'Using {gen_img_number} generated images per patient')

    if args.only_real:
        datalist = __generate_raw_gen_datalist(data=train, quantity_of_gen_img=0)
    else:
        datalist = __generate_raw_gen_datalist(data=train, quantity_of_gen_img=args.number_of_fake_data_per_patient)

    if len(datalist) != gen_img_number:
        real_healthy_idx = datalist[(datalist['is_real'] == True) & (datalist['status'] == 'healthy')].index
        real_unhealthy_idx = datalist[(datalist['is_real'] == True) & (datalist['status'] == 'unhealthy')].index

        if args.only_real:
            size = gen_img_number // 2
        else:
            size = args.real_size // 2

        real_healthy = datalist.loc[np.random.choice(real_healthy_idx, size=size, replace=False)]
        real_unhealthy = datalist.loc[np.random.choice(real_unhealthy_idx, size=size, replace=False)]

        if args.only_real:
            to_concat = [real_healthy, real_unhealthy]
        else:
            fake_healthy_idx = datalist[(datalist['is_real'] == False) & (datalist['status'] == 'healthy')].index
            fake_unhealthy_idx = datalist[(datalist['is_real'] == False) & (datalist['status'] == 'unhealthy')].index

            size = (gen_img_number - args.real_size) // 2

            fake_healthy = datalist.loc[np.random.choice(fake_healthy_idx, size=size, replace=False)]
            fake_unhealthy = datalist.loc[np.random.choice(fake_unhealthy_idx, size=size, replace=False)]

            to_concat = [real_healthy, real_unhealthy, fake_healthy, fake_unhealthy]

        datalist = pd.concat(to_concat, axis=0, ignore_index=True).sample(frac=1).reset_index(drop=True)

    __show_stats(dataset=datalist, only_real=args.only_real)

    filename = args.src_ids.split("/")[-1]
    if args.number_of_data_lists.find(',') > -1:
        filename = filename.replace(".tsv", f"_{gen_img_number}.tsv")

    datalist.to_csv(
        f'{str(output_dir)}/{filename}',
        index=False,
        sep="\t"
    )
