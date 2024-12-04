"""
Script assigns patients to given set group: train, validation or test.
"""
from json import dumps
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def create_datalist(set_paths: list[Path], desc: str) -> 'pd.DataFrame':
    data_list = []

    for sub_dir in tqdm(set_paths, desc=desc):
        flair_img_paths = sorted(list(sub_dir.glob('**/*flair*.png')))

        for flair_img_path in flair_img_paths:
            flair_rel_path = flair_img_path.relative_to(sub_dir.parent)
            seg_rel_path = flair_rel_path.parent / (flair_rel_path.name.replace('flair', 'seg'))

            data_list.append({
                "flair": flair_rel_path,
                "seg": seg_rel_path,
                "status": flair_rel_path.name.split('_')[2][:-4]
            })

    return pd.DataFrame(data_list)


def calculate_statistics(data: 'pd.DataFrame') -> dict:
    stats = data_df.groupby('status').size()
    return {
        'total': data.shape[0],
        'healthy_quantity': int(stats.healthy),
        'healthy_percentage': float(round(stats.healthy / (stats.healthy + stats.unhealthy) * 100, 2))
    }


ids_dir = Path('/data/ids/raw')
ids_dir.mkdir(parents=True, exist_ok=True)

data_dir = Path('/data/raw/extracted')
sub_dirs = sorted([x for x in data_dir.iterdir() if x.is_dir()])

whole_dataset_quantity = len(sub_dirs)
train_quantity = int(whole_dataset_quantity * 0.8)
val_quantity = (whole_dataset_quantity - train_quantity) // 2

train_sub_dirs, test_sub_dirs = train_test_split(sub_dirs, train_size=train_quantity)
test_sub_dirs, val_sub_dirs = train_test_split(test_sub_dirs, train_size=val_quantity)

data_df = create_datalist(set_paths=train_sub_dirs, desc='Train set')
train_stats = calculate_statistics(data_df)
data_df.to_csv(ids_dir / 'train.tsv', index=False, sep='\t')

data_df = create_datalist(set_paths=val_sub_dirs, desc='Validation set')
val_stats = calculate_statistics(data_df)
data_df.to_csv(ids_dir / 'validation.tsv', index=False, sep='\t')

data_df = create_datalist(set_paths=test_sub_dirs, desc='Test set')
test_stats = calculate_statistics(data_df)
data_df.to_csv(ids_dir / 'test.tsv', index=False, sep='\t')

metadata = {
    'patients': {
        'quantity': whole_dataset_quantity
    },
    'train': train_stats,
    'validation': val_stats,
    'test': test_stats
}

meta_dir = '/data/metadata/dataset'
Path(meta_dir).mkdir(parents=True, exist_ok=True)

with open(f'{meta_dir}/02_split_dataset.json', 'w') as f:
    f.write(dumps(metadata, indent=4))

print(dumps(metadata, indent=4))
