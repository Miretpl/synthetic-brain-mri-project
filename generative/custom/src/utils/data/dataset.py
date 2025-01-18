from os.path import join
from typing import Optional

import pandas as pd
from monai import transforms
from monai.data import Dataset
from monai.transforms import Compose
from torch.utils.data import DataLoader

from utils.config.config import ExperimentConfig


def __get_datalist(
        ids_path: str, filename: str, data_path: Optional[str] = None, diversity: bool = False
) -> list[dict]:
    if diversity:
        data_dicts = [{
            'flair': f'{data_path}/01045/03_flair_unhealthy_{idx}.png',  # This will be used as path for saving image
            'seg': f'{data_path}/01045/03_seg_unhealthy.png'
        } for idx in range(1000)]
    else:
        df = pd.read_csv(join(ids_path, filename), sep='\t')

        if data_path is not None:
            data_dicts = [
                {
                    'flair': join(data_path, row['flair']),
                    'seg': join(data_path, row['seg'])
                }
                for index, row in df.iterrows()
            ]
        else:
            data_dicts = [
                {
                    'flair': row['flair'],
                    'seg': row['seg']
                }
                for index, row in df.iterrows()
            ]

    print(f'Found {len(data_dicts)} subjects')
    return data_dicts


def __get_transforms() -> tuple[Compose, Compose, Compose]:
    common = [
        transforms.LoadImaged(keys=['flair', 'seg']),
        transforms.EnsureChannelFirstd(keys=['flair', 'seg']),
        transforms.Rotate90d(keys=['flair', 'seg'], k=-1, spatial_axes=(0, 1)),
        transforms.Flipd(keys=['flair', 'seg'], spatial_axis=1),
        transforms.ScaleIntensityRanged(
            keys=['flair'], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True
        )
    ]

    return transforms.Compose(common), transforms.Compose(common), transforms.Compose([
        *common,
        transforms.RandFlipd(keys=["flair", "seg"], prob=0.5, spatial_axis=0),
        transforms.RandAffined(
            keys=["flair", "seg"],
            translate_range=(-2, 2),
            scale_range=(-0.01, 0.01),
            spatial_size=[160, 224],
            prob=0.25,
        )
    ])


def get_datasets(config: ExperimentConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    test_transforms, val_transforms, train_transforms = __get_transforms()

    train_dicts = __get_datalist(
        ids_path=config.dataset_ids_path, data_path=config.dataset_root_path, filename=config.training_ids
    )
    val_dicts = __get_datalist(
        ids_path=config.dataset_ids_path, data_path=config.dataset_root_path, filename=config.validation_ids
    )
    test_dicts = __get_datalist(
        ids_path=config.dataset_ids_path, data_path=config.dataset_root_path, filename=config.test_ids
    )

    train_ds = Dataset(data=train_dicts, transform=train_transforms)
    val_ds = Dataset(data=val_dicts, transform=val_transforms)
    test_ds = Dataset(data=test_dicts, transform=test_transforms)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=False,
        pin_memory=False,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.val_batch_size,
        num_workers=config.num_workers,
        drop_last=False,
        pin_memory=False,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config.test_batch_size,
        num_workers=config.num_workers,
        drop_last=False,
        pin_memory=False,
        persistent_workers=True
    )

    return train_loader, val_loader, test_loader


def get_result_datasets(config: ExperimentConfig, ids: str, diversity: bool) -> DataLoader:
    transform = transforms.Compose([
        transforms.LoadImaged(keys=['seg']),
        transforms.EnsureChannelFirstd(keys=['seg']),
        transforms.Rotate90d(keys=['seg'], k=-1, spatial_axes=(0, 1)),
        transforms.Flipd(keys=['seg'], spatial_axis=1)
    ])

    ds = Dataset(
        data=__get_datalist(
            ids_path=config.dataset_ids_path,
            data_path=config.dataset_root_path,
            filename=ids,
            diversity=diversity
        ),
        transform=transform
    )

    return DataLoader(
        ds,
        batch_size=config.gen_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=False,
        pin_memory=False,
        persistent_workers=True
    )


def get_raw_dataloader(config: ExperimentConfig, ids: str) -> DataLoader:
    ds = Dataset(
        data=__get_datalist(
            ids_path=config.dataset_ids_path,
            filename=ids
        )
    )

    return DataLoader(
        ds,
        batch_size=1,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=False,
        pin_memory=False,
        persistent_workers=True
    )
