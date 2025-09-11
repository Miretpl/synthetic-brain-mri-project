"""Utility functions for testing."""
from __future__ import annotations

import collections
from json import dumps, load
from pathlib import Path

import pandas as pd
from monai import transforms
from monai.data import Dataset
from torch.utils.data import DataLoader


def __get_transform_for_fid() -> transforms.Compose:
    return transforms.Compose([
        transforms.LoadImaged(keys=["real"]),
        transforms.EnsureChannelFirstd(keys=["real"]),
        transforms.Rotate90d(keys=["real"], k=-1, spatial_axes=(0, 1)),
        transforms.Flipd(keys=["real"], spatial_axis=1),
        transforms.ScaleIntensityRanged(
            keys=["real"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True
        ),
        transforms.ToTensord(keys=["real"]),
    ])


def __get_transform_for_msssim() -> transforms.Compose:
    return transforms.Compose([
        transforms.LoadImaged(keys=["real", "fake"]),
        transforms.EnsureChannelFirstd(keys=["real", "fake"]),
        transforms.Rotate90d(keys=["real", "fake"], k=-1, spatial_axes=(0, 1)),
        transforms.Flipd(keys=["real", "fake"], spatial_axis=1),
        transforms.ScaleIntensityRanged(
            keys=["real", "fake"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True
        ),
        transforms.ToTensord(keys=["real", "fake"]),
    ])


def __get_dataset_row(row: pd.Series, real_root: str, fake_root: str | None = None) -> dict:
    if fake_root is None:
        return {
            "real": f"{real_root}/{row['flair']}",
        }

    return {
        "real": f"{real_root}/{row['flair']}",
        "fake": f"{fake_root}/{row['flair']}"
    }


def __update(d: dict, u: dict) -> dict:
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = __update(d.get(k, {}), v)
        else:
            d[k] = v

    return d


def save_metadata(data: dict, access_mode: str) -> None:
    meta_dir = '/data/metadata/generation'
    Path(meta_dir).mkdir(parents=True, exist_ok=True)

    content = {}
    if "a" in access_mode:
        with open(f"{meta_dir}/metrics.json") as f:
            content = load(f)

    content = __update(content, data)

    with open(f'{meta_dir}/metrics.json', "w") as f:
        f.write(dumps(content, indent=4))


def get_test_dataloader(
    batch_size: int,
    test_ids: str,
    real_root: str,
    num_workers: int = 8,
    fake_root: str | None = None,
    upper_limit: int | None = None,
):
    if fake_root is None:
        test_transforms = __get_transform_for_fid()
    else:
        test_transforms = __get_transform_for_msssim()

    test_dicts = get_datalist(ids_path=test_ids, fake_root=fake_root, real_root=real_root, upper_limit=upper_limit)
    test_ds = Dataset(data=test_dicts, transform=test_transforms)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=False,
        persistent_workers=True,
    )

    return test_loader


def get_datalist(
    ids_path: str,
    real_root: str,
    fake_root: str | None = None,
    upper_limit: int | None = None,
):
    """Get data dicts for data loaders."""
    df = pd.read_csv(ids_path, sep="\t")

    if upper_limit is not None:
        df = df[:upper_limit]

    data_dicts = []
    for index, row in df.iterrows():
        data_dicts.append(__get_dataset_row(
            row=row,
            real_root=real_root,
            fake_root=fake_root
        ))

    print(f"Found {len(data_dicts)} subjects.")
    return data_dicts
