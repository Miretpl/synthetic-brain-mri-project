"""Utility functions for testing."""
from __future__ import annotations

from typing import Optional

import pandas as pd
from monai import transforms
from monai.data import Dataset
from torch.utils.data import DataLoader


def get_raw_dataloader(
        batch_size: int,
        ids: str,
        num_workers: int = 8
):
    dicts = get_datalist(ids_path=ids)
    ds = Dataset(data=dicts)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=False,
        persistent_workers=True,
    )


def get_test_dataloader(
    batch_size: int,
    test_ids: Optional[str],
    root_path: str,
    num_workers: int = 8,
    upper_limit: int | None = None,
):
    test_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["seg"]),
            transforms.EnsureChannelFirstd(keys=["seg"]),
            transforms.Rotate90d(keys=["seg"], k=-1, spatial_axes=(0, 1)),  # Fix flipped image read
            transforms.Flipd(keys=["seg"], spatial_axis=1),  # Fix flipped image read
            transforms.ScaleIntensityRanged(
                keys=["seg"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True
            ),
            transforms.ToTensord(keys=["seg"]),
        ]
    )

    test_dicts = get_datalist(ids_path=test_ids, root_path=root_path, upper_limit=upper_limit)
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
    ids_path: Optional[str],
    root_path: Optional[str] = None,
    upper_limit: int | None = None,
):
    """Get data dicts for data loaders."""
    if ids_path is None:
        data_dicts = [{
            "flair": f'{root_path}/01045/03_flair_unhealthy_{idx}.png',
            "seg": f'{root_path}/01045/03_seg_unhealthy.png'
        } for idx in range(1000)]
    else:
        df = pd.read_csv(ids_path, sep="\t")

        if upper_limit is not None:
            df = df[:upper_limit]

        data_dicts = []
        if root_path is not None:
            for index, row in df.iterrows():
                data_dicts.append(
                    {
                        "flair": f'{root_path}/{row["flair"]}',
                        "seg": f'{root_path}/{row["seg"]}',
                        "report": "T1-weighted image of a brain.",
                    }
                )
        else:
            for index, row in df.iterrows():
                data_dicts.append(
                    {
                        "flair": row["flair"],
                        "seg": row["seg"]
                    }
                )

    print(f"Found {len(data_dicts)} subjects.")
    return data_dicts
