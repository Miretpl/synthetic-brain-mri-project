"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import torch.utils.data as data
from pathlib import Path
from PIL import Image
import pandas as pd
from os.path import join

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
    ".tif",
    ".TIF",
    ".tiff",
    ".TIFF",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, ids_path, max_dataset_size=float("inf")):
    images = []
    dir_path = Path(dir)
    assert dir_path.is_dir(), f"{dir} is not a valid directory"

    if ids_path is None:
        images = [{
            'flair': f'{dir_path}/01045/03_flair_unhealthy.png',
            'path': f'{dir_path}/01045/03_flair_unhealthy_{idx}.png',
            # This will be used as path for saving image
            'seg': f'{dir_path}/01045/03_seg_unhealthy.png'
        } for idx in range(1000)]
    else:
        df = pd.read_csv(ids_path, sep='\t')

        if dir_path is not None:
            images = [
                {
                    'flair': join(dir_path, row['flair']),
                    'path': join(dir_path, row['flair']),
                    'seg': join(dir_path, row['seg'])
                }
                for index, row in df.iterrows()
            ]
        else:
            images = [
                {
                    'flair': row['flair'],
                    'path': row['flair'],
                    'seg': row['seg']
                }
                for index, row in df.iterrows()
            ]

    return images[: min(max_dataset_size, len(images))]


def default_loader(path):
    return Image.open(path).convert("RGB")


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False, loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in: " + root + "\n" "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
