"""
Script for extracting PNG images from raw NII data. Scripts saves raw images in extracted directory. Images saved
under generation directory have modified segmentation masks with information about brain region.
"""
from glob import glob
from json import dumps
from os.path import isfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL.Image import fromarray
from nibabel import load
from tqdm import tqdm


def get_image(path: str, source_image: Optional[np.ndarray] = None) -> tuple[np.ndarray, list]:
    empty_seg_number = None
    img = load(path).get_fdata()[40:200, 11:235, 20:140:20]

    if source_image is not None:
        empty_seg_number = [np.isclose(img[..., i].max(), 0) for i in range(img.shape[2])]

        _, thresh = cv2.threshold(source_image, 1, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((6, 6), np.uint8))
        thresh = np.where(thresh == 255, 3, 0)

        img = np.where(img == 0, thresh, img)

    img = img.astype('float32')

    if not np.isclose(img.max(), 0):
        img = (img - img.min()) / (img.max() - img.min())

    return (img * 255).astype('uint8'), empty_seg_number


images_paths = sorted(glob('/data/raw/BraTS2021_Training_Data/**/*flair.nii.gz'))
img_quantity, empty_seg_map_img_quantity = 0, 0

for flair_path in tqdm(images_paths, desc='Image extraction'):
    seg_path = flair_path.replace('flair', 'seg')

    if not isfile(flair_path):
        raise ValueError(f'File {seg_path} does not exist. Missing segmentation map for flair image.')

    patient_id = flair_path.split('/')[-1].split('_')[1]

    extracted_patient_dir = Path(
        flair_path.replace('BraTS2021_Training_Data/BraTS2021_', 'extracted/')
    ).parents[0]
    extracted_patient_dir.mkdir(parents=True, exist_ok=True)

    generation_patient_dir = Path(
        flair_path.replace('raw/BraTS2021_Training_Data/BraTS2021_', 'generation/')
    ).parents[0]
    generation_patient_dir.mkdir(parents=True, exist_ok=True)

    flair_img, _ = get_image(flair_path)
    seg_org_img, _ = get_image(seg_path)
    seg_mod_img, healthy_quantity = get_image(seg_path, source_image=flair_img)

    if healthy_quantity:
        empty_seg_map_img_quantity += sum(healthy_quantity)

    img_quantity = len(images_paths) * flair_img.shape[2]

    for i in range(flair_img.shape[2]):
        flair_slice = fromarray(flair_img[:, :, i])
        seg_org_slice = fromarray(seg_org_img[:, :, i])
        seg_mod_slice = fromarray(seg_mod_img[:, :, i])

        flair_name = f'{i:02d}_flair_{"healthy" if healthy_quantity[i] else "unhealthy"}.png'
        seg_name = flair_name.replace('flair', 'seg')

        flair_slice.save(extracted_patient_dir / flair_name)
        flair_slice.save(generation_patient_dir / flair_name)
        seg_org_slice.save(extracted_patient_dir / seg_name)
        seg_mod_slice.save(generation_patient_dir / seg_name)


metadata = {
    'quantity': img_quantity,
    'healthy_quantity': empty_seg_map_img_quantity.tolist(),
    'healthy_percentage': round(empty_seg_map_img_quantity / img_quantity * 100, 2).tolist()
}

meta_dir = '/data/metadata/dataset'
Path(meta_dir).mkdir(parents=True, exist_ok=True)

with open(f'{meta_dir}/01_png_img_extraction.json', 'w') as f:
    f.write(dumps(metadata, indent=4))

print(dumps(metadata, indent=4))
