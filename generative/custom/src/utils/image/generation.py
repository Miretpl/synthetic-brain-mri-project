from os import makedirs
from os.path import join

import numpy as np
import torch
from PIL.Image import fromarray
from comet_ml import Experiment
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.config.config import ExperimentConfig
from utils.experiment.custom_ddpm_pipeline import CustomDDPMPipeline


def __generate_images(
        pipeline: CustomDDPMPipeline,
        data_loader: DataLoader,
        config: ExperimentConfig,
        use_control: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | np.ndarray:
    data_loader_iter = iter(data_loader)

    generated_images, samples, controls = [], [], []

    if config.images_to_generate > config.val_batch_size:
        r_val = config.images_to_generate // config.val_batch_size
    else:
        r_val = 1

    desc = 'Image generation (Unconditional)'

    if use_control:
        desc = 'Image generation (Conditional)'

    for batch_idx in tqdm(range(r_val), desc=desc, position=0):
        data = next(data_loader_iter)

        if use_control:
            samples.append(data['flair'].cpu().numpy()[:config.images_to_generate])
            controls.append(data['seg'].cpu().numpy()[:config.images_to_generate])

            generated_images.append(pipeline(
                batch_size=config.val_batch_size,
                control=data['seg'],
                generator=torch.manual_seed(config.seed + batch_idx)
            ).images[:config.images_to_generate])
        else:
            generated_images.append(pipeline(
                batch_size=config.val_batch_size,
                generator=torch.manual_seed(config.seed + batch_idx)
            ).images[:config.images_to_generate])

    if use_control:
        return (
            np.concatenate(generated_images[:config.images_to_generate], axis=0),
            np.concatenate(samples[:config.images_to_generate], axis=0),
            np.concatenate(controls[:config.images_to_generate], axis=0)
        )

    return np.concatenate(generated_images[:config.images_to_generate], axis=0)


def __create_root_paths_for_images(epoch: int, config: ExperimentConfig) -> tuple:
    test_un_dir = join(config.results_dir, f'epoch_{epoch:04d}', 'samples', 'unconditional')
    test_con_dir = join(config.results_dir, f'epoch_{epoch:04d}', 'samples', 'conditional')

    makedirs(test_un_dir, exist_ok=True)
    makedirs(test_con_dir, exist_ok=True)

    return test_un_dir, test_con_dir


def __transform_images(data: np.ndarray, prediction: bool) -> np.ndarray:
    if prediction is True:
        result = []

        for img in data:
            result.append((np.interp(img, (img.min(), img.max()), (0, +1)) * 255).astype(np.uint8))

        return np.array(result)

    return (data * 255).astype(np.uint8) if data.max() <= 1 else data.astype(np.uint8)


def ___merge_images(samples: np.ndarray, controls: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    result = np.empty(shape=(0, *samples.shape[1:-1], samples.shape[-1] * 3), dtype=np.uint8)

    for sample, control, pred in zip(samples, controls, predictions):
        result = np.append(result, np.concatenate([sample, control, pred], axis=-1)[None, ...], axis=0)

    return result


def __concat_all_images(data: np.ndarray) -> np.ndarray:
    result = np.empty(shape=(0, data.shape[-1]), dtype=np.uint8)

    for i, img in enumerate(data):
        result = np.append(result, img[0], axis=0)

        if i < data.shape[0] - 1:
            result = np.append(result, np.full((5, data.shape[-1]), fill_value=254, dtype=np.uint8), axis=0)

    return result


def __send_img_to_tracker(img: np.ndarray, tracker: Experiment, name: str, step: int, is_conditional: bool) -> None:
    tracker.log_image(
        image_data=img,
        name=name,
        image_minmax=(0, 255),
        image_channels='first',
        step=step,
        metadata={'gen_type': 'conditional' if is_conditional else 'unconditional'}
    )


def __save_images_with_tracker(
        data: np.ndarray, tracker: Experiment, step: int, is_conditional: bool, is_all: bool
) -> None:
    if is_all:
        __send_img_to_tracker(img=data, tracker=tracker, name='all', step=step, is_conditional=is_conditional)
    else:
        for i in range(data.shape[0]):
            __send_img_to_tracker(
                img=data[i], tracker=tracker, name=f'{i:04d}', step=step, is_conditional=is_conditional
            )


def __save_images_local(data: np.ndarray, root_path: str, is_all: bool) -> None:
    if is_all:
        fromarray(data).save(join(root_path, 'all.png'))
    else:
        for i, img in enumerate(data):
            fromarray(img[0]).save(join(root_path, f'{i:04d}.png'))


def generate_images(
        pipeline: CustomDDPMPipeline,
        data_loader: torch.utils.data.DataLoader,
        config: ExperimentConfig,
        epoch: int,
        step: int,
        tracker: Experiment
) -> None:
    images_un = __generate_images(pipeline=pipeline, data_loader=data_loader, config=config, use_control=False)
    images_con, samples, controls = __generate_images(
        pipeline=pipeline, data_loader=data_loader, config=config, use_control=True
    )

    samples = __transform_images(data=samples, prediction=False)
    controls = __transform_images(data=controls, prediction=False)
    images_un = __transform_images(data=images_un, prediction=True)
    images_con = __transform_images(data=images_con, prediction=True)

    images_con = ___merge_images(samples=samples, controls=controls, predictions=images_con)

    un_all_images = __concat_all_images(data=images_un)
    con_all_images = __concat_all_images(data=images_con)

    if config.use_comet_ml:
        __save_images_with_tracker(data=images_un, tracker=tracker, step=step, is_conditional=False, is_all=False)
        __save_images_with_tracker(data=images_con, tracker=tracker, step=step, is_conditional=True, is_all=False)
        __save_images_with_tracker(data=un_all_images, tracker=tracker, step=step, is_conditional=False, is_all=True)
        __save_images_with_tracker(data=con_all_images, tracker=tracker, step=step, is_conditional=True, is_all=True)

    images_un_dir, images_con_dir = __create_root_paths_for_images(epoch=epoch, config=config)
    __save_images_local(data=images_un, root_path=images_un_dir, is_all=False)
    __save_images_local(data=images_con, root_path=images_con_dir, is_all=False)
    __save_images_local(data=un_all_images, root_path=images_un_dir, is_all=True)
    __save_images_local(data=con_all_images, root_path=images_con_dir, is_all=True)
