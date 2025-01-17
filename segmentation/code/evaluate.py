from collections import defaultdict
from os import makedirs
from os.path import join
from pathlib import Path
from typing import Callable, Any

import numpy as np
import argparse

import torch
from PIL import Image
from torchmetrics import JaccardIndex
from torchmetrics.functional import dice

from model import UNET
from datasets import get_datalist, get_data_loader
from tqdm import tqdm
from torchvision import transforms

from vars import NUM_CLASSES, IMAGE_HEIGHT, IMAGE_WIDTH

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def __calculate_metric(
        func: Callable, pred: torch.Tensor, target: torch.Tensor, status: tuple, average: str = None
) -> tuple:
    result_healthy, result_unhealthy = [], []

    if average is not None:
        for p, t, s in zip(pred, target, status):
            if s == 'healthy':
                result_healthy.append(dice(p, t, average=average, num_classes=NUM_CLASSES)[None, ...])
            else:
                result_unhealthy.append(dice(p, t, average=average, num_classes=NUM_CLASSES)[None, ...])
    else:
        for p, t, s in zip(pred, target, status):
            if s == 'healthy':
                result_healthy.append(func(p, t)[None, ...])
            else:
                result_unhealthy.append(func(p, t)[None, ...])

    return torch.concatenate(result_healthy, dim=0).mean(dim=0), torch.concatenate(result_unhealthy, dim=0).mean(dim=0)


def __get_dict_result(micro: tuple, macro: tuple, weighted: tuple) -> dict:
    return {
        'healthy': {
            'micro': micro[0].item(),
            'macro': macro[0].item(),
            'weighted': weighted[0].item()
        },
        'unhealthy': {
            'micro': micro[1].item(),
            'macro': macro[1].item(),
            'weighted': weighted[1].item()
        },
        'avg': {
            'micro': np.average(micro),
            'macro': np.average(macro),
            'weighted': np.average(weighted)
        }
    }


def _merge_results(data: list) -> defaultdict[Any, dict]:
    result = defaultdict(dict)

    for key in data[0].keys():
        for type_ in data[0][key].keys():
            result[key][type_] = round(torch.Tensor([el[key][type_] for el in data]).mean().item(), 4)

    return result


def save_predictions(data: torch.utils.data.DataLoader, model: torch.nn.Module, result_images_dir_path: str) -> tuple:
    model.eval()

    jac_micro = JaccardIndex(task='multiclass', num_classes=NUM_CLASSES, average='micro')
    jac_macro = JaccardIndex(task='multiclass', num_classes=NUM_CLASSES, average='macro')
    jac_weighted = JaccardIndex(task='multiclass', num_classes=NUM_CLASSES, average='weighted')

    iou_metrics, dice_metrics = [], []

    with torch.no_grad():
        for idx, batch in tqdm(enumerate(data), total=len(data), desc='Evaluation'):
            x, y, status, path = batch
            x, y = x.to(device), y.to(device)

            predictions = model(x)
            pred_labels = torch.argmax(predictions, dim=1)
            pred_labels = pred_labels.float()

            pred_labels = transforms.Resize(
                (IMAGE_HEIGHT, IMAGE_WIDTH),
                antialias=None
            )(pred_labels).to(torch.int64)[:, None, ...]

            x, y, pred_labels = x.cpu(), y.cpu(), pred_labels.cpu()

            iou_metrics.append(__get_dict_result(
                micro=__calculate_metric(func=jac_micro, pred=pred_labels, target=y, status=status),
                macro=__calculate_metric(func=jac_macro, pred=pred_labels, target=y, status=status),
                weighted=__calculate_metric(func=jac_weighted, pred=pred_labels, target=y, status=status)
            ))

            dice_metrics.append(__get_dict_result(
                micro=__calculate_metric(func=dice, pred=pred_labels, target=y, status=status, average='micro'),
                macro=__calculate_metric(func=dice, pred=pred_labels, target=y, status=status, average='macro'),
                weighted=__calculate_metric(func=dice, pred=pred_labels, target=y, status=status, average='weighted')
            ))

            x, y, pred_labels = x * 255, y / NUM_CLASSES * 255, pred_labels / NUM_CLASSES * 255
            im = np.concatenate([
                x.numpy(),
                y.numpy(),
                pred_labels.numpy()
            ], axis=3).astype(np.uint8)

            for img_all, img_pred, p in zip(im, pred_labels.numpy().astype(np.uint8), path):
                img_all = Image.fromarray(img_all[0])
                img_pred = Image.fromarray(img_pred[0])

                full_raw_path = Path(str(join(result_images_dir_path, p)))
                makedirs(full_raw_path.parent, exist_ok=True)

                img_all.save(join(full_raw_path.parent, full_raw_path.name.replace('.png', '_all.png')))
                img_pred.save(join(full_raw_path.parent, full_raw_path.name.replace('.png', '_pred.png')))

    return _merge_results(iou_metrics), _merge_results(dice_metrics)


parser = argparse.ArgumentParser()

parser.add_argument("--runs_dir", help="Localisation of run directory.")
parser.add_argument(
    "--epoch_number",
    type=int,
    default=100,
    help="Number of epoch from which model will be retrieved."
)
parser.add_argument("--results_dir", help="Localisation of results directory.")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")

args = parser.parse_args()

test_set = get_data_loader(
    data=get_datalist(
        ids="/data/ids/raw/test.tsv",
        real_data_root_path="/data/raw/extracted",
    ),
    transforms=transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH))
    ]),
    batch_size=args.batch_size,
    evaluation=True
)

net = UNET(in_channels=1, classes=NUM_CLASSES).to(device)
checkpoint = torch.load(join(f'{args.runs_dir}/epoch_{args.epoch_number:04d}/model.pth'))
net.load_state_dict(checkpoint['model_state_dict'])
net.eval()

result_dataset_images_dir_path = join(args.results_dir, 'images')
Path(result_dataset_images_dir_path).mkdir(parents=True, exist_ok=True)

iou, dice = save_predictions(
    data=test_set,
    model=net,
    result_images_dir_path=result_dataset_images_dir_path
)

with open(join(args.results_dir, 'metrics.txt'), 'w') as f:
    f.write(f'IoU: {dict(iou)}\nDice: {dict(dice)}')
