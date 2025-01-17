import argparse
from os import listdir
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

from model import UNET

from datasets import get_datalist, get_data_loader
from loss import dice_loss, focal_loss
from vars import NUM_CLASSES


def train_function(
        data: torch.utils.data.DataLoader,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        current_epoch: int,
        epochs: int
) -> float:
    t_tqdm = tqdm(data, desc=f'Epoch {current_epoch + 1}/{epochs}')
    t_tqdm.set_postfix({'loss': float('inf')})

    loss_hist, loss = [], None
    for index, batch in enumerate(t_tqdm):
        x, y = batch
        pred = model(x)

        pred = pred.reshape((*pred.shape[:2], -1))
        y = y.reshape((y.shape[0], -1))

        y = F.one_hot(y, num_classes=NUM_CLASSES)
        y = y.permute(0, 2, 1).to(torch.float32)

        loss = loss_fn(pred, y)
        loss_hist.append(loss.item())
        t_tqdm.set_postfix({'loss': torch.Tensor(loss_hist).mean().item()})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


train_transforms = transforms.Compose([
    transforms.Resize((160, 224))
])

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", help="Model output directory.")
parser.add_argument("--train_ids", help="Location of file with test ids.")
parser.add_argument("--real_data_path", help="Location of file with test ids.")
parser.add_argument("--fake_data_path", default=None, help="Location of file with test ids.")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
parser.add_argument(
    "--model_checkpoint_period", type=int, default=20, help="Number of epochs after which model is saved."
)

args = parser.parse_args()

output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)

runs_dir_content = listdir(args.output_dir)

if len(runs_dir_content) == 0:
    attempt = 1
else:
    attempt = max([int(i) for i in runs_dir_content]) + 1

output_dir = Path(args.output_dir, f'{attempt:02d}')
output_dir.mkdir(exist_ok=True, parents=True)

train_set = get_data_loader(
    data=get_datalist(
        ids=args.train_ids,
        real_data_root_path=args.real_data_path,
        fake_data_root_path=args.fake_data_path,
    ),
    transforms=train_transforms,
    batch_size=args.batch_size
)

unet = UNET(
    in_channels=1,
    classes=NUM_CLASSES
).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')).train()
optimizer = torch.optim.Adam(unet.parameters(), lr=0.0001)

loss_values, epoch = [], 0

for e in range(epoch, args.epochs):
    loss_val = train_function(
        data=train_set,
        model=unet,
        optimizer=optimizer,
        loss_fn=focal_loss,
        current_epoch=e,
        epochs=args.epochs
    )
    loss_values.append(loss_val)

    if e > 0 and e % args.model_checkpoint_period == 0:
        output_path = Path(f'{output_dir}/epoch_{e:04d}')
        output_path.mkdir(exist_ok=True, parents=True)

        torch.save(
            obj={
                'model_state_dict': unet.state_dict(),
                'optim_state_dict': optimizer.state_dict(),
                'epoch': e,
                'loss_values': loss_values
            },
            f=f'{output_path}/model.pth',
        )

output_path = Path(f'{output_dir}/epoch_{args.epochs:04d}')
output_path.mkdir(exist_ok=True, parents=True)

torch.save(
    obj={
        'model_state_dict': unet.state_dict(),
        'optim_state_dict': optimizer.state_dict(),
        'epoch': args.epochs,
        'loss_values': loss_values
    },
    f=f'{output_path}/model.pth',
)
