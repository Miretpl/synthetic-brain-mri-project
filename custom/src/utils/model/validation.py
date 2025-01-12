import numpy as np
import torch
from accelerate import Accelerator
from diffusers import DDPMScheduler
from generative.metrics import MultiScaleSSIMMetric
from monai.transforms import ScaleIntensityRange
from torch.utils.data import DataLoader
from torcheval.metrics import FrechetInceptionDistance
from tqdm import tqdm

from utils.config.config import ExperimentConfig
from utils.experiment.pipeline import create_pipeline
from utils.loss.loss_func import calculate_loss


fid = FrechetInceptionDistance()
ms_ssim = MultiScaleSSIMMetric(spatial_dims=2, data_range=1.0, kernel_size=7)
change_scale = ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True)


def calculate_loss_no_grad(
        model: torch.nn.Module,
        noise_scheduler: DDPMScheduler,
        data_loader: DataLoader,
        accelerator: Accelerator,
        desc: str
) -> float:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data_loader = accelerator.prepare(model, data_loader, device_placement=(device, device))

    model.eval()
    model.enable_xformers_memory_efficient_attention()

    loss_batch = torch.empty(len(data_loader), dtype=torch.float, device=device)
    for step, batch in tqdm(enumerate(data_loader), desc=desc, total=len(data_loader), position=0):
        sample_batch, control_batch = batch['flair'], batch['seg']

        with torch.no_grad():
            noise = torch.randn_like(sample_batch, device=device).float()

            timesteps = torch.randint(
                high=noise_scheduler.config['num_train_timesteps'], size=(sample_batch.shape[0],), device=device
            ).int()

            noise_pred = model(
                sample=noise_scheduler.add_noise(original_samples=sample_batch, noise=noise, timesteps=timesteps),
                control=control_batch,
                timestep=timesteps
            ).sample

            loss = calculate_loss(pred=noise_pred, target=noise)

        loss_batch[step] = loss.detach()
    return loss_batch.mean().item()


def calculate_fid(pred: torch.Tensor, target: torch.Tensor) -> float:
    target = change_scale(target)
    fid.update(target.expand(-1, 3, -1, -1), True)
    fid.update(pred.expand(-1, 3, -1, -1), False)
    return fid.compute().item()


def calculate_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    return ms_ssim(target, pred).mean().item()


metrics = {
    'fid': calculate_fid,
    'ms-ssim': calculate_ssim
}


def calculate_metric(
        model: torch.nn.Module,
        data_loader: DataLoader,
        accelerator: Accelerator,
        noise_scheduler: DDPMScheduler,
        config: ExperimentConfig,
        metric: str
) -> float:
    model.eval()

    pipeline = create_pipeline(
        model=accelerator.unwrap_model(model),
        noise_scheduler=noise_scheduler
    )

    result = []
    for step, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Calculate {metric.upper()} score'):
        pred = pipeline(
            batch_size=batch['seg'].shape[0],
            control=batch['seg'],
            generator=torch.manual_seed(config.seed + step),
            output_type='tensor'
        )

        result.append(metrics[metric](pred=pred, target=batch['flair']))

    return np.array(result).mean()
