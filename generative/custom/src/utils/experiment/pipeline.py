import torch
from diffusers import DDPMScheduler

from utils.experiment.custom_ddpm_pipeline import CustomDDPMPipeline


def create_pipeline(model: torch.nn.Module, noise_scheduler: DDPMScheduler) -> CustomDDPMPipeline:
    pipeline = CustomDDPMPipeline(unet=model, scheduler=noise_scheduler)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline
