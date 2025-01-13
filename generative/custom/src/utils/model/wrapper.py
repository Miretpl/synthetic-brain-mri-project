import os
from os.path import join
from typing import Optional

import torch
from accelerate import Accelerator
from comet_ml import Experiment
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.config.config import ExperimentConfig
from utils.loss.loss_func import calculate_loss
from utils.experiment.pipeline import create_pipeline
from utils.image.generation import generate_images
from utils.model.validation import calculate_loss_no_grad, calculate_metric


class ModelWrapper:
    def __init__(
            self,
            model: torch.nn.Module,
            noise_scheduler: DDPMScheduler,
            accelerator: Accelerator,
            optimizer: torch.optim.Optimizer,
            lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
            tracker: Optional[Experiment],
            config: ExperimentConfig
    ) -> None:
        self.__model = model
        self.__noise_scheduler = noise_scheduler
        self.__accelerator = accelerator
        self.__optimizer = optimizer
        self.__lr_scheduler = lr_scheduler
        self.__tracker = tracker
        self.__config = config

        self.__step, self.__best_val_loss = 0, float('inf')

        self.__post_init__()

    def __post_init__(self) -> None:
        os.makedirs(self.__config.results_dir, exist_ok=True)

    def __train_step(self, sample: torch.FloatTensor, control: torch.FloatTensor) -> torch.Tensor:
        noise = torch.randn_like(sample, device=sample.device).float()

        timesteps = torch.randint(
            high=self.__noise_scheduler.config['num_train_timesteps'], size=(sample.shape[0],), device=sample.device
        ).int()

        noisy_sample = self.__noise_scheduler.add_noise(original_samples=sample, noise=noise, timesteps=timesteps)
        noisy_control = self.__noise_scheduler.add_noise(original_samples=control, noise=noise, timesteps=timesteps)

        with self.__accelerator.accumulate(self.__model):
            noise_pred = self.__model(
                sample=noisy_sample,
                control=noisy_control,
                timestep=timesteps
            ).sample

            loss = calculate_loss(pred=noise_pred, target=noise)

            self.__accelerator.backward(loss)

            if self.__accelerator.sync_gradients:
                self.__accelerator.clip_grad_norm_(
                    parameters=self.__model.parameters(),
                    max_norm=1.0
                )

            self.__optimizer.step()
            self.__lr_scheduler.step()
            self.__optimizer.zero_grad()

        return loss.detach()

    def __send_metrics_to_tracker(self, metrics: dict, epoch: int) -> None:
        self.__tracker.log_metrics(dic=metrics, step=self.__step, epoch=epoch)

    def __save_model(self, epoch: int) -> None:
        pipeline = create_pipeline(
            model=self.__accelerator.unwrap_model(self.__model),
            noise_scheduler=self.__noise_scheduler
        )

        pipeline.save_pretrained(
            save_directory=str(join(self.__config.results_dir, f'epoch_{epoch:04d}', 'model')),
            safe_serialization=False
        )

    def __update_best_loss_and_model(self, loss: float, epoch: int) -> bool:
        if loss < self.__best_val_loss:
            print(f'New best val loss: {loss}. Old loss: {self.__best_val_loss}. Saving the model.')

            self.__best_val_loss = loss
        else:
            print(f'Best val loss {self.__best_val_loss} not improved. Current val loss: {loss}. Saving the model.')

        self.__save_model(epoch=epoch)
        if self.__config.end_loss_threshold:
            return self.__best_val_loss < self.__config.end_loss_threshold

        return True

    def __train_epoch(self, data_loader: DataLoader, epoch: int) -> None:
        self.__model.train()
        e_tqdm = tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Epoch {epoch}', position=0)

        for step, batch in e_tqdm:
            sample, control = batch['flair'], batch['seg']

            loss = self.__train_step(sample=sample, control=control)
            self.__step += 1

            e_tqdm.set_postfix({
                'train_loss': loss.item(),
                'lr': self.__lr_scheduler.get_last_lr()[0],
                'step': self.__step
            })

            if self.__config.use_comet_ml:
                self.__send_metrics_to_tracker(metrics={'train_loss': loss.item()}, epoch=epoch)

    def __validate_epoch(self, data_loader: DataLoader, epoch: int) -> bool:
        self.__model.eval()

        loss = calculate_loss_no_grad(
            model=self.__model,
            noise_scheduler=self.__noise_scheduler,
            data_loader=data_loader,
            accelerator=self.__accelerator,
            desc='Validation loss'
        )

        if self.__config.use_comet_ml:
            self.__send_metrics_to_tracker(metrics={'val_loss': loss}, epoch=epoch)

        return self.__update_best_loss_and_model(loss=loss, epoch=epoch)

    def __generate_results(self, data_loader: DataLoader, epoch: int) -> None:
        self.__model.eval()

        pipeline = create_pipeline(
            model=self.__accelerator.unwrap_model(self.__model),
            noise_scheduler=self.__noise_scheduler
        )

        generate_images(
            pipeline=pipeline,
            data_loader=data_loader,
            config=self.__config,
            epoch=epoch,
            step=self.__step,
            tracker=self.__tracker
        )

    def __evaluation_step(self, data_loader: DataLoader, epoch: int) -> None:
        result = {}

        for metric in ('fid', 'ms-ssim'):
            result[metric.upper()] = calculate_metric(
                model=self.__model,
                data_loader=data_loader,
                accelerator=self.__accelerator,
                noise_scheduler=self.__noise_scheduler,
                config=self.__config,
                metric=metric
            )

        if self.__config.use_comet_ml:
            self.__send_metrics_to_tracker(metrics=result, epoch=epoch)

    def run(
            self,
            train_dataloader: DataLoader,
            val_dataloader: DataLoader,
            test_dataloader: DataLoader
    ) -> None:
        marker = self.__validate_epoch(data_loader=val_dataloader, epoch=0)
        self.__generate_results(data_loader=val_dataloader, epoch=0)

        epochs = self.__config.epochs if self.__config.end_loss_threshold is None else self.__config.epochs * 1000

        for epoch in range(epochs):
            self.__train_epoch(data_loader=train_dataloader, epoch=epoch)

            if (epoch + 1) % self.__config.model_val_epochs == 0 or epoch == self.__config.epochs - 1:
                marker = self.__validate_epoch(data_loader=val_dataloader, epoch=epoch)
                self.__generate_results(data_loader=val_dataloader, epoch=epoch)

            if (epoch + 1) % self.__config.model_eval_epochs == 0 or epoch == self.__config.epochs - 1 or marker:
                self.__evaluation_step(data_loader=test_dataloader, epoch=epoch)

            if marker:
                print(f'Finishing training due to val loss: {self.__best_val_loss} being below threshold'
                      f' {self.__config.end_loss_threshold}.')
                break

        self.__accelerator.end_training()
