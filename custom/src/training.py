import comet_ml
import torch
from diffusers import DDPMScheduler, get_cosine_schedule_with_warmup

from utils.data.dataset import get_datasets
from utils.experiment.wrapper import ExperimentWrapper
from base import config, model, noise_scheduler


train_loader, val_loader, test_loader = get_datasets(config=config)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_loader) * config.epochs)
)

experiment = ExperimentWrapper(
    config=config,
    model=model,
    noise_scheduler=noise_scheduler,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler
)

experiment.run(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    test_dataloader=test_loader
)
