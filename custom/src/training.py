import comet_ml
import torch
from diffusers import DDPMScheduler, get_cosine_schedule_with_warmup

from utils.config.config import ExperimentConfig
from utils.data.dataset import get_datasets
from utils.experiment.wrapper import ExperimentWrapper
from utils.model.custom_unet_2d import CustomUNet2DModel


config = ExperimentConfig()

train_loader, val_loader, test_loader = get_datasets(config=config)

model = CustomUNet2DModel(
    sample_size=(160, 224),
    in_channels=1,
    out_channels=1,
    layers_per_block=4,
    block_out_channels=(32, 32, 64, 64, 128, 128),
    dropout=0.3,
    down_block_types=(
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    )
)

noise_scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps)
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
