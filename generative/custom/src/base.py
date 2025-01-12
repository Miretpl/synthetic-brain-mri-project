from diffusers import DDPMScheduler

from utils.config.config import ExperimentConfig
from utils.model.custom_unet_2d import CustomUNet2DModel


config = ExperimentConfig()

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
