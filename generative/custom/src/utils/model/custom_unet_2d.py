from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers import ModelMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.embeddings import GaussianFourierProjection, Timesteps, TimestepEmbedding
from diffusers.models.unet_2d import UNet2DOutput
from diffusers.models.unet_2d_blocks import get_down_block, UNetMidBlock2D, get_up_block


class CustomUNet2DModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        sample_size: Union[int, Tuple],
        in_channels: int,
        out_channels: int,
        layers_per_block: int,
        block_out_channels: Tuple,
        down_block_types: Tuple,
        up_block_types: Tuple,
        time_embedding_type: str = 'positional',
        freq_shift: int = 0,
        flip_sin_to_cos: bool = True,
        mid_block_scale_factor: float = 1,
        downsample_padding: int = 1,
        downsample_type: str = 'conv',
        upsample_type: str = 'conv',
        dropout: float = 0.0,
        act_fn: str = 'silu',
        attention_head_dim: int = 8,
        norm_num_groups: int = 32,
        attn_norm_num_groups: Optional[int] = None,
        norm_eps: float = 1e-5,
        resnet_time_scale_shift: str = 'default',
        add_attention: bool = True,
        num_train_timesteps: Optional[int] = None
    ) -> None:
        super().__init__()

        self.sample_size = sample_size
        self.time_embedding_type = time_embedding_type

        time_embed_dim = block_out_channels[0] * 4

        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f'Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: '
                f'{down_block_types}. `up_block_types`: {up_block_types}.'
            )

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f'Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: '
                f'{block_out_channels}. `down_block_types`: {down_block_types}.'
            )

        # input
        self.conv_in_sample = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))
        self.conv_in_control = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))

        # time
        if time_embedding_type == 'fourier':
            self.time_proj = GaussianFourierProjection(embedding_size=block_out_channels[0], scale=16)
            timestep_input_dim = 2 * block_out_channels[0]
        elif time_embedding_type == 'positional':
            self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
            timestep_input_dim = block_out_channels[0]
        elif time_embedding_type == 'learned':
            self.time_proj = nn.Embedding(num_train_timesteps, block_out_channels[0])
            timestep_input_dim = block_out_channels[0]
        else:
            raise ValueError('Invalid time_embedding_type.')

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        self.down_blocks_sample = nn.ModuleList([])
        self.mid_block_sample = None
        self.up_blocks = nn.ModuleList([])

        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel * 2,
                out_channels=output_channel * 2,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=attention_head_dim if attention_head_dim is not None else output_channel * 2,
                downsample_padding=downsample_padding,
                resnet_time_scale_shift=resnet_time_scale_shift,
                downsample_type=downsample_type,
                dropout=dropout,
            )
            self.down_blocks_sample.append(down_block)

        self.mid_block_sample = UNetMidBlock2D(
            in_channels=block_out_channels[-1] * 2,
            temb_channels=time_embed_dim,
            dropout=dropout,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift=resnet_time_scale_shift,
            attention_head_dim=attention_head_dim if attention_head_dim is not None else block_out_channels[-1] * 2,
            resnet_groups=norm_num_groups,
            attn_groups=attn_norm_num_groups,
            add_attention=add_attention,
        )

        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel * 2,
                out_channels=output_channel * 2,
                prev_output_channel=prev_output_channel * 2,
                temb_channels=time_embed_dim,
                add_upsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=attention_head_dim if attention_head_dim is not None else output_channel * 2,
                resnet_time_scale_shift=resnet_time_scale_shift,
                upsample_type=upsample_type,
                dropout=dropout,
            )
            self.up_blocks.append(up_block)

        num_groups_out = norm_num_groups if norm_num_groups is not None else min(block_out_channels[0] // 4, 32)
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0] * 2, num_groups=num_groups_out, eps=norm_eps
        )
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0] * 2, out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        sample: torch.FloatTensor,
        control: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int]
    ) -> Union[UNet2DOutput, Tuple]:
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)
        t_emb = self.time_proj(timesteps)

        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        # 2. pre-process
        skip_sample = torch.concat([sample, control], dim=1)
        sample = torch.concat([
            self.conv_in_sample(sample),
            self.conv_in_control(control)
        ], dim=1)

        # 3. down
        down_block_data = (sample,)
        for down_sample in self.down_blocks_sample:
            if hasattr(down_sample, 'skip_conv'):
                sample, res_samples, skip_sample = down_sample(hidden_states=sample, temb=emb, skip_sample=skip_sample)
            else:
                sample, res_samples = down_sample(hidden_states=sample, temb=emb)

            down_block_data += res_samples

        # 4. mid
        sample = self.mid_block_sample(sample, emb)
        skip_sample = None

        # 5. up
        for i, up_sample in enumerate(self.up_blocks):
            res_samples = down_block_data[-len(up_sample.resnets):]
            down_block_data = down_block_data[: -len(up_sample.resnets)]

            if hasattr(up_sample, 'skip_conv'):
                sample, skip_sample = up_sample(sample, res_samples, emb, skip_sample)
            else:
                sample = up_sample(sample, res_samples, emb)

        # 6. post-process
        sample = self.conv_out(self.conv_act(self.conv_norm_out(sample)))

        if skip_sample is not None:
            sample += skip_sample

        if self.time_embedding_type == 'fourier':
            timesteps = timesteps.reshape((sample.shape[0], *([1] * len(sample.shape[1:]))))
            sample = sample / timesteps

        return UNet2DOutput(sample=sample)
