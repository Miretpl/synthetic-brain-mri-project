from typing import Optional, Union, List

import torch
from diffusers import DiffusionPipeline, ImagePipelineOutput, DDPMScheduler
from diffusers.utils.torch_utils import randn_tensor


class CustomDDPMPipeline(DiffusionPipeline):
    model_cpu_offload_seq = 'unet'

    def __init__(self, unet: torch.nn.Module, scheduler: DDPMScheduler) -> None:
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        control: Optional[torch.Tensor] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = None
    ) -> Union[ImagePipelineOutput, torch.Tensor]:
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        image = randn_tensor(image_shape, generator=generator, device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        if control is None:
            control = torch.zeros(image_shape, device=self.device)
        else:
            control = control.to(self.device)

        for t in self.progress_bar(self.scheduler.timesteps):
            image = self.scheduler.step(
                self.unet(image, control, t).sample,
                t,
                image,
                generator=generator
            ).prev_sample

        image = (image / 2 + 0.5).clamp(0, 1)

        if output_type == 'tensor':
            return image

        return ImagePipelineOutput(images=image.cpu().numpy())
