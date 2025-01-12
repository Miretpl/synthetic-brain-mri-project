import torch
from accelerate import Accelerator
from accelerate.utils import PrecisionType, LoggerType
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader

from utils.config.config import ExperimentConfig
from utils.model.wrapper import ModelWrapper


class ExperimentWrapper:
    def __init__(
            self,
            config: ExperimentConfig,
            model: torch.nn.Module,
            noise_scheduler: DDPMScheduler,
            optimizer: torch.optim.Optimizer,
            lr_scheduler: torch.optim.lr_scheduler.LambdaLR
    ) -> None:
        self.__config = config
        self.__model = model
        self.__noise_scheduler = noise_scheduler
        self.__optimizer = optimizer
        self.__lr_scheduler = lr_scheduler

        self.__accelerator, self.__tracker, self.__device = None, None, None

        self.__post_init__()

    def __post_init__(self) -> None:
        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.__init_accelerator_and_tracker()

    def __init_accelerator_and_tracker(self) -> None:
        self.__accelerator = Accelerator(
            mixed_precision=PrecisionType.FP16,
            gradient_accumulation_steps=self.__config.gradient_accumulation_steps,
            log_with=LoggerType.COMETML if self.__config.use_comet_ml else None
        )

        if self.__config.use_comet_ml:
            self.__accelerator.init_trackers(
                project_name='master-thesis',
                init_kwargs={
                    str(LoggerType.COMETML): {
                        'auto_metric_logging': False,
                        'auto_metric_step_rate': False,
                        'auto_output_logging': False
                    }
                }
            )

            self.__tracker = self.__accelerator.get_tracker(name=str(LoggerType.COMETML), unwrap=True)
            self.__tracker.add_tag(f'Run:{self.__config.run_id}')

    def run(self, train_dataloader: DataLoader, val_dataloader: DataLoader, test_dataloader: DataLoader) -> None:
        self.__model.train()
        self.__model.enable_xformers_memory_efficient_attention()

        model, optimizer, train_dataloader, val_dataloader, test_dataloader, lr_scheduler = self.__accelerator.prepare(
            self.__model, self.__optimizer, train_dataloader, val_dataloader, test_dataloader, self.__lr_scheduler,
            device_placement=(self.__device, self.__device, self.__device, self.__device, self.__device, self.__device)
        )

        model_wrapper = ModelWrapper(
            model=model,
            noise_scheduler=self.__noise_scheduler,
            accelerator=self.__accelerator,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            tracker=self.__tracker,
            config=self.__config
        )

        model_wrapper.run(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader
        )
