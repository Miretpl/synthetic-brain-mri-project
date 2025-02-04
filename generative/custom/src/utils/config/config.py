from os import listdir
from dataclasses import dataclass
from pathlib import Path


MODELS_ROOT = '/models/runs'
Path(MODELS_ROOT).mkdir(parents=True, exist_ok=True)

result_root_dir_content = listdir(MODELS_ROOT)

if len(result_root_dir_content) == 0:
    attempt = 1
else:
    attempt = max([int(i) for i in result_root_dir_content]) + 1


@dataclass
class ExperimentConfig:
    use_comet_ml = False
    seed = 0
    train_batch_size = 2
    val_batch_size = 2
    test_batch_size = 64
    gen_batch_size = 24
    num_workers = 4
    images_to_generate = 8
    epochs = 200
    gradient_accumulation_steps = 8
    learning_rate = 1e-4
    lr_warmup_steps = 500
    end_loss_threshold = 0.0002
    dataset_root_path = '/data/generation'
    dataset_ids_path = '/data/ids/raw'
    training_ids = 'train.tsv'
    validation_ids = 'validation.tsv'
    test_ids = 'test.tsv'
    run_id = attempt
    results_dir = f'{MODELS_ROOT}/{attempt:02d}'
    results_dir_root = MODELS_ROOT
    num_train_timesteps = 2000
    model_eval_epochs = 50
    model_val_epochs = 5
