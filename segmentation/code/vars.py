from os import makedirs
from os.path import join

SEG_MODEL_TRAIL = 5
DATASET_TYPE = 'raw'

MODEL_DIR_PATH = join('..', 'cache', 'unet', f'{SEG_MODEL_TRAIL:02d}')
RESULTS_DIR_PATH = join('..', 'results', f'{SEG_MODEL_TRAIL:02d}')

makedirs(MODEL_DIR_PATH, exist_ok=True)
makedirs(RESULTS_DIR_PATH, exist_ok=True)

DATASET_TRAIL = 3
LOAD_MODEL = False
BATCH_SIZE = 16
EPOCHS = 100
NUM_CLASSES = 4
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 224

EVAL = True
PLOT_LOSS = False

# IDS_ROOT_PATH = join('..', 'dataset', 'Generative_Brain_ControlNet', 'ids', f'{DATASET_TRAIL:02d}', 'mixed')
IDS_ROOT_PATH = join('../..', 'dataset', 'all', 'ids', 'segmentation', f'{DATASET_TRAIL:02d}')
IDS_EVAL_ROOT_PATH = join('../..', 'dataset', 'all', 'ids', 'generation', f'{DATASET_TRAIL:02d}')
TRAIN_IDS = 'train.tsv'
VAL_IDS = 'validation.tsv'
TEST_IDS = 'test.tsv'
DATASET_TYPE_MAPPING = {
    'raw': 'raw',
    'custom-50': 'gen_custom\\50',
    'custom-100': 'gen_custom\\100',
    'controlnet-50': 'gen_controlnet\\50',
    'controlnet-100': 'gen_controlnet\\100',
}
