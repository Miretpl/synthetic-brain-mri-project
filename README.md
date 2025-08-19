# Information
The content of this repository represents work done by Przemysław Mirowski (owner of this repository) during Master's 
thesis titled "Generation of brain scan images from segmentation maps using diffusion models" done at Lodz University 
of Technology in Poland.

This work is licensed under [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/?ref=chooser-v1).

## Citation
```text
@mastersthesis{mirowski2024,
    author={Przemysław Mirowski},
    title={Generation of brain scan images from segmentation maps using diffusion models},
    school={Lodz University of Technology},
    year=2024
}
```

## General information
All content of the repository was tested on Windows 11 23H2 with Docker Desktop 4.37.1 and NVIDIA Studio Driver 566.36.
Computer configuration is listed in the table below:

| Graphic card                 | Memory | CPU               |
|------------------------------|--------|-------------------|
| NVIDIA GeForce RTX 3080 12GB | 64 GB  | AMD Ryzen 7 5800X |

Below there are descriptions regarding every part of the work:
1. Data preparation - focuses on preparing data for generative model training and creating sets of ids for generative 
   and segmentation model training,
2. Generative models - focuses on proposed and ControlNet model training, data generation for evaluation and segmentation
   model, and evaluation of generative models
3. Segmentation model - focuses on segmentation model training

All sections are separate from each other which means that when there is command execution it should be done from root 
repository directory.

## Data preparation
To run scripts for data preparation you need to execute below commands:
1. Move to dataset directory
   ```shell
   cd ./dataset
   ```
2. Run PowerShell script (build and run docker container)
   ```shell
   ./run.ps1 `
      -dataPath "C:\Users\$env:USERNAME\Desktop\data" `
      -modelsPath "C:\Users\$env:USERNAME\Desktop\models"
   ```
   where you need to create `data` and `models` directories. Under `data` directory you will have `raw` directory 
   created with `BraTS2021_Training_Data.tar` file downloaded and unpacked from 
   [BraTS2021](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1) website
3. After finished data preparation there will be couple new directories created:
   1. /data/raw/extracted - there are raw data which was extracted from nii.gz files to png
   2. /data/metadata/dataset - there are some information regarding generated data
   3. /data/ids/raw - there are files with information about which patient data belongs to which set group: train, validation or test

## Generative models
### Training
#### Proposed model
To train proposed model you need to execute below commands:
1. Move to custom model directory
   ```shell
   cd ./generative/custom
   ```
2. Run PowerShell script (build and run docker container)
   ```shell
   .\run.ps1 `
      -dataPath "C:\Users\$env:USERNAME\Desktop\data" `
      -modelPath "C:\Users\$env:USERNAME\Desktop\models\generation\custom"
   ```
   where you need to create `generation/custom` directory under `models`.
3. Model training (running script instead docker container)
   ```shell
   ./src/bash/training/01_training.sh
   ```

When we will have our final model ready we can start to evaluation and generation of data for segmentation model (all
command should be executed in previously created docker container):
1. Data generation for reconstruction analysis
   ```shell
   ./src/bash/generation/test/01_reconstruction.sh
   ```
   before running script you need to provide proper `--run_id` value (if it is the last run it will be the newest name 
   of the directory under `/models/generation/custom/runs` in docker container or 
   `C:\Users\$env:USERNAME\Desktop\models\generation\custom\runs` in local).
2. Data generation for diversity analysis
   ```shell
   ./src/bash/generation/test/02_diversity.sh
   ```
   before running script you need to provide proper `--run_id` value (if it is the last run it will be the newest name 
   of the directory under `/models/generation/custom/runs` in docker container or 
   `C:\Users\$env:USERNAME\Desktop\models\generation\custom\runs` in local). By default, for diversity test there will 
   be 1000 images generated. If you want to change that number you can modify value of `--img_to_gen_per_seg_map` 
   parameter inside the script.
3. Data generation for segmentation model
   ```shell
   ./src/bash/generation/seg/01_whole_train_set.sh
   ```
4. Copy segmentation maps for segmentation model
   ```shell
   ./src/bash/generation/seg/02_copy_seg_masks.sh
   ```

#### ControlNet model
To train ControlNet model you need to execute below commands:
1. Move to ControlNet model directory
   ```shell
   cd ./generative/generative_brain_controlnet
   ```
2. Run PowerShell script (build and run docker container)
   ```shell
   .\run.ps1 `
      -dataPath "C:\Users\$env:USERNAME\Desktop\data" `
      -configPath "C:\Users\$env:USERNAME\Desktop\synthetic-brain-mri-project\generative\generative_brain_controlnet\configs" `
      -artifactPath "C:\Users\$env:USERNAME\Desktop\models\generation\controlnet\artifacts" `
      -modelPath "C:\Users\$env:USERNAME\Desktop\models\generation\controlnet\runs" `
      -resultPath "C:\Users\$env:USERNAME\Desktop\models\generation\controlnet\results"
   ```
   where you need to create `generation/controlnet` directory under `models`. Under newly created `controlnet` directory
   you need to create also `artifacts`, `runs` and `results` directories. Also, below command will work if the content 
   of this repository will be cloned under `C:\Users\$env:USERNAME\Desktop` path.
3. Model training - autoencoder
   ```shell
   ./src/bash/training/01_train_aekl.sh
   ```
4. Model training - diffusion model
   ```shell
   ./src/bash/training/02_train_ldm.sh
   ```
   where inside the script you need to update `mlrun_id` parameter with run_id which was printed out in console during
   autoencoder training.
5. Model training - ControlNet
   ```shell
   ./src/bash/training/03_train_controlnet.sh
   ```
   where inside the script you need to update `stage1_mlrun_id` (autoencoder) and `ldm_mlrun_id` (diffusion model) 
   parameters with run_id values printed during training of autoencoder and diffusion model.

When we will have our final model ready we can start to evaluation and generation of data for segmentation model (all
command should be executed in previously created docker container):
1. Conversion of MLFlow models to PyTorch
   ```shell
   ./src/bash/training/04_convert_mlflow_to_pytorch.sh
   ```
   where inside the script you need to update `stage1_mlrun_id` (autoencoder), `ldm_mlrun_id` (diffusion model) and 
   `controlnet_mlrun_id` (ControlNet) parameters with run_id values printed during training of autoencoder, diffusion 
   and ControlNet model.
2. Data generation for reconstruction analysis
   ```shell
   ./src/bash/generation/test/01_reconstruction.sh
   ```
3. Data generation for diversity analysis
   ```shell
   ./src/bash/generation/test/02_diversity.sh
   ```
4. Data generation for segmentation model
   ```shell
   ./src/bash/generation/seg/01_whole_train_set.sh
   ```
5. Copy segmentation maps for segmentation model
   ```shell
   ./src/bash/generation/seg/02_copy_seg_masks.sh
   ```

### Model evaluation
To run proposed and ControlNet models evaluation (calculation of FID and MS-SSIM scores) you need to execute below 
commands:
1. Move to testing directory
   ```shell
   cd ./generative/testing
   ```
2. Run PowerShell command (build and run docker container)
   ```shell
   ./run.ps1 `
      -dataPath "C:\Users\$env:USERNAME\Desktop\data" `
      -modelsGenPath "C:\Users\$env:USERNAME\Desktop\models\generation"
   ```
3. Run below command to generate MS-SSIM (reconstruction) for proposed model
   ```shell
   ./src/bash/testing/custom/01_reconstruction_ms-ssim.sh
   ```
4. Run below command to generate FID (reconstruction) for proposed model
   ```shell
   ./src/bash/testing/custom/01_reconstruction_fid.sh
   ```
5. Run below command to generate MS-SSIM (diversity) for proposed model
   ```shell
   ./src/bash/testing/custom/03_diversity_ms-ssim.sh
   ```
6. Run below command to generate MS-SSIM (reconstruction) for ControlNet model
   ```shell
   ./src/bash/testing/controlnet/01_reconstruction_ms-ssim.sh
   ```
7. Run below command to generate FID (reconstruction) for ControlNet model
   ```shell
   ./src/bash/testing/controlnet/01_reconstruction_fid.sh
   ```
8. Run below command to generate MS-SSIM (diversity) for ControlNet model
   ```shell
   ./src/bash/testing/controlnet/03_diversity_ms-ssim.sh
   ```

## Segmentation model
### Training
To train segmentation model you need to execute below commands:
1. Move to custom model directory
   ```shell
   cd ./segmentation
   ```
2. Run PowerShell script (build and run docker container)
   ```shell
   .\run.ps1 `
      -dataPath "C:\Users\$env:USERNAME\Desktop\data" `
      -modelPath "C:\Users\$env:USERNAME\Desktop\models\segmentation\artifacts" `
      -resultsPath "C:\Users\$env:USERNAME\Desktop\models\segmentation\results"
   ```
3. Start training of segmentation model
   ```shell
   ./bash/01_training.sh
   ```

### Evaluation
To evaluate segmentation models you need to execute below commands:
1. Move to custom model directory
   ```shell
   cd ./segmentation
   ```
2. Run PowerShell script (build and run docker container)
   ```shell
   .\run.ps1 `
      -dataPath "C:\Users\$env:USERNAME\Desktop\data" `
      -modelsPath "C:\Users\$env:USERNAME\Desktop\models\segmentation\artifacts" `
      -resultsPath "C:\Users\$env:USERNAME\Desktop\models\segmentation\results"
   ```
3. Start training of segmentation model
   ```shell
   ./bash/02_evaluation.sh
   ```