param(
    [String]$dataPath,
    [String]$modelsPath
)

Write-Host "Docker image - build"
docker build . -f docker/Dockerfile -t synthetic-brain-mri:segmentation-1.0.0

Write-Host "Docker image - run"
docker run `
    --rm `
    --gpus all `
    --ipc=host `
    -v "${dataPath}:/data" `
    -v "${modelsPath}:/models" `
    -it `
    synthetic-brain-mri:segmentation-1.0.0 `
    bash