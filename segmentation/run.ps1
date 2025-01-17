param(
    [String]$dataPath,
    [String]$modelsPath,
    [String]$resultsPath
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
    -v "${resultsPath}:/results" `
    -it `
    synthetic-brain-mri:segmentation-1.0.0 `
    bash