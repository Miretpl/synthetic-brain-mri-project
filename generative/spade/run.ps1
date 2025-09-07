param(
    [String]$dataPath,
    [String]$modelPath
)

Write-Host "Docker image - build"
docker build -t synthetic-brain-mri:spade-1.0.0 -f ./docker/Dockerfile .

Write-Host "Docker image - run"
docker run `
    --rm `
    --gpus all `
    -v "${dataPath}:/data" `
    -v "${modelPath}:/models" `
    -it `
    synthetic-brain-mri:spade-1.0.0 `
    bash