param(
    [String]$dataPath,
    [String]$modelsPath
)

Write-Host "Docker image - build"
docker build . -f dataset/docker/Dockerfile -t synthetic-brain-mri:dataset-1.0.0

Write-Host "Docker image - run"
docker run `
    --rm `
    -v "${dataPath}:/data" `
    -v "${modelsPath}:/models" `
    synthetic-brain-mri:dataset-1.0.0