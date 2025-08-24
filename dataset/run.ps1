param(
    [String]$dataPath
)

Write-Host "Docker image - build"
docker build . -f docker/Dockerfile -t synthetic-brain-mri:dataset-1.0.0

Write-Host "Docker image - run"
docker run `
    --rm `
    -v "${dataPath}:/data" `
    synthetic-brain-mri:dataset-1.0.0