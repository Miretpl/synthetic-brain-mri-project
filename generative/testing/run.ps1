param(
    [String]$dataPath,
    [String]$modelsGenPath
)

Write-Host "Docker image - build"
docker build -t synthetic-brain-mri:gen-test-1.0.0 -f ./docker/Dockerfile .

Write-Host "Docker image - run"
docker run `
    --rm `
    --gpus all `
    --ipc=host `
    -v "${dataPath}:/data" `
    -v "${modelsGenPath}:/generation" `
    -it `
    synthetic-brain-mri:gen-test-1.0.0 `
    bash