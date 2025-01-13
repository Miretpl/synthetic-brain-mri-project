param(
    [String]$dataPath,
    [String]$configPath,
    [String]$artifactPath,
    [String]$modelPath,
    [String]$resultPath
)

Write-Host "Docker image - build"
docker build -t synthetic-brain-mri:controlnet-1.0.0 -f ./docker/Dockerfile .

Write-Host "Docker image - run"
docker run `
    --rm `
    --gpus all `
    --ipc=host `
    -v "${dataPath}:/data" `
    -v "${configPath}:/config" `
    -v "${artifactPath}:/project/mlruns" `
    -v "${modelPath}:/project/outputs/runs" `
    -v "${resultPath}:/results" `
    -it `
    synthetic-brain-mri:controlnet-1.0.0 `
    bash