param(
    [int]$Port = 8080
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$datasetRoot = Join-Path $repoRoot "dataset"
$labelStudioExe = Join-Path $repoRoot ".venv-labelstudio\Scripts\label-studio.exe"

if (-not (Test-Path $labelStudioExe)) {
    throw "Label Studio environment not found at $labelStudioExe"
}

if (-not (Test-Path $datasetRoot)) {
    New-Item -ItemType Directory -Path $datasetRoot | Out-Null
}

$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED = "true"
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT = $datasetRoot

Write-Host "Label Studio local file root: $datasetRoot"
Write-Host "Source storage path: $(Join-Path $datasetRoot 'input')"
Write-Host "Target storage path: $(Join-Path $datasetRoot 'output\annotations')"
Write-Host "Starting Label Studio on port $Port..."

& $labelStudioExe start --port $Port
