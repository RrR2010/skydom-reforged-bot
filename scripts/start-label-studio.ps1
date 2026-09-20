param(
    [int]$Port = 8080
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$labelStudioExe = Join-Path $repoRoot ".venv-labelstudio\Scripts\label-studio.exe"

if (-not (Test-Path $labelStudioExe)) {
    throw "Label Studio environment not found at $labelStudioExe"
}

$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED = "true"
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT = $repoRoot

Write-Host "Label Studio local file root: $repoRoot"
Write-Host "Dataset storage path: $(Join-Path $repoRoot 'dataset')"
Write-Host "Starting Label Studio on port $Port..."

& $labelStudioExe start --port $Port
