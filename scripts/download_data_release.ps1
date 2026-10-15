param(
    [string]$Release = "data-v1",
    [string]$Destination = "data-release"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$AssetDir = Join-Path $RepoRoot $Destination
New-Item -ItemType Directory -Force -Path $AssetDir | Out-Null

gh release download $Release --repo yuhaoyan04/ai-quant-research --dir $AssetDir
python (Join-Path $PSScriptRoot "data_release.py") restore `
    --repo-root $RepoRoot `
    --asset-dir $AssetDir `
    --manifest (Join-Path $AssetDir "data_release_manifest.json")
