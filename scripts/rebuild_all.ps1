param(
    [string]$StartDate = "2006-01-01",
    [string]$EndDate = "",
    [switch]$SkipDownload,
    [switch]$CoreOnly,
    [switch]$IsolateBars
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

function Invoke-QuantStep {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Program,
        [Parameter(Mandatory = $true)][AllowEmptyCollection()][string[]]$Arguments
    )

    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed with exit code ${LASTEXITCODE}: $Name"
    }
}

$requiredCommands = @("python", "quant-free-lake", "quant-materialize", "quant-research")
foreach ($command in $requiredCommands) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        throw "Missing command '$command'. Activate the environment and run: pip install -e `".[dev]`""
    }
}

$logDir = Join-Path $repoRoot "run_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logDir "rebuild_$timestamp.log"
Start-Transcript -Path $logPath | Out-Null

try {
    Invoke-QuantStep "Unit and regression tests" "python" @("-m", "pytest", "-q")

    if (-not $SkipDownload) {
        $downloadArgs = @("bootstrap", "--start", $StartDate)
        $crosscheckArgs = @("crosscheck", "--start", $StartDate)
        $verifyArgs = @("verify", "--start", $StartDate)
        if ($EndDate) {
            $downloadArgs += @("--end", $EndDate)
            $crosscheckArgs += @("--end", $EndDate)
            $verifyArgs += @("--end", $EndDate)
        }
        if ($IsolateBars) {
            $downloadArgs += "--isolate-bars"
        }
        Invoke-QuantStep "BaoStock PIT snapshots and daily bars" "quant-free-lake" $downloadArgs
        Invoke-QuantStep "Download completeness gate" "quant-free-lake" $verifyArgs
        Invoke-QuantStep "AKShare deterministic cross-check" "quant-free-lake" $crosscheckArgs
    }

    $coreSteps = @(
        @{ Name = "Quality, factor and label materialization"; Program = "quant-materialize"; Args = @("--missing") },
        @{ Name = "Single-factor research"; Program = "quant-research"; Args = @("factor-research") },
        @{ Name = "Eligible-panel staging"; Program = "quant-preprocess"; Args = @("--stage") },
        @{ Name = "Leakage-safe cross-sectional preprocessing"; Program = "quant-preprocess"; Args = @("--all") },
        @{ Name = "Feature diagnostics"; Program = "quant-feature-analysis"; Args = @() },
        @{ Name = "Ridge baseline"; Program = "quant-model-baseline"; Args = @() },
        @{ Name = "Baseline portfolio"; Program = "quant-portfolio"; Args = @() },
        @{ Name = "Liquidity stress"; Program = "quant-liquidity-stress"; Args = @() },
        @{ Name = "Turnover buffer"; Program = "quant-turnover-buffer"; Args = @() },
        @{ Name = "ADV exposures"; Program = "quant-capacity"; Args = @("build-exposures") },
        @{ Name = "Capacity analysis"; Program = "quant-capacity"; Args = @("analyze") },
        @{ Name = "Capacity execution simulation"; Program = "quant-capacity-execution"; Args = @() },
        @{ Name = "HGB challenger"; Program = "quant-model-challenger"; Args = @() },
        @{ Name = "Challenger portfolio"; Program = "quant-challenger-portfolio"; Args = @() },
        @{ Name = "Model interpretation"; Program = "quant-model-interpretation"; Args = @() },
        @{ Name = "Size stress exposures"; Program = "quant-size-stress"; Args = @("build-exposures") },
        @{ Name = "Size stress analysis"; Program = "quant-size-stress"; Args = @("analyze") },
        @{ Name = "Execution flags"; Program = "quant-execution-constraints"; Args = @("build-flags") },
        @{ Name = "Execution constraint analysis"; Program = "quant-execution-constraints"; Args = @("analyze") },
        @{ Name = "Robustness suite"; Program = "quant-robustness"; Args = @() }
    )
    foreach ($step in $coreSteps) {
        Invoke-QuantStep $step.Name $step.Program $step.Args
    }

    if (-not $CoreOnly) {
        $extendedSteps = @(
            @{ Name = "Preprocessing ablation"; Program = "quant-factor-preprocessing-ablation"; Args = @() },
            @{ Name = "Locked preprocessing historical OOS"; Program = "quant-factor-preprocessing-oos"; Args = @() },
            @{ Name = "Multi-horizon label panels"; Program = "quant-label-panel"; Args = @("all") },
            @{ Name = "IC decay"; Program = "quant-ic-decay"; Args = @() },
            @{ Name = "Leave-one-factor-out"; Program = "quant-lofo"; Args = @() },
            @{ Name = "Label horizon study"; Program = "quant-label-horizon"; Args = @() },
            @{ Name = "PIT size exposure"; Program = "quant-risk-exposure"; Args = @("build-size") },
            @{ Name = "Size neutralization"; Program = "quant-size-neutralization"; Args = @() },
            @{ Name = "Unified model benchmark"; Program = "quant-model-benchmark"; Args = @() },
            @{ Name = "RF hyperparameter stability"; Program = "quant-rf-hyperparameter-stability"; Args = @() },
            @{ Name = "RF return forensics"; Program = "quant-rf-return-forensics"; Args = @() },
            @{ Name = "RF next-day execution audit"; Program = "quant-rf-oos-execution-audit"; Args = @() },
            @{ Name = "RF target holdings"; Program = "quant-rf-target-holdings"; Args = @() },
            @{ Name = "RF daily retry execution"; Program = "quant-rf-daily-retry"; Args = @("all", "--split", "both") }
        )
        foreach ($step in $extendedSteps) {
            Invoke-QuantStep $step.Name $step.Program $step.Args
        }
    }

    Invoke-QuantStep "Research dashboard" "quant-research-dashboard" @()
    Invoke-QuantStep "Publication audit" "quant-audit" @()
    Write-Host "`nRebuild completed. Log: $logPath" -ForegroundColor Green
}
finally {
    Stop-Transcript | Out-Null
}
