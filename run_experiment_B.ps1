<#
.SYNOPSIS
    Automated Master Runner Script for Experiment B (3D Spatiotemporal Sequence Transfer Learning).
.DESCRIPTION
    Executes Experiment B Settings 1 to 4 across all 5 cross-validation folds.
#>

param (
    [switch]$Setting1Only,
    [switch]$Setting2Only,
    [switch]$AllFolds,
    [int]$Fold = 0
)

$ErrorActionPreference = "Stop"
$ProjectRoot = "E:\Video_Engagement"
Set-Location $ProjectRoot

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host " EXPERIMENT B: 3D VIDEO SEQUENCE TRANSFER PIPELINE" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# Activate Virtual Environment
$EnvPath = "$ProjectRoot\OUC-CGE\ouc_env\Scripts\Activate.ps1"
if (Test-Path $EnvPath) {
    . $EnvPath
    Write-Host "[INFO] Activated Virtual Environment: ouc_env" -ForegroundColor Green
} else {
    Write-Host "[WARNING] ouc_env not found at $EnvPath. Using default python." -ForegroundColor Yellow
}

$PyCmd = "python"

# 1. Pipeline Verification Audit
Write-Host "`n[STEP 1] Running Pipeline Integrity Validator..." -ForegroundColor Yellow
& $PyCmd -m cross_dataset.validation.pipeline_validator
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Pipeline validation failed!" -ForegroundColor Red
    exit 1
}

# 2. Setting 1: Train & Evaluate 3D Model on OUC-CGE
if (-not $Setting2Only) {
    Write-Host "`n[STEP 2] Executing Setting 1: 3D R(2+1)D-18 Model on OUC-CGE..." -ForegroundColor Yellow
    if ($AllFolds) {
        & $PyCmd train.py --config configs/experiment_B/train_ouc_3d.yaml --all-folds
    } else {
        & $PyCmd train.py --config configs/experiment_B/train_ouc_3d.yaml --fold $Fold
    }
}

# 3. Setting 2: Train & Evaluate 3D Model on COLER
if (-not $Setting1Only) {
    Write-Host "`n[STEP 3] Executing Setting 2: 3D R(2+1)D-18 Model on COLER..." -ForegroundColor Yellow
    if ($AllFolds) {
        & $PyCmd train.py --config configs/experiment_B/train_coler_3d.yaml --all-folds
    } else {
        & $PyCmd train.py --config configs/experiment_B/train_coler_3d.yaml --fold $Fold
    }
}

# 4. Setting 3: Direct Temporal Transfer (OUC 3D -> COLER Test)
Write-Host "`n[STEP 4] Executing Setting 3: Direct Temporal Transfer (OUC 3D Model -> COLER Test)..." -ForegroundColor Yellow
& $PyCmd evaluate.py --config configs/experiment_B/eval_ouc_to_coler_3d.yaml

# 5. Setting 4: Reverse Direct Temporal Transfer (COLER 3D -> OUC Test)
Write-Host "`n[STEP 5] Executing Setting 4: Reverse Direct Temporal Transfer (COLER 3D Model -> OUC Test)..." -ForegroundColor Yellow
& $PyCmd evaluate.py --config configs/experiment_B/eval_coler_to_ouc_3d.yaml

Write-Host "`n==================================================" -ForegroundColor Green
Write-Host " EXPERIMENT B PIPELINE EXECUTION COMPLETE!" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green
