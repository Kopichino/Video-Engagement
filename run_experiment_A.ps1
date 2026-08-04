# PowerShell Master Automation Script for Experiment A (Settings 1 to 4)
# Target Environment: Python in E:\Video_Engagement\OUC-CGE\ouc_env\Scripts\python.exe
# Usage:
#   .\run_experiment_A.ps1            (Runs Fold 0 fast pipeline)
#   .\run_experiment_A.ps1 -AllFolds  (Runs full 5-Fold Cross Validation pipeline)

param (
    [switch]$AllFolds
)

$PythonPath = "E:\Video_Engagement\OUC-CGE\ouc_env\Scripts\python.exe"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " EXECUTING EXPERIMENT A: FRAME-LEVEL TRANSFER (20 EPOCHS) " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

if ($AllFolds) {
    Write-Host "[MODE] Full 5-Fold Stratified Cross-Validation Enabled!" -ForegroundColor Green
    $FoldFlag = "--all-folds"
} else {
    Write-Host "[MODE] Single-Fold Execution (Fold 0)" -ForegroundColor Yellow
    $FoldFlag = "--fold 0"
}

# Setting 1: Train & Evaluate on OUC-CGE
Write-Host "`n[STEP 1/4] Running Setting 1: Train & Evaluate on OUC-CGE..." -ForegroundColor Yellow
& $PythonPath train.py --config configs/experiment_A/train_ouc.yaml $FoldFlag
if ($LASTEXITCODE -ne 0) {
    Write-Host "Setting 1 failed! Aborting pipeline." -ForegroundColor Red
    exit 1
}

# Setting 2: Train & Evaluate on COLER
Write-Host "`n[STEP 2/4] Running Setting 2: Train & Evaluate on COLER..." -ForegroundColor Yellow
& $PythonPath train.py --config configs/experiment_A/train_coler.yaml $FoldFlag
if ($LASTEXITCODE -ne 0) {
    Write-Host "Setting 2 failed! Aborting pipeline." -ForegroundColor Red
    exit 1
}

# Setting 3: Direct Transfer - OUC Model evaluated on COLER Test Set
Write-Host "`n[STEP 3/4] Running Setting 3: Direct Transfer (OUC Model -> COLER Test Set)..." -ForegroundColor Yellow
& $PythonPath evaluate.py --config configs/experiment_A/eval_ouc_to_coler.yaml
if ($LASTEXITCODE -ne 0) {
    Write-Host "Setting 3 failed! Aborting pipeline." -ForegroundColor Red
    exit 1
}

# Setting 4: Direct Transfer - COLER Model evaluated on OUC-CGE Test Set
Write-Host "`n[STEP 4/4] Running Setting 4: Direct Transfer (COLER Model -> OUC-CGE Test Set)..." -ForegroundColor Yellow
& $PythonPath evaluate.py --config configs/experiment_A/eval_coler_to_ouc.yaml
if ($LASTEXITCODE -ne 0) {
    Write-Host "Setting 4 failed! Aborting pipeline." -ForegroundColor Red
    exit 1
}

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host " EXPERIMENT A (SETTINGS 1 - 4) COMPLETED SUCCESSFULLY!    " -ForegroundColor Green
Write-Host " All Checkpoints, Logs, & Predictions Saved in outputs/   " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
