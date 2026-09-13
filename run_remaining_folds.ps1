$ErrorActionPreference = "Continue"
Write-Host "=================================================="
Write-Host " CONTINUING FOLD 3 (AUTO-RESUMING FROM EPOCH 14) "
Write-Host "=================================================="
& "E:\Video_Engagement\OUC-CGE\ouc_env\Scripts\python.exe" -u train.py --config configs/experiment_B/train_ouc_3d.yaml --fold 3
$fold3_exit = $LASTEXITCODE

if ($fold3_exit -eq 0) {
    Write-Host "=================================================="
    Write-Host " FOLD 3 COMPLETE! STARTING FOLD 4 TRAINING...     "
    Write-Host "=================================================="
    & "E:\Video_Engagement\OUC-CGE\ouc_env\Scripts\python.exe" -u train.py --config configs/experiment_B/train_ouc_3d.yaml --fold 4
    $fold4_exit = $LASTEXITCODE
    Write-Host "Fold 4 completed with exit code: $fold4_exit"
} else {
    Write-Host "Fold 3 exited with code: $fold3_exit"
}
