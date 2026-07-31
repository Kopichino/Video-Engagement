# Training Walkthrough: 5-Fold CV with Pause/Resume

## What's Running

Training is launched as a **background task** using the new script [train_5fold_resumable.py](file:///e:/Video_Engagement/OUC-CGE/train_5fold_resumable.py).

| Setting | Value |
|---------|-------|
| **Model** | SLOW 8x8 R50 (ResNet-50 3D CNN) |
| **Epochs** | 25 per fold |
| **Folds** | 0, 1, 2, 3, 4 (all 5) |
| **GPU** | NVIDIA GeForce RTX 4060 Laptop GPU |
| **Batch Size** | 8 |
| **Mixed Precision** | FP16 enabled |
| **Checkpoint** | Every 5 epochs |
| **Validation** | Every 5 epochs |
| **Mode** | FRESH START (old fold_0 checkpoints deleted) |

## How to Pause Training

While training is running, create a file called `PAUSE` in the OUC-CGE directory:

```powershell
# PAUSE training:
echo. > "E:\Video_Engagement\OUC-CGE\PAUSE"
```

The script checks for this file between folds. Training will halt after the current fold's active epoch completes and wait until the file is removed.

## How to Resume Training

```powershell
# 1. Delete the PAUSE file:
Remove-Item "E:\Video_Engagement\OUC-CGE\PAUSE"

# 2. Re-run the script (it auto-resumes from last checkpoint):
& "E:\Video_Engagement\OUC-CGE\ouc_env\Scripts\python.exe" "E:\Video_Engagement\OUC-CGE\train_5fold_resumable.py"
```

> [!IMPORTANT]
> The `--fresh` flag is only needed the FIRST time. After that, just run **without** `--fresh` to resume.

## How to Check Progress

```powershell
# Check the progress tracker:
type "E:\Video_Engagement\OUC-CGE\work_dirs\training_progress.json"

# Check fold 0 training stats:
type "E:\Video_Engagement\OUC-CGE\work_dirs\fold_0\json_stats.log"

# Count completed epochs for a fold:
dir "E:\Video_Engagement\OUC-CGE\work_dirs\fold_0\checkpoints"
```

## After Training Completes

Run the metrics consolidation to get final accuracy, F1, and QWK scores:

```powershell
& "E:\Video_Engagement\OUC-CGE\ouc_env\Scripts\python.exe" "E:\Video_Engagement\OUC-CGE\consolidate_metrics.py"
```

## Checkpoint Structure

```
work_dirs/
├── fold_0/
│   ├── checkpoints/
│   │   ├── checkpoint_epoch_00005.pyth   (242 MB each)
│   │   ├── checkpoint_epoch_00010.pyth
│   │   ├── checkpoint_epoch_00015.pyth
│   │   ├── checkpoint_epoch_00020.pyth
│   │   └── checkpoint_epoch_00025.pyth
│   ├── json_stats.log    (epoch-level metrics)
│   └── stdout.log        (detailed training output)
├── fold_1/ ... fold_4/   (same structure)
└── training_progress.json (overall progress tracker)
```

## Known Issue: "Failed to load video" Messages

These are **not errors** — they are INFO-level messages from the SlowFast framework for files with Unicode characters (full-width parentheses `（）`) in directory names. The framework automatically retries with other samples. The previous training run had the same messages and ran successfully for 12+ epochs.
