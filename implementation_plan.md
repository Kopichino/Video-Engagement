# 5-Fold Cross-Validation Pipeline for OUC-CGE

Automated 5-fold cross-validation training on the OUC-CGE classroom engagement dataset using the Facebook SlowFast framework (SLOW 8x8 R50 model).

## User Review Required

> [!IMPORTANT]
> **Video Path Mapping Issue**: The `ouc_cge_5fold_splits.csv` contains **absolute paths** pointing to `E:\Video_Engagement\OUC-CGE\high\...`, but the actual video data lives in `E:\Video_Engagement\OUC-CGE-dataset\high\...`. The split generation script will remap `E:\Video_Engagement\OUC-CGE\` → `E:\Video_Engagement\OUC-CGE-dataset\` in all paths. Please confirm this mapping is correct.

> [!IMPORTANT]
> **Model Choice**: The existing repo config `SLOW_8x8_R50.yaml` achieves 97.8% accuracy per the README. I plan to use this as the base model. If you'd prefer a different architecture (e.g., `X3D`, `SlowFast`, `MViTv2`), let me know.

> [!WARNING]
> **Python Environment**: The error `ModuleNotFoundError: No module named 'pandas'` shows you ran the script with system Python 3.13, but the repo has a venv at `OUC-CGE/ouc_env` (Python 3.12.3 via Anaconda). All scripts will be written to work **without pandas** — using only Python stdlib `csv` module, avoiding any dependency issues. The training itself will use the `ouc_env` venv which should have PyTorch/SlowFast dependencies installed.

## Open Questions

> [!IMPORTANT]
> **Batch Size**: The SLOW_8x8_R50 config uses `BATCH_SIZE: 16` for training. On an Omen 16 GPU (likely RTX 4060/4070 with 8-12GB VRAM), this should work. If you experience OOM errors, we can reduce to 8. Want me to default to 16 or a lower value?

> [!IMPORTANT]
> **Epochs**: The existing config trains for 50 epochs. For 5-fold CV this means 5×50=250 total epochs. Is 50 epochs per fold acceptable, or would you prefer a different count?

---

## Proposed Changes

### 1. Split Generation Script

#### [NEW] [generate_fold_splits.py](file:///e:/Video_Engagement/OUC-CGE/generate_fold_splits.py)

A **dependency-free** Python script (stdlib `csv` only, no pandas) that:
- Reads `ouc_cge_5fold_splits.csv` (columns: `video_path,class_label,fold,phase`)
- Remaps absolute video paths → absolute paths pointing to `OUC-CGE-dataset`
- Maps labels: `Low`→0, `Medium`→1, `High`→2
- For each fold 0-4, generates two files in `E:\Video_Engagement\OUC-CGE\splits\fold_X\`:
  - `train.csv`: space-separated `<absolute_video_path> <label>` (no header) — matching the Kinetics dataset format
  - `val.csv`: same format for validation samples
- Prints summary counts for verification

---

### 2. Training Configuration

#### [NEW] [5fold_SLOW_8x8_R50.yaml](file:///e:/Video_Engagement/OUC-CGE/configs/Kinetics/5fold_SLOW_8x8_R50.yaml)

Base config for 5-fold training, derived from `SLOW_8x8_R50.yaml`:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `TRAIN.ENABLE` | `True` | Enable training (original had False) |
| `TEST.ENABLE` | `False` | Disable test-mode inference |
| `DATA.NUM_FRAMES` | 8 | Temporal sampling: 8 frames |
| `DATA.SAMPLING_RATE` | 8 | Stride between frames (8×8=64 frame span) |
| `DATA.TRAIN_CROP_SIZE` | 224 | 224×224 spatial resolution |
| `DATA.TRAIN_JITTER_SCALES` | [256, 320] | Random scale jitter for train augmentation |
| `DATA.TEST_CROP_SIZE` | 256 | Val/test spatial size |
| `DATA.MEAN` | [0.45, 0.45, 0.45] | ImageNet-style normalization |
| `DATA.STD` | [0.225, 0.225, 0.225] | ImageNet-style normalization |
| `DATA.RANDOM_FLIP` | True | Horizontal flip augmentation (train only) |
| `DATA.PATH_LABEL_SEPARATOR` | `" "` | Space separator in CSV |
| `SOLVER.MAX_EPOCH` | 50 | Epochs per fold |
| `TRAIN.EVAL_PERIOD` | 5 | Validate every 5 epochs |
| `TRAIN.CHECKPOINT_PERIOD` | 5 | Save checkpoint every 5 epochs |
| `TRAIN.BATCH_SIZE` | 16 | Batch size |
| `MODEL.NUM_CLASSES` | 3 | Low/Medium/High |

**Key preprocessing already handled by the SlowFast Kinetics pipeline:**
- ✅ **Uniform Temporal Sampling**: `decoder.decode()` uniformly samples `NUM_FRAMES` frames with `SAMPLING_RATE` stride
- ✅ **Spatial Resizing**: `spatial_sampling()` resizes to `[256, 320]` scale range, then center/random crops to 224×224
- ✅ **ImageNet Normalization**: `tensor_normalize()` with MEAN/STD values
- ✅ **Data Augmentation**: Random horizontal flip + random spatial crop for training; center crop for validation

No custom dataset code modifications needed — the existing `Kinetics` dataset class handles all this.

---

### 3. Master Runner Script

#### [NEW] [run_5fold.py](file:///e:/Video_Engagement/OUC-CGE/run_5fold.py)

The core automation script that:

1. **Loops folds 0-4**, for each:
   - Sets `DATA.PATH_TO_DATA_DIR` to `splits/fold_X/` (where `train.csv` and `val.csv` live)
   - Sets `DATA.PATH_PREFIX` to empty (since paths are absolute)
   - Sets `OUTPUT_DIR` to `work_dirs/fold_X/`
   - Uses `NUM_GPUS: 1` on `cuda:0`
   - Calls the SlowFast `train()` function directly via Python API (not subprocess)

2. **After each fold completes**, parses the best validation accuracy from the training logs

3. **After all 5 folds**, runs the metrics consolidation

**Architecture**: Uses `slowfast.config.defaults.get_cfg()` + `cfg.merge_from_file()` + `cfg.merge_from_list()` programmatically (same approach as `run_net.py`), then calls `train()` from `train_net.py`.

---

### 4. Metrics Consolidation

#### [NEW] [consolidate_metrics.py](file:///e:/Video_Engagement/OUC-CGE/consolidate_metrics.py)

Post-training script that:
1. For each fold 0-4, loads the best checkpoint and runs inference on that fold's validation set
2. Collects all predictions and true labels
3. Computes per-fold and averaged:
   - **Accuracy** (overall)
   - **Macro F1-Score** 
   - **Quadratic Weighted Kappa (QWK)** using Cohen's kappa with quadratic weights
4. Generates a formatted summary table
5. Saves results to `work_dirs/5fold_summary.txt`

> [!NOTE]
> QWK computation will use a standalone implementation (no sklearn dependency needed at training time — it's just a matrix computation). The formula: `QWK = 1 - (weighted_sum_observed / weighted_sum_expected)` with quadratic weight matrix.

---

### Output Directory Structure

```
OUC-CGE/
├── splits/
│   ├── fold_0/
│   │   ├── train.csv          # ~6100 samples
│   │   └── val.csv            # ~1500 samples  
│   ├── fold_1/
│   │   ├── train.csv
│   │   └── val.csv
│   ├── ...
│   └── fold_4/
│       ├── train.csv
│       └── val.csv
├── work_dirs/
│   ├── fold_0/
│   │   └── checkpoints/       # Auto-created by SlowFast
│   ├── fold_1/
│   │   └── checkpoints/
│   ├── ...
│   ├── fold_4/
│   │   └── checkpoints/
│   └── 5fold_summary.txt      # Final metrics report
├── configs/Kinetics/
│   └── 5fold_SLOW_8x8_R50.yaml  # Base config
├── generate_fold_splits.py     # Split generation
├── run_5fold.py                # Master runner
└── consolidate_metrics.py      # Metrics consolidation
```

---

## Verification Plan

### Automated Tests
1. Run `generate_fold_splits.py` and verify:
   - All 10 CSV files created (5 folds × 2 splits)
   - Row counts sum correctly across folds
   - Labels are {0, 1, 2} only
   - Video paths exist on disk (spot-check)

2. Dry-run `run_5fold.py` on fold 0 for 1 epoch to verify the pipeline works end-to-end

### Manual Verification
- Monitor GPU utilization during training
- Review training logs for loss convergence
- Check that checkpoints are saved to correct directories

### Execution Commands
```powershell
# Step 1: Generate fold splits
& E:\Video_Engagement\OUC-CGE\ouc_env\Scripts\python.exe E:\Video_Engagement\OUC-CGE\generate_fold_splits.py

# Step 2: Launch full 5-fold training
& E:\Video_Engagement\OUC-CGE\ouc_env\Scripts\python.exe E:\Video_Engagement\OUC-CGE\run_5fold.py

# Step 3 (after training): Consolidate metrics
& E:\Video_Engagement\OUC-CGE\ouc_env\Scripts\python.exe E:\Video_Engagement\OUC-CGE\consolidate_metrics.py
```
