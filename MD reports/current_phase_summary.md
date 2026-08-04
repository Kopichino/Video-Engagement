# Video Engagement Recognition — Complete Project Phase Summary

> **Generated**: July 31, 2026  
> **Researcher**: Koppineni  
> **Hardware**: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM), 16 GB RAM  
> **Status**: **Phase 1 Complete** — OUC-CGE 5-Fold baseline training finished. Cross-dataset work not yet started.

---

## 1. Research Objective

Build and evaluate deep-learning models for **classroom engagement recognition** from video, using two complementary datasets. The long-term goal is a **cross-dataset generalization study** — testing whether engagement features learned from one classroom setting (country, camera angle, annotation granularity) transfer to another.

### Research Questions

1. Can the OUC-CGE paper's SLOW R50 results be reproduced under rigorous 5-fold cross-validation?
2. Can a model trained on Chinese group-level classroom videos (OUC-CGE) generalize to Vietnamese individual-level student data (COLER), and vice versa?
3. Can a unified model trained on both datasets outperform single-dataset models?

---

## 2. Datasets

### 2.1 OUC-CGE (OUC Classroom Group Engagement)

| Property | Value |
|----------|-------|
| **Origin** | China — Ocean University of China |
| **Total clips** | 7,439 MP4 video clips (original README claims 7,705; 142 are corrupted) |
| **Clip duration** | 10 seconds each, 30 fps |
| **Resolution** | ~1092 × 614 |
| **Granularity** | **Group-level** — entire classroom in frame |
| **Classes** | 3: **High** (2,345 clips), **Mid** (2,053 clips), **Low** (3,041 clips) |
| **Domains** | STEM and Humanities classroom sessions |
| **Stored at** | `E:\Video_Engagement\OUC-CGE-dataset\{high,mid,low}\` |

**Class imbalance**: The "Low" class has ~48% more samples than "Mid". This is addressed during training via downsampling in the split generation script.

**Corrupted files**: 142 videos failed OpenCV validation and were excluded (logged in `splits/corrupted_files.txt`). Breakdown: 84 High, 13 Medium, 45 Low.

**Known data issue fixed**: A critical **Unicode encoding bug** was discovered early in the project. ~41% of dataset directory names used full-width parentheses `（）` instead of ASCII `()`. Python's cp1252 codec on Windows couldn't resolve these paths, causing videos to be silently skipped during training. This was identified as the root cause of initial ~51% accuracy (random chance for 3 classes). All affected paths are now handled correctly by the pipeline.

### 2.2 COLER (Collaborative Learning Engagement Recognition)

| Property | Value |
|----------|-------|
| **Origin** | Vietnam |
| **Total samples** | 3,924 individual-level annotated clips (18,546 total files across modalities) |
| **Clip duration** | 5 seconds, 15 uniformly sampled frames per clip |
| **Granularity** | **Individual-level** — cropped per-student regions |
| **Classes** | 4: **Very Engaged** (903), **Engaged** (1,052), **Normal** (807), **Disengaged** (1,162) |
| **Modalities** | RGB frames (PNG) + Skeleton CSVs + Emotion crops (JPG) + Context masks (SAM) + ResNet context features |
| **Recording** | HD 720p, 30fps, 5 camera angles (Goc1–Goc5), 3 table layouts per angle |
| **Annotation** | 5 independent annotators, majority vote, validated by 498 professional educators |
| **Stored at** | `E:\Video_Engagement\Ver4-20260630T020728Z-3-005\Ver4\` |

**Benchmark model (from paper)**: CORE-Net achieved **89.63% accuracy**, **89.34% F1**, **94.80 QWK** on COLER using a multimodal architecture (BlockGCN + MoViNet + Facial encoder + Context encoder with SAM segmentation).

### 2.3 Label Taxonomy Mismatch

| OUC-CGE (3 classes, group) | COLER (4 classes, individual) | Proposed 3-Class Mapping |
|----------------------------|-------------------------------|--------------------------|
| High                       | Very Engaged                  | **High** ← High / (VeryEngaged + Engaged) |
| Mid                        | Engaged                       | **Mid** ← Mid / Normal |
| Low                        | Normal                        | **Low** ← Low / Disengaged |
| —                          | Disengaged                    | — |

An alternative **binary mapping** (Engaged vs. Not Engaged) is also documented but not yet implemented.

---

## 3. Repository Structure

```
E:\Video_Engagement\
│
├── OUC-CGE-dataset/                    # Raw video dataset (7,439 MP4s)
│   ├── high/                           # ~2,345 high-engagement clips
│   ├── mid/                            # ~2,053 medium-engagement clips
│   └── low/                            # ~3,041 low-engagement clips
│
├── Ver4-20260630T020728Z-3-005/        # COLER dataset (extracted from 1.9 GB zip)
│   └── Ver4/
│       └── Goc{1-5}/Table{1-3}/        # Camera angles × table layouts
│           ├── Context/                # ResNet context features
│           ├── Context_Mask/           # SAM segmentation masks
│           ├── Disengagement/          # Disengaged student data
│           ├── Engagement/             # Engaged student data
│           ├── Normal/                 # Normal engagement data
│           └── Very engagement/        # Very engaged student data
│
├── OUC-CGE/                            # SlowFast framework + custom pipeline
│   ├── configs/Kinetics/
│   │   └── 5fold_SLOW_8x8_R50.yaml    # Custom training config (20 epochs)
│   ├── splits/                         # Auto-generated 5-fold CSVs
│   │   ├── corrupted_files.txt         # 142 corrupted files log
│   │   ├── seed.txt                    # Random seed: 37247
│   │   └── fold_{0-4}/                 # train.csv + val.csv per fold
│   ├── work_dirs/                      # Training outputs
│   │   ├── training_progress.json      # Fold completion tracker
│   │   ├── 5fold_summary.txt           # ★ Final consolidated results
│   │   └── fold_{0-4}/                 # Checkpoints + logs per fold
│   ├── slowfast/                       # Facebook SlowFast codebase
│   │   ├── models/                     # ResNet, MViT, SlowFast builders
│   │   ├── datasets/                   # Kinetics dataloader (used for our data)
│   │   ├── config/                     # Default config system
│   │   ├── utils/                      # Checkpointing, distributed, logging
│   │   └── visualization/             # Vis utilities
│   ├── tools/                          # Runner scripts
│   │   ├── train_net.py                # Main training entry point
│   │   ├── test_net.py                 # Testing/inference script
│   │   └── run_net.py                  # CLI dispatcher
│   ├── ouc_env/                        # Python 3.12 virtual environment
│   ├── generate_fold_splits.py         # ★ Custom: fold split generator
│   ├── run_5fold.py                    # ★ Custom: 5-fold training orchestrator
│   ├── train_5fold_resumable.py        # ★ Custom: pause/resume training script
│   ├── consolidate_metrics.py          # ★ Custom: post-training metrics calculator
│   ├── cal_metrics.py                  # Metrics calculation utilities
│   ├── train.csv / val.csv / test.csv  # Original paper's 8:1:1 split (not used)
│   ├── training_output.log             # Full stdout log (2.1 MB)
│   └── training_error.log              # stderr log
│
├── eda_results/                        # EDA visualization outputs
│   ├── metadata.csv                    # Extracted video metadata (1.1 MB)
│   ├── class_distribution.png          # Class balance chart
│   ├── duration_analysis.png           # Video duration distribution
│   ├── fps_analysis.png                # Frame rate analysis
│   ├── resolution_analysis.png         # Resolution distribution
│   ├── frame_count_analysis.png        # Frame count distribution
│   ├── file_size_analysis.png          # File size distribution
│   ├── motion_analysis.png             # Optical flow / motion analysis
│   ├── face_detection_analysis.png     # Face detection results
│   ├── frame_level_analysis.png        # Frame-level feature analysis
│   └── visual_samples.png             # Sample frames grid (13.6 MB)
│
├── OUC_CGE_EDA.ipynb                   # ★ OUC-CGE exploratory data analysis (9.8 MB)
├── COLER_EDA.ipynb                     # ★ COLER exploratory data analysis (1.0 MB)
├── preprocessing.ipynb                 # Data preprocessing notebook
├── ouc_cge_5fold_splits.csv            # Pre-computed fold assignments (3.0 MB)
├── gpu_testing.py                      # PyTorch/TensorFlow GPU verification
│
├── COLER_study.md                      # Detailed COLER dataset & CORE-Net study notes
├── research_progress_and_plan.md       # Research roadmap with weekly timeline
├── implementation_plan.md              # 5-fold pipeline technical design
├── OUC_CGE&COLER_implementation plan   # Cross-dataset research strategy
├── cv_training_explanation.md          # Tutorial: folds, epochs, iterations
├── pipeline_execution_guide.md         # Step-by-step PowerShell execution guide
├── walkthrough.md                      # Training session walkthrough
├── project_summary.md                  # Earlier project summary
│
├── .gitignore                          # Excludes datasets, checkpoints, envs
└── Ver4-20260630T020728Z-3-005.zip     # COLER raw archive (1.9 GB)
```

---

## 4. Custom Pipeline Scripts

Four custom Python scripts were developed to automate the 5-fold cross-validation workflow:

### 4.1 `generate_fold_splits.py`
- Scans `OUC-CGE-dataset/{high,mid,low}/` recursively for all `.mp4` files
- Validates each video with OpenCV — skips 142 corrupted files
- Applies `StratifiedKFold` (5 splits) to ensure proportional class distribution
- **Downsamples** majority classes so each class has exactly the same count (2,053 = the minimum class count)
- Writes space-separated CSVs in SlowFast Kinetics format: `<absolute_path> <label>` (no header)
- Uses a fixed random seed (`37247`) for reproducibility; time-based seeds supported for repeated CV runs
- **Output**: `splits/fold_{0-4}/train.csv` and `splits/fold_{0-4}/val.csv`

### 4.2 `run_5fold.py`
- Original 5-fold orchestrator
- Loops folds 0–4, configuring `DATA.PATH_TO_DATA_DIR`, `DATA.PATH_PREFIX`, and `OUTPUT_DIR` per fold
- Calls SlowFast's `train()` function directly via Python API (no subprocess)
- Supports CLI arguments: `--max-epochs`, `--eval-period`, `--checkpoint-period`, `--folds`

### 4.3 `train_5fold_resumable.py`
- Enhanced version with **pause/resume** support
- Create a file named `PAUSE` in the OUC-CGE directory to pause between folds
- Automatically resumes from the last checkpoint if re-run without `--fresh`
- Tracks progress in `work_dirs/training_progress.json` (fold status, best accuracy, best epoch)
- Supports `--fresh` flag to wipe old checkpoints and start from scratch

### 4.4 `consolidate_metrics.py`
- Post-training evaluation script
- Loads the best checkpoint per fold, runs full validation inference
- Computes **Accuracy**, **Macro F1-Score**, and **Quadratic Weighted Kappa (QWK)**
- Uses standalone NumPy implementations (no scikit-learn dependency at inference time)
- Generates confusion matrices per fold
- Outputs results to `work_dirs/5fold_summary.txt`

---

## 5. Model & Training Configuration

### 5.1 Model Architecture

| Parameter | Value |
|-----------|-------|
| **Architecture** | SLOW 8×8 R50 (3D ResNet-50, slow pathway only) |
| **Backbone** | ResNet-50 with 3D convolutional kernels |
| **Temporal input** | 8 frames sampled at rate 8 (64-frame span) |
| **Spatial input** | 224 × 224 (train), 256 × 256 (val) |
| **Classification head** | 3 classes (Low=0, Mid=1, High=2) with 50% dropout |
| **Loss function** | Cross-entropy |
| **Pretrained weights** | None (trained from scratch) |

### 5.2 Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| **Optimizer** | SGD (momentum=0.9, weight_decay=1e-4, Nesterov=True) |
| **Learning rate** | Base LR=0.1, cosine annealing schedule |
| **Warmup** | 5 epochs, starting LR=0.01 |
| **Epochs per fold** | 20 |
| **Batch size** | 8 (reduced from 16 to fit 8 GB VRAM) |
| **Mixed precision** | FP16 enabled (leverages RTX 4060 Tensor Cores) |
| **Data workers** | 0 (set to 0 to avoid Windows multiprocessing issues) |
| **Validation** | Every 5 epochs |
| **Checkpointing** | Every 5 epochs (~242 MB per checkpoint) |
| **Augmentation** | Random horizontal flip, random spatial crop, scale jitter [256, 320] |
| **Normalization** | ImageNet-style: mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225] |

### 5.3 Data Splits

- **Total usable videos**: 7,297 (after excluding 142 corrupted files)
- **After downsampling**: 6,159 balanced videos (2,053 per class)
- **Per fold**: ~1,231 validation samples, ~4,926 training samples
- **Split strategy**: StratifiedKFold with seed 37247

---

## 6. Training Results — OUC-CGE 5-Fold Baseline (COMPLETE ✅)

All 5 folds completed between **July 23–24, 2026** (approximately 24 hours total).

### 6.1 Per-Fold Summary

| Fold | Val Accuracy | Macro-F1 | QWK | Best Train Acc (epoch 20) | Val Size |
|------|-------------|----------|------|--------------------------|----------|
| 0 | 51.14% | 50.33% | 0.2979 | 52.04% | 1,224 |
| 1 | 54.00% | 53.32% | 0.3417 | 54.98% | 1,224 |
| 2 | 55.07% | 53.32% | 0.4135 | 55.88% | 1,224 |
| 3 | **59.64%** | **59.06%** | **0.5067** | 60.05% | 1,224 |
| 4 | 56.13% | 54.97% | 0.4281 | 56.70% | 1,224 |
| **Average** | **55.20%** | **54.20%** | **0.3976** | — | — |
| **Std Dev** | ±2.77% | ±2.85% | ±0.0723 | — | — |

### 6.2 Confusion Matrices (Per Fold)

**Fold 0** (Accuracy: 51.14%):
```
         Low    Mid   High
Low      272     55     81
Mid      142    144    122
High     136     62    210
```

**Fold 3** (Best fold — Accuracy: 59.64%):
```
         Low    Mid   High
Low      270     76     62
Mid      143    174     91
High      57     65    286
```

### 6.3 Fold 0 Training Curve (Detailed)

| Epoch | Train Loss | Train Top-1 Err | LR | GPU Mem | Val Top-1 Err |
|-------|-----------|-----------------|-----|---------|---------------|
| 1 | 1.706 | 64.87% | 0.025 | 3.49 G | — |
| 5 | 1.104 | 61.23% | 0.085 | 3.50 G | **60.05%** |
| 10 | 1.065 | 56.21% | 0.050 | 3.62 G | **52.78%** |
| 15 | 1.019 | 51.33% | 0.015 | 3.73 G | **51.88%** |
| 20 | 0.979 | 48.00% | 0.000 | 3.73 G | **47.96%** |

**Observations**:
- Steady loss decrease from 1.706 → 0.979 across 20 epochs
- Consistent accuracy improvement: training error dropped from 64.87% → 48.00%
- Validation error tracked training error closely — **no overfitting observed**
- GPU memory usage: 3.49 G → 3.73 G (well within 8 GB limit)
- The model is still improving at epoch 20 — **more epochs would likely yield better results**

### 6.4 Gap vs. Original Paper

| Metric | Our Result (5-Fold CV, 20 epochs) | Original Paper (8:1:1 split, 50 epochs) |
|--------|----------------------------------|----------------------------------------|
| Accuracy | **55.20% ± 2.77%** | **97.8%** |
| Evaluation | Rigorous 5-fold CV with downsampling | Simple train/val/test split |
| Epochs | 20 | 50 |
| Pretrained | No (from scratch) | Likely pretrained on Kinetics-400 |

**Possible reasons for the gap**:
1. **Insufficient epochs** (20 vs. 50) — model was still improving at epoch 20
2. **No pretrained weights** — training from scratch vs. Kinetics-400 pretrained backbone
3. **Stricter evaluation** — 5-fold CV is harder than a single favorable split
4. **Class balancing via downsampling** — reduces effective training set size
5. **Reduced batch size** (8 vs. 16) — affects batch normalization statistics

---

## 7. EDA Completed

### 7.1 OUC-CGE EDA (`OUC_CGE_EDA.ipynb` — 9.8 MB)

A comprehensive exploratory analysis was performed, generating 10 visualization plots:

| Analysis | Key Finding |
|----------|-------------|
| **Class distribution** | Imbalanced: Low (3,041) > High (2,345) > Mid (2,053) |
| **Duration** | All clips ~10 seconds as expected |
| **FPS** | Consistent 30 fps across all clips |
| **Resolution** | ~1092 × 614 (uniform) |
| **Frame count** | ~300 frames per clip |
| **File size** | Varies by content complexity |
| **Motion (optical flow)** | Analyzed inter-frame motion patterns |
| **Face detection** | Face visibility analysis across engagement levels |
| **Visual samples** | Sample frames from each engagement class |

All outputs saved to `eda_results/` with a full `metadata.csv` (1.1 MB) containing per-video statistics.

### 7.2 COLER EDA (`COLER_EDA.ipynb` — 1.0 MB)

Multimodal exploratory analysis covering:
- RGB frame samples across all 4 engagement classes
- Skeleton data structure and joint distributions
- Emotion crop characteristics
- Distribution across camera angles (Goc1–5) and table configurations
- Sample counts: Very Engaged (903), Engaged (1,052), Normal (807), Disengaged (1,162)

### 7.3 COLER Study Notes (`COLER_study.md`)

Detailed study guide covering:
- CORE-Net architecture (4 modality branches: Pose/BlockGCN, Behavior/MoViNet, Facial/MoViNet, Context/SAM+ViT)
- Set Transformer for group interaction modeling
- Hierarchical multimodal attention fusion
- Ordinal-aware hybrid loss function (CE + EMD² + neighbor regularizer)
- Benchmark results: CORE-Net achieves 89.63% accuracy on COLER

---

## 8. Infrastructure & Environment

### 8.1 Hardware

| Component | Specification |
|-----------|---------------|
| **GPU** | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB VRAM |
| **RAM** | 16 GB (15.29 GB usable) |
| **OS** | Windows |
| **Storage** | Local SSD (`E:\Video_Engagement\`) |

### 8.2 Software Environment

| Component | Version/Details |
|-----------|----------------|
| **Python** | 3.12 (via Anaconda, in `OUC-CGE/ouc_env/`) |
| **PyTorch** | With CUDA support (FP16 mixed precision) |
| **Framework** | Facebook SlowFast (PyTorch-based) |
| **Decoding backend** | PyAV |
| **Key packages** | numpy, opencv-python (cv2), scikit-learn (for StratifiedKFold) |

### 8.3 Version Control

- Two `.gitignore` files configured (root + OUC-CGE)
- Excludes: raw datasets, virtual environments, checkpoints (`.pyth`), cache files, IDE settings
- The 1.9 GB COLER zip file and raw video folders are not tracked

### 8.4 Training Time Benchmarks

| Metric | Value |
|--------|-------|
| Time per epoch (1 fold) | ~18 minutes |
| Time per fold (20 epochs) | ~6 hours |
| Total 5-fold training | ~24 hours |
| Checkpoint size | ~242 MB per checkpoint file |
| Total checkpoints per fold | 4 (epochs 5, 10, 15, 20) |

---

## 9. Documentation Created

| Document | Purpose | Size |
|----------|---------|------|
| `research_progress_and_plan.md` | Master research roadmap with weekly timeline, PPT slide plan, and Claude prompt for presentation generation | 17 KB |
| `implementation_plan.md` | Technical design for the 5-fold CV pipeline | 8 KB |
| `OUC_CGE&COLER_implementation plan` | Cross-dataset research strategy with phased approach | 10 KB |
| `cv_training_explanation.md` | Tutorial explaining folds, epochs, iterations with Mermaid diagrams | 9 KB |
| `pipeline_execution_guide.md` | Step-by-step PowerShell commands for the full pipeline | 4 KB |
| `walkthrough.md` | Training session walkthrough with pause/resume instructions | 3 KB |
| `project_summary.md` | Earlier snapshot of project state | 5 KB |
| `COLER_study.md` | Deep-dive study notes on COLER dataset and CORE-Net architecture | 14 KB |

---

## 10. What Has Been Completed (Phase 1) ✅

- [x] **GPU environment setup** — PyTorch + CUDA verified on RTX 4060
- [x] **OUC-CGE dataset acquired** — 7,439 MP4 clips organized in `high/mid/low/`
- [x] **COLER dataset acquired** — Extracted from 1.9 GB zip, 18,546 files across modalities
- [x] **OUC-CGE EDA** — Full exploratory analysis with 10 visualization plots and metadata CSV
- [x] **COLER EDA** — Multimodal exploratory analysis notebook
- [x] **COLER study notes** — Detailed understanding of CORE-Net architecture and ordinal loss
- [x] **Unicode bug identified and fixed** — Full-width parentheses in directory names causing 41% data loss
- [x] **5-fold split generation pipeline** — Stratified, balanced, with corrupted file filtering
- [x] **Training configuration** — SLOW 8×8 R50 YAML tuned for 8 GB VRAM (batch 8, FP16)
- [x] **5-fold training orchestrator** — With pause/resume and progress tracking
- [x] **Metrics consolidation pipeline** — Accuracy, Macro-F1, QWK, confusion matrices
- [x] **All 5 folds trained** (20 epochs each) — Completed July 23–24, 2026
- [x] **Results consolidated** — `5fold_summary.txt` generated with full metrics
- [x] **Cross-dataset research plan** — Label mapping strategy, 3 experiment designs documented
- [x] **Presentation outline** — 12-slide PPT plan with speaker notes template

---

## 11. What Has NOT Been Done Yet (Phases 2–4) ⬜

### Phase 2: Extended OUC-CGE Baseline (Not Started)
- [ ] Train with more epochs (35–50) to close the gap with the original paper's 97.8%
- [ ] Train with **Kinetics-400 pretrained weights** (the original paper likely used pretrained)
- [ ] Experiment with other architectures: X3D (96.8%), SLOW-NLN (97.4%)
- [ ] Hyperparameter tuning: warmup schedule, learning rate, batch size effects

### Phase 3: Cross-Dataset Data Pipeline (Not Started)
- [ ] Implement 3-class label mapping for COLER
- [ ] Extract key frames from OUC-CGE videos (1 or 3 per clip)
- [ ] Build unified data loader for frame-based cross-dataset work
- [ ] Optionally reconstruct COLER video clips from PNG frames for video-level transfer

### Phase 4: Cross-Dataset Experiments (Not Started)
- [ ] **Experiment A (Frame-Level Transfer)**: Train ResNet-50 image classifier on OUC-CGE frames → test on COLER frames (and vice versa)
- [ ] **Experiment B (Video-Level Transfer)**: Train SLOW R50 on OUC-CGE → test on reconstructed COLER clips
- [ ] **Experiment C (Joint Training)**: Shared backbone + dual classification heads, trained on both datasets
- [ ] Compute cross-dataset metrics and feature visualizations (t-SNE)

### Phase 5: Analysis & Write-Up (Not Started)
- [ ] Compile all results into paper-ready tables and figures
- [ ] Write research paper sections (Introduction, Related Work, Methodology, Experiments, Results)
- [ ] Create visualizations: confusion matrices, accuracy curves, t-SNE feature maps

---

## 12. Key Metrics Reference

| Metric | Description | Current Best |
|--------|-------------|-------------|
| **Top-1 Accuracy** | % correctly classified | 55.20% ± 2.77% (5-fold avg) |
| **Macro F1-Score** | Class-balanced harmonic mean of precision & recall | 54.20% ± 2.85% |
| **QWK** | Ordinal agreement (accounts for High > Mid > Low ordering) | 0.3976 ± 0.0723 |

---

## 13. Quick Reference: Execution Commands

```powershell
# Virtual environment Python path
"E:\Video_Engagement\OUC-CGE\ouc_env\Scripts\python.exe"

# Step 1: Generate fold splits
& "E:\Video_Engagement\OUC-CGE\ouc_env\Scripts\python.exe" "E:\Video_Engagement\OUC-CGE\generate_fold_splits.py"

# Step 2: Train all 5 folds (with resume support)
& "E:\Video_Engagement\OUC-CGE\ouc_env\Scripts\python.exe" "E:\Video_Engagement\OUC-CGE\train_5fold_resumable.py"

# Step 2 (alternative): Train with custom epochs
& "E:\Video_Engagement\OUC-CGE\ouc_env\Scripts\python.exe" "E:\Video_Engagement\OUC-CGE\run_5fold.py" --max-epochs 50 --eval-period 5 --checkpoint-period 5

# Step 3: Consolidate final metrics
& "E:\Video_Engagement\OUC-CGE\ouc_env\Scripts\python.exe" "E:\Video_Engagement\OUC-CGE\consolidate_metrics.py"

# Monitor training
Get-Content "E:\Video_Engagement\OUC-CGE\training_output.log" -Tail 5
type "E:\Video_Engagement\OUC-CGE\work_dirs\training_progress.json"

# Pause/Resume training
echo. > "E:\Video_Engagement\OUC-CGE\PAUSE"       # Pause
Remove-Item "E:\Video_Engagement\OUC-CGE\PAUSE"   # Resume
```

---

## 14. Checkpoint Inventory

All 5 folds have completed checkpoints saved (4 per fold × ~242 MB each ≈ **4.8 GB total**):

| Fold | Checkpoints Available | Best Val Accuracy |
|------|----------------------|------------------|
| fold_0 | epochs 5, 10, 15, 20 | 52.04% (epoch 20) |
| fold_1 | epochs 5, 10, 15, 20 | 54.98% (epoch 20) |
| fold_2 | epochs 5, 10, 15, 20 | 55.88% (epoch 20) |
| fold_3 | epochs 5, 10, 15, 20 | 60.05% (epoch 20) |
| fold_4 | epochs 5, 10, 15, 20 | 56.70% (epoch 20) |

All folds reached their best accuracy at epoch 20, confirming the model was still improving and would benefit from longer training.

---

*End of Phase Summary — Last updated July 31, 2026*
