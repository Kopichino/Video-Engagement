# Cross-Dataset Video Engagement Analysis — Progress Report & Research Plan

---

## 1. Project Overview

**Research Goal**: Build and evaluate video-based classroom engagement recognition models using two complementary datasets, and study cross-dataset generalization.

| Item | Details |
|------|---------|
| **Researcher** | Koppineni (RTX 4060 Laptop, 8GB VRAM) |
| **Dataset 1** | OUC-CGE — 7,439 video clips, 3 classes (High/Mid/Low), group-level classroom engagement, China |
| **Dataset 2** | COLER — 18,546 samples (PNG frames + skeleton CSVs + emotion crops), 4 classes (Very Engaged/Engaged/Normal/Disengaged), individual-level, Vietnam |
| **Model** | SLOW 8×8 R50 (SlowFast framework), ResNet-50 backbone, 3D convolutions for temporal modeling |
| **Training** | 5-Fold Cross-Validation, 20 epochs per fold, SGD + cosine LR decay, mixed precision (FP16) |

---

## 2. Current Training Status (as of July 24, 2026)

### 2.1 Bug Fix Applied
Before training, we discovered and fixed a **critical Unicode encoding bug** — 41% of training data paths used full-width parentheses `（）` that Python's cp1252 codec couldn't resolve. Files were silently skipped. This was the reason the earlier 1-epoch run gave ~51% accuracy (random chance). All directory names and CSV paths have been corrected.

### 2.2 Fold 0 Results (Completed)

| Validation Checkpoint | Top-1 Error | **Accuracy** |
|-----------------------|-------------|-------------|
| Epoch 5 (warmup just ended) | 60.05% | 39.95% |
| Epoch 10 | 52.78% | 47.22% |
| Epoch 15 | 51.88% | 48.12% |
| **Epoch 20 (final)** | **47.96%** | **52.04%** |

**Observation**: Steady improvement as cosine LR schedule decays. Model is learning, but 20 epochs with training from scratch (no pretrained weights) is still early. The original paper achieved 97.8% with 50 epochs on a simpler train/test split (not 5-fold CV).

### 2.3 Fold 1 (In Progress)

| Validation Checkpoint | Top-1 Error | Accuracy |
|-----------------------|-------------|----------|
| Epoch 5 | 61.60% | 38.40% |
| Epoch 10–20 | *(training in progress)* | — |

### 2.4 Folds 2–4: Queued

**Estimated completion**: ~18 min/epoch × 20 epochs × 3 remaining folds ≈ **18 hours** from now.

### 2.5 Training Configuration

```yaml
Model: SLOW 8×8 R50 (ResNet-50 with slow pathway, 8-frame input, sampling rate 8)
Batch Size: 8
Optimizer: SGD (momentum=0.9, weight_decay=1e-4, nesterov=True)
LR Schedule: Cosine (base_lr=0.1, warmup_epochs=5, warmup_start_lr=0.01)
Epochs: 20 per fold
Mixed Precision: FP16 enabled
Validation: Every 5 epochs
Checkpoints: Every 5 epochs
Pretrained Weights: None (training from scratch)
```

---

## 3. Research Contributions (Proposed)

### Contribution 1: OUC-CGE Baseline Reproduction
Reproduce the SLOW R50 results from the OUC-CGE paper using **rigorous 5-fold cross-validation** (the original paper used a simpler train/val/test split). This establishes a more robust baseline.

### Contribution 2: Cross-Dataset Transfer Study
Train on OUC-CGE, evaluate on COLER (and vice versa). This tests whether engagement features generalize across:
- **Countries**: China vs. Vietnam
- **Granularity**: Group-level vs. individual-level
- **Classroom layouts**: Different physical environments

### Contribution 3: Unified Multi-Dataset Engagement Model
Fine-tune a single backbone on both datasets simultaneously using dataset-specific classification heads.

---

## 4. Cross-Dataset Validation Strategy

### 4.1 The Challenge: Different Label Taxonomies

| OUC-CGE (3 classes, group-level) | COLER (4 classes, individual-level) |
|----------------------------------|-------------------------------------|
| High | Very Engaged |
| Mid | Engaged |
| Low | Normal |
| — | Disengaged |

### 4.2 Label Normalization — 3-Class Mapping (Recommended)

| Unified Label | OUC-CGE Mapping | COLER Mapping | Justification |
|---------------|-----------------|---------------|---------------|
| **High** | High | Very Engaged + Engaged | Both represent active, positive engagement |
| **Mid** | Mid | Normal | Neutral/passive engagement |
| **Low** | Low | Disengaged | Lack of engagement |

**Why 3 classes?** Matches the ordinal engagement scale (High/Mid/Low) standard in the literature. Merging "Very Engaged" + "Engaged" is defensible because both indicate positive participation.

**Alternative — Binary (2-class):**
| Unified Label | OUC-CGE | COLER |
|---------------|---------|-------|
| Engaged | High | Very Engaged + Engaged |
| Not Engaged | Mid + Low | Normal + Disengaged |

### 4.3 The Modality Gap

| Property | OUC-CGE | COLER |
|----------|---------|-------|
| Input format | Raw MP4 video clips (10s, 30fps) | Pre-extracted PNG frames (15 per 5s clip) + skeleton CSVs + emotion crops |
| Granularity | Whole classroom (group) | Individual student (cropped) |
| Resolution | ~1092×614 | Varies (cropped regions) |
| Temporal | 300 frames per clip | 15 sampled frames per clip |

**Key constraint**: You cannot directly feed COLER frames into SlowFast (it expects video files). Solutions below.

### 4.4 Cross-Dataset Experiment Plan

#### Experiment A: Frame-Level Transfer (Fastest Path — Recommended First)
1. **Extract key frames from OUC-CGE**: Sample 1 or 3 frames (start/mid/end) per video clip
2. **Use COLER frames directly**: Already extracted as PNGs
3. **Train a ResNet-50 (2D) image classifier** on frames from Dataset A, test on frames from Dataset B
4. **Model**: Standard ResNet-50 or ViT-B/16 pretrained on ImageNet, fine-tuned for engagement

**Why this works**: Removes the temporal dimension, making both datasets directly comparable as image classification tasks. Fast to implement and train.

#### Experiment B: Video-Level Transfer (More Rigorous)
1. **Reconstruct COLER video clips**: Stitch the 15 PNG frames per sample into short video clips (e.g., 5s at 3fps)
2. **Train SlowFast on OUC-CGE → test on reconstructed COLER clips**
3. **Train SlowFast on COLER clips → test on OUC-CGE**

**Why this matters**: Tests whether the 3D temporal features learned by SlowFast transfer across datasets.

#### Experiment C: Multi-Dataset Joint Training (Most Novel)
1. Create a combined dataloader that alternates batches between datasets
2. Use a **shared ResNet-50 backbone** with **two classification heads** (3-class for OUC-CGE, 4-class for COLER)
3. Train jointly, report performance on each dataset's test set
4. Compare joint vs. single-dataset training

### 4.5 Recommended Model Architecture for Cross-Dataset Work

| Experiment | Model | Why |
|-----------|-------|-----|
| Frame-level transfer | **ResNet-50 (ImageNet pretrained)** | Works on both datasets as images; fast training; strong baseline |
| Video-level transfer | **SLOW R50 (current model)** | Already configured for OUC-CGE; needs COLER video reconstruction |
| Joint training | **ResNet-50 backbone + dual heads** | Shared features with dataset-specific classifiers |
| Advanced (if time) | **ViT-B/16 (ImageNet pretrained)** | State-of-the-art image features; strong transfer learning |

---

## 5. Forward Path — Week-by-Week Timeline

### Week 1 (Current): OUC-CGE Baseline
- [x] Fix Unicode encoding bug in dataset paths
- [x] Configure 5-fold CV pipeline (20 epochs, FP16, cosine LR)
- [x] Start fold 0 training → completed (52.04% val accuracy)
- [/] Folds 1–4 training (in progress, ~18 hours remaining)
- [ ] Run `consolidate_metrics.py` for full results (Accuracy, Macro-F1, QWK, confusion matrices)
- [ ] Analyze results: Is 20 epochs enough? Consider extending to 35–50 if accuracy plateaus

### Week 2: Cross-Dataset Data Pipeline
- [ ] Implement 3-class label mapping for COLER
- [ ] Extract key frames from OUC-CGE videos (1 or 3 per clip)
- [ ] Build unified data loader for both datasets
- [ ] Optionally reconstruct COLER video clips from PNG frames

### Week 3: Cross-Dataset Experiments
- [ ] **Experiment A**: Train ResNet-50 on OUC-CGE frames → test on COLER frames (and vice versa)
- [ ] **Experiment B**: Train SLOW R50 on OUC-CGE videos → test on reconstructed COLER clips
- [ ] **Experiment C**: Joint training with dual classification heads
- [ ] Compute cross-dataset metrics: Accuracy, Macro-F1, confusion matrices

### Week 4: Analysis & Write-Up
- [ ] Compile all results into tables and figures
- [ ] Write research paper sections: Introduction, Related Work, Methodology, Experiments, Results
- [ ] Create visualizations: t-SNE of learned features, confusion matrices, accuracy curves

---

## 6. PPT Slide Plan (for Professor Presentation)

### Slide 1: Title Slide
**"Cross-Dataset Video Engagement Analysis: OUC-CGE and COLER"**
- Your name, university, date
- Advisor name

### Slide 2: Research Motivation
- Classroom engagement recognition is critical for education quality
- Existing models are trained and tested on single datasets → unknown generalization
- **Research question**: *Can engagement features learned from one classroom setting transfer to another?*
- Two complementary datasets: OUC-CGE (China, group-level) and COLER (Vietnam, individual-level)

### Slide 3: Dataset Overview
- Side-by-side comparison table of OUC-CGE vs COLER
- Key stats: sample counts, class distributions, modalities
- Visual examples from each dataset (sample frames)
- Highlight the **complementary nature**: different countries, granularities, modalities

### Slide 4: OUC-CGE EDA Highlights
- Class distribution bar chart: High (2,345), Mid (2,053), Low (3,041)
- Sample frames showing High/Mid/Low engagement
- Key observations: class imbalance (Low is overrepresented)

### Slide 5: COLER EDA Highlights
- Class distribution: Very Engaged, Engaged, Normal, Disengaged
- Multi-modal data: RGB frames + skeleton data + emotion crops
- Sample visualization of the three modalities

### Slide 6: Current Progress — OUC-CGE Baseline
- Model: SLOW 8×8 R50 (3D ResNet-50)
- Training: 5-Fold CV, 20 epochs, cosine LR, FP16
- Bug fix story: Unicode encoding issue causing 41% data loss → fixed
- **Fold 0 results**: 52.04% accuracy (20 epochs)
- Fold 1–4: in progress
- Comparison to original paper: 97.8% (50 epochs, simple split) vs our rigorous 5-fold CV

### Slide 7: Label Normalization Strategy
- The challenge: OUC-CGE has 3 classes, COLER has 4 classes
- Proposed 3-class mapping table
- Justification for merging "Very Engaged" + "Engaged" → "High"
- Alternative: Binary (Engaged vs. Not Engaged) as a simpler baseline

### Slide 8: Cross-Dataset Experiment Design
- **Experiment A**: Frame-level transfer (ResNet-50 image classifier)
- **Experiment B**: Video-level transfer (SlowFast with reconstructed COLER clips)
- **Experiment C**: Joint training with shared backbone + dual heads
- Diagram showing the data flow for each experiment

### Slide 9: Model Architecture for Cross-Dataset Work
- Diagram: Shared ResNet-50 backbone → Dataset A head (3 classes) + Dataset B head (4 classes)
- Transfer learning approach: ImageNet pretrained → fine-tune on engagement data
- Why ResNet-50: proven, efficient, works as both 2D (frames) and 3D (videos)

### Slide 10: Expected Contributions
1. **Rigorous OUC-CGE baseline** with 5-fold CV (more robust than original paper)
2. **First cross-dataset evaluation** between OUC-CGE and COLER (novel contribution — no one has done this)
3. **Unified engagement model** trained on both datasets simultaneously
4. **Analysis of engagement feature transferability** across countries, granularities, and classroom layouts

### Slide 11: Timeline
- Week 1: OUC-CGE 5-fold baseline (✅ in progress)
- Week 2: Data pipeline for cross-dataset experiments
- Week 3: Cross-dataset experiments
- Week 4: Results analysis and paper writing

### Slide 12: Questions & Discussion
- Open questions for professor:
  - Preferred label mapping: 3-class or binary?
  - Should we prioritize frame-level or video-level transfer?
  - Any additional models to benchmark (e.g., ViT, TimeSformer)?
  - Target venue for publication?

---

## 7. Master Prompt for Claude (PPT Generation)

Copy and paste the following prompt into Claude to generate your presentation:

---

```
You are a research presentation designer. Create a professional, visually detailed PowerPoint presentation (in markdown/outline format with speaker notes) for the following research project.

## Context
I am a graduate student presenting my research progress to my professor. The project involves training deep learning models for classroom engagement recognition using two datasets:

1. **OUC-CGE Dataset** (China): 7,439 MP4 video clips (10s each, 30fps), 3 classes — High (2,345 samples), Mid (2,053), Low (3,041). Group-level classroom engagement.
2. **COLER Dataset** (Vietnam): 18,546 samples with PNG frames + skeleton CSVs + emotion crops, 4 classes — Very Engaged, Engaged, Normal, Disengaged. Individual-level student engagement.

## Current Progress
- Completed EDA on both datasets
- Training SLOW 8×8 R50 (3D ResNet-50) model on OUC-CGE using 5-fold cross-validation (20 epochs per fold)
- Fixed a critical Unicode encoding bug that was causing 41% of training data to be silently skipped
- Fold 0 completed: validation accuracy improved from 39.95% (epoch 5) → 47.22% (epoch 10) → 48.12% (epoch 15) → 52.04% (epoch 20)
- Fold 1 in progress. Folds 2-4 queued. Estimated completion: ~18 hours
- Original OUC-CGE paper reported 97.8% accuracy with 50 epochs on a simpler train/test split. Our 5-fold CV is more rigorous.

## Research Plan
### Contribution 1: OUC-CGE Baseline with Rigorous 5-Fold CV
Reproduce and improve upon the original paper's results using proper cross-validation.

### Contribution 2: Cross-Dataset Transfer Study (Novel — no one has done this)
- **Challenge**: OUC-CGE has 3 classes, COLER has 4. OUC-CGE is videos, COLER is frames.
- **Label mapping**: Collapse COLER to 3 classes — merge "Very Engaged"+"Engaged" → High, "Normal" → Mid, "Disengaged" → Low
- **Experiment A (Frame-level)**: Extract frames from OUC-CGE, use COLER frames directly. Train ResNet-50 image classifier on Dataset A, test on Dataset B.
- **Experiment B (Video-level)**: Reconstruct COLER clips from frames. Train SlowFast on one dataset, test on the other.
- **Experiment C (Joint training)**: Shared ResNet-50 backbone with dataset-specific classification heads, trained on both datasets simultaneously.

### Contribution 3: Unified Multi-Dataset Engagement Model

## Timeline
- Week 1 (current): OUC-CGE 5-fold baseline (in progress)
- Week 2: Build cross-dataset data pipeline, implement label mapping
- Week 3: Run cross-dataset experiments
- Week 4: Analysis and paper writing

## Presentation Requirements
- Create **12 slides** following this structure:
  1. Title slide
  2. Research motivation & research question
  3. Dataset overview (side-by-side comparison)
  4. OUC-CGE EDA highlights
  5. COLER EDA highlights
  6. Current progress — OUC-CGE baseline results
  7. Label normalization strategy
  8. Cross-dataset experiment design
  9. Model architecture diagram
  10. Expected contributions
  11. Timeline (Gantt chart style)
  12. Questions & discussion

- For each slide, provide:
  - **Title**
  - **Key bullet points** (max 5 per slide)
  - **Visual suggestion** (chart type, diagram, table, or image to include)
  - **Speaker notes** (what to say when presenting this slide, 3-5 sentences)

- Use a **professional academic tone**
- Emphasize the **novelty** of cross-dataset evaluation (no one has done OUC-CGE ↔ COLER before)
- Highlight the **practical impact**: "Can a model trained in Chinese classrooms work in Vietnamese classrooms?"
- Include relevant **numbers and statistics** wherever possible
- Make the presentation suitable for a **10-15 minute talk**
```

---

## 8. Key Metrics Reference

| Metric | What It Measures | Ideal Value |
|--------|-----------------|-------------|
| **Top-1 Accuracy** | % of samples correctly classified | Higher is better (100% max) |
| **Macro-F1** | Harmonic mean of precision & recall, averaged across classes (accounts for class imbalance) | Higher is better (1.0 max) |
| **QWK (Quadratic Weighted Kappa)** | Agreement between predictions and ground truth, accounting for ordinal nature of labels (High > Mid > Low) | Higher is better (1.0 = perfect, 0 = random, negative = worse than random) |
| **Confusion Matrix** | Per-class breakdown of predictions vs ground truth | Diagonal-dominant = good |

---

## 9. Commands Cheat Sheet

```powershell
# Check training progress (latest stats)
Get-Content "E:\Video_Engagement\OUC-CGE\training_output.log" -Tail 5

# Check for errors
Get-Content "E:\Video_Engagement\OUC-CGE\training_error.log" -Tail 10

# See all validation results so far
Select-String -Path "E:\Video_Engagement\OUC-CGE\training_output.log" -Pattern "val_epoch"

# See fold transitions
Select-String -Path "E:\Video_Engagement\OUC-CGE\training_output.log" -Pattern "FOLD|fold_"

# Pause training
echo. > "E:\Video_Engagement\OUC-CGE\PAUSE"

# After all folds complete — run consolidation for full metrics
& "E:\Video_Engagement\OUC-CGE\ouc_env\Scripts\python.exe" "E:\Video_Engagement\OUC-CGE\consolidate_metrics.py"
```
