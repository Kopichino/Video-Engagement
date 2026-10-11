# Detailed Technical Guide & Concepts: Experiment A & 5-Fold Cross Validation

**Author**: Koppesh  
**Date**: August 4, 2026  
**Document Goal**: Clarify 5-Fold Cross Validation execution, explain the fundamental difference between Phase 1 (3D SlowFast) and Experiment A (2D ResNet-50 Frame-Level), and detail how frame-level transfer works across Settings 1 to 4.

---

## 1. Clarification 1: 5-Fold Cross Validation vs. Single-Fold Execution

### What does `fold: 0` mean?
In `configs/experiment_A/train_ouc.yaml`, `fold: 0` specifies that **Fold 0 is held out as the validation set**, while Folds 1, 2, 3, and 4 are combined for training. Running `fold: 0` provides a fast 1-fold evaluation (1/5th of the cross-validation dataset).

### How does full 5-Fold Stratified Cross Validation work?
To report publication-grade metrics (Mean Accuracy $\pm$ Std, Mean Macro F1 $\pm$ Std), the model must be trained and evaluated **5 separate times**, where each fold takes a turn as the validation/test set:

| Run Number | Training Folds | Validation / Test Fold | Command Line |
|---|---|---|---|
| **Fold 0** | Folds 1, 2, 3, 4 | **Fold 0** | `python train.py --config configs/experiment_A/train_ouc.yaml --fold 0` |
| **Fold 1** | Folds 0, 2, 3, 4 | **Fold 1** | `python train.py --config configs/experiment_A/train_ouc.yaml --fold 1` |
| **Fold 2** | Folds 0, 1, 3, 4 | **Fold 2** | `python train.py --config configs/experiment_A/train_ouc.yaml --fold 2` |
| **Fold 3** | Folds 0, 1, 2, 4 | **Fold 3** | `python train.py --config configs/experiment_A/train_ouc.yaml --fold 3` |
| **Fold 4** | Folds 0, 1, 2, 3 | **Fold 4** | `python train.py --config configs/experiment_A/train_ouc.yaml --fold 4` |

### How to run all 5 folds automatically in one command?
We updated `train.py` to support the `--all-folds` flag! Running the command below will execute Folds 0 through 4 sequentially and automatically compute the overall 5-Fold Mean $\pm$ Standard Deviation metrics:

```powershell
# Run Full 5-Fold Stratified Cross-Validation for Setting 1
python train.py --config configs/experiment_A/train_ouc.yaml --all-folds

# Run Full 5-Fold Stratified Cross-Validation for Setting 2
python train.py --config configs/experiment_A/train_coler.yaml --all-folds
```

---

## 2. Clarification 2: Phase 1 Baseline (3D SlowFast) vs. Experiment A (2D ResNet-50)

You asked: *"Is setting 1 similar to the one I already did? How is this frame-level setting different?"*

### Summary Comparison Table

| Attribute | Phase 1 Baseline (What you did earlier) | Experiment A (What we are doing now) |
|---|---|---|
| **Model Architecture** | 3D ResNet-50 (SlowFast 8×8) | 2D ResNet-50 (`torchvision` ImageNet pretrained) |
| **Input Data Type** | 3D Spatio-Temporal Video Tensors (`.mp4`) | 2D Static Image Keyframe Tensors (`.png`) |
| **Input Shape** | `[Batch, Channel, Frames, Height, Width]` $\rightarrow$ `[32, 3, 16, 224, 224]` | `[Batch, Channel, Height, Width]` $\rightarrow$ `[32, 3, 224, 224]` |
| **Learning Dimension** | **Spatial + Temporal Motion** (Captures movement over time) | **Spatial Visual Cues** (Captures facial expressions, posture, gaze per frame) |
| **Granularity** | Video-Level / Clip-Level | Frame-Level / Image-Level |
| **Pretraining** | Trained from scratch on OUC-CGE | ImageNet Pretrained (Transfer Learning) |

### Plain-Language Explanation for your Professor
- **Phase 1 Baseline (3D Video Classification)**:
  The 3D SlowFast model looks at **short video clips** (16 consecutive frames playing over time). It learns *motion features*—such as a student leaning forward, writing, nodding, or raising a hand.
- **Experiment A (2D Frame-Level Spatial Transfer)**:
  Experiment A converts videos into **individual static image frames**. A 2D ResNet-50 model processes each frame like a single photograph, learning *spatial features*—such as whether a student's face looks attentive, distracted, or neutral at a specific instant.

### Why do 2D Frame-Level Transfer before 3D Video Transfer?
1. **Common Domain Ground**: The COLER dataset consists natively of cropped student `.png` images, whereas OUC-CGE consists of `.mp4` videos. Frame extraction allows both datasets to be represented in the exact same 2D image format.
2. **Spatial Feature Foundation**: By training a 2D ResNet-50 on static frames first (Experiment A), we evaluate how well pure spatial features (facially/posturally expressed engagement) transfer across different classroom environments before introducing complex 3D temporal modeling (Experiment B).

---

## 3. Detailed Breakdown of All 4 Settings in Experiment A

```
                                  EXPERIMENT A FRAMEWORK
                                            │
         ┌──────────────────────────────────┴──────────────────────────┐
         ▼                                                             ▼
IN-DOMAIN BASELINES                                          DIRECT TRANSFER EVALUATIONS
┌─────────────────────────────────────────┐                  ┌─────────────────────────────────────────┐
│ Setting 1: Train & Evaluate on OUC-CGE  │                  │ Setting 3: OUC Model -> COLER Test Set  │
│ (2D Frame ResNet-50, 3-class taxonomy)  │                  │ (Evaluates cross-domain generalization) │
├─────────────────────────────────────────┤                  ├─────────────────────────────────────────┤
│ Setting 2: Train & Evaluate on COLER    │                  │ Setting 4: COLER Model -> OUC Test Set  │
│ (2D Frame ResNet-50, 3-class taxonomy)  │                  │ (Evaluates cross-domain generalization) │
└─────────────────────────────────────────┘                  └─────────────────────────────────────────┘
```

### Setting 1: In-Domain OUC-CGE Frame Training
- **Input**: Extracted keyframes from OUC-CGE classroom videos (`data/extracted_frames/OUC-CGE/`).
- **Goal**: Establish the in-domain 2D frame classification baseline on OUC-CGE.
- **Output Checkpoint**: `outputs/exp_a_train_ouc/checkpoints/best_model.pth`.

### Setting 2: In-Domain COLER Frame Training
- **Input**: Student crop PNG frames from COLER classroom recordings (`data/extracted_frames/COLER/`).
- **Goal**: Establish the in-domain 2D frame classification baseline on COLER.
- **Output Checkpoint**: `outputs/exp_a_train_coler/checkpoints/best_model.pth`.

### Setting 3: Direct Transfer (OUC-CGE Model $\rightarrow$ COLER Test Set)
- **Input**: COLER test frames.
- **Goal**: Load the OUC-CGE trained model (`outputs/exp_a_train_ouc/checkpoints/best_model.pth`) and test it directly on COLER **without any fine-tuning**.
- **Research Value**: Measures how well spatial features learned in Chinese classrooms transfer to Vietnamese classrooms (quantifies domain shift drop).

### Setting 4: Direct Transfer (COLER Model $\rightarrow$ OUC-CGE Test Set)
- **Input**: OUC-CGE test frames.
- **Goal**: Load the COLER trained model (`outputs/exp_a_train_coler/checkpoints/best_model.pth`) and test it directly on OUC-CGE **without any fine-tuning**.
- **Research Value**: Measures the reverse cross-dataset transfer generalization.

---

## 4. Updated Execution Script Commands

### To run 1-Fold (Fold 0) Quick Experiments (20 Epochs):
```powershell
# Setting 1 (OUC-CGE)
python train.py --config configs/experiment_A/train_ouc.yaml --fold 0

# Setting 2 (COLER)
python train.py --config configs/experiment_A/train_coler.yaml --fold 0

# Setting 3 (OUC -> COLER Direct Transfer)
python evaluate.py --config configs/experiment_A/eval_ouc_to_coler.yaml

# Setting 4 (COLER -> OUC Direct Transfer)
python evaluate.py --config configs/experiment_A/eval_coler_to_ouc.yaml
```

### To run Full 5-Fold Stratified Cross-Validation (20 Epochs x 5 Folds):
```powershell
# Full 5-Fold Cross-Validation for Setting 1
python train.py --config configs/experiment_A/train_ouc.yaml --all-folds

# Full 5-Fold Cross-Validation for Setting 2
python train.py --config configs/experiment_A/train_coler.yaml --all-folds
```

---

*Technical Concept Guide & 5-Fold Execution Documentation Complete.*
