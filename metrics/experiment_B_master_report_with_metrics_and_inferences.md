# Master Scientific Report: Experiment B (3D Spatiotemporal Sequence Modeling & Temporal Transfer)

**Project Title**: Cross-Dataset Video Engagement Recognition  
**Author**: Koppesh  
**Model Architecture**: 3D Convolutional Network (`torchvision.models.video.r2plus1d_18`, Kinetics-400 Pretrained)  
**Input Resolution**: Contiguous 16-frame video clips $[C, T, H, W] = [3, 16, 224, 224]$  
**Taxonomy**: Harmonized 3-Class Taxonomy (`Low` = 0, `Mid` = 1, `High` = 2)  
**Hardware Profile**: NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM), PyTorch AMP (FP16), Gradient Accumulation Steps = 4  
**Primary Artifact Directory**: `outputs/`  
**Status**: **100% COMPLETE & VERIFIED** (All 4 Settings & Cross-Validation Folds Finished)

---

## 1. Executive Summary & Research Goal

While **Experiment A** evaluated static 2D spatial visual features (ResNet-50), **Experiment B** investigates whether incorporating explicit temporal dynamics through **3D spatiotemporal convolutions ($R(2+1)D$-18)** improves engagement recognition and mitigates cross-dataset domain shift.

We evaluate across the exact same 4-setting comparative matrix:
1. **Setting 1**: In-Domain OUC-CGE (Group-level classroom video recordings, 5-Fold CV)
2. **Setting 2**: In-Domain COLER (Individual student webcam video sequences, 5-Fold CV)
3. **Setting 3**: Direct Temporal Transfer (OUC-CGE 3D Model $\rightarrow$ COLER Test Set, Zero-Shot)
4. **Setting 4**: Reverse Direct Temporal Transfer (COLER 3D Model $\rightarrow$ OUC-CGE Test Set, Zero-Shot)

---

## 2. Master Comparative Results Matrix: Experiment B vs Experiment A

| Setting ID | Experimental Setting | Training Domain | Testing Domain | Exp B (3D R(2+1)D) Accuracy | Exp B Macro F1 | Exp B QWK Score | Exp A (2D ResNet-50) Accuracy | 2D $\rightarrow$ 3D Delta |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Setting 1** | In-Domain OUC-CGE | OUC-CGE | OUC-CGE (5-Fold CV) | **99.04% $\pm$ 0.29%** | **0.9901 $\pm$ 0.0030** | **0.9886 $\pm$ 0.0038** | 99.25% $\pm$ 0.26% | -0.21% |
| **Setting 2** | In-Domain COLER | COLER | COLER (5-Fold CV) | **91.04% $\pm$ 2.42%** | **0.9145 $\pm$ 0.0211** | **0.8784 $\pm$ 0.0151** | 96.38% $\pm$ 3.01% | -5.34%* |
| **Setting 3** | Direct Transfer (OUC $\rightarrow$ COLER) | OUC-CGE | COLER (Zero-Shot) | **24.60%** (25.12% all-fold) | **0.1342** (0.1435 all-fold) | **-0.0089** | 52.84% | -28.24% ⚠️ |
| **Setting 4** | Reverse Transfer (COLER $\rightarrow$ OUC) | COLER | OUC-CGE (Zero-Shot) | **38.60%** | **0.2311** | **-0.0255** | 29.15% | **+9.45%** 📈 |

*\*Note on Setting 2 Sample Resolution*: In Experiment A, 2D frames were sampled independently (~6,845 total frame samples). In Experiment B, models evaluate 16 consecutive frames as a single temporal sequence clip (625 total sequence samples), strictly preventing sequence-level frame leakage.

---

## 3. Setting-by-Setting Detailed Metrics Breakdown

### 3.1 Setting 1: In-Domain OUC-CGE 3D (5-Fold Cross-Validation)

Target dataset: **OUC-CGE** (7,364 total evaluated video sequences across 5 folds).

| Fold ID | Test Accuracy | Macro F1 | Quadratic Weighted Kappa (QWK) | Balanced Accuracy | Convergence Epoch |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 0** | 98.58% | 0.9853 | 0.9842 | 0.9854 | Epoch 16 |
| **Fold 1** | 98.91% | 0.9888 | 0.9895 | 0.9896 | Epoch 12 |
| **Fold 2** | **99.46%** | **0.9943** | **0.9948** | **0.9940** | Epoch 10 |
| **Fold 3** | 99.12% | 0.9912 | 0.9852 | 0.9914 | Epoch 13 |
| **Fold 4** | 99.12% | 0.9907 | 0.9895 | 0.9904 | Epoch 16 |
| **5-Fold Mean** | **99.04%** | **0.9901** | **0.9886** | **0.9902** | — |
| **Std Dev ($\pm$)** | **$\pm$ 0.29%** | **$\pm$ 0.0030** | **$\pm$ 0.0038** | **$\pm$ 0.0028** | — |

---

### 3.2 Setting 2: In-Domain COLER 3D (5-Fold Cross-Validation)

Target dataset: **COLER** (625 total evaluated video sequences across 5 folds).

| Fold ID | Test Accuracy | Macro F1 | Quadratic Weighted Kappa (QWK) | Balanced Accuracy | Convergence Epoch |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 0** | 87.30% | 0.8835 | 0.8626 | 0.8800 | Epoch 6 |
| **Fold 1** | 90.48% | 0.9085 | 0.8926 | 0.8952 | Epoch 6 |
| **Fold 2** | **94.40%** | **0.9453** | **0.9003** | **0.9385** | Epoch 4 |
| **Fold 3** | 92.80% | 0.9291 | 0.8700 | 0.9167 | Epoch 3 |
| **Fold 4** | 90.24% | 0.9060 | 0.8664 | 0.9162 | Epoch 4 |
| **5-Fold Mean** | **91.04%** | **0.9145** | **0.8784** | **0.9093** | — |
| **Std Dev ($\pm$)** | **$\pm$ 2.42%** | **$\pm$ 0.0211** | **$\pm$ 0.0151** | **$\pm$ 0.0201** | — |

#### **Setting 2 Aggregated Confusion Matrix (All 625 Holdout Sequences)**:
```
                    Predicted Low (0)   Predicted Mid (1)   Predicted High (2)   Total True
Actual Low (0)             142                  1                  11               154
Actual Mid (1)               0                173                  24               197
Actual High (2)              3                 17                 254               274
Total Predicted            145                191                 289               625
```
- **Low Engagement**: Precision: **97.93%**, Recall: **92.21%**, F1: **0.9498**
- **Mid Engagement**: Precision: **90.58%**, Recall: **87.82%**, F1: **0.8918**
- **High Engagement**: Precision: **87.89%**, Recall: **92.70%**, F1: **0.9023**

---

### 3.3 Setting 3: Direct Temporal Transfer (OUC 3D Model $\rightarrow$ COLER Test Set)

Model trained on OUC-CGE (`outputs/exp_b_train_ouc_3d/checkpoints/best_model.pth`) and evaluated on COLER test sequences **without fine-tuning**.

- **Accuracy**: **24.60%** (25.12% $\pm$ 0.72% across all 5 COLER folds)
- **Macro F1**: **0.1342** (0.1435 across all 5 COLER folds)
- **QWK Score**: **-0.0089** (discordant ranking)
- **Test Loss**: **3.1176**

#### **Setting 3 Confusion Matrix (Fold 0)**:
```
                    Predicted Low (0)   Predicted Mid (1)   Predicted High (2)   Total True
Actual Low (0)              31                  0                   0                31
Actual Mid (1)              37                  0                   3                40
Actual High (2)             55                  0                   0                55
Total Predicted            123                  0                   3               126
```
- **Low Class Accuracy**: 100.0% (31/31)
- **Mid Class Accuracy**: 0.00% (0/40)
- **High Class Accuracy**: 0.00% (0/55)

---

### 3.4 Setting 4: Reverse Direct Temporal Transfer (COLER 3D Model $\rightarrow$ OUC-CGE Test Set)

Model trained on COLER (`outputs/exp_b_train_coler_3d/checkpoints/best_model.pth`) and evaluated on OUC-CGE test sequences **without fine-tuning**.

- **Accuracy**: **38.60%** (vs 29.15% in Experiment A, a **+9.45% improvement**)
- **Macro F1**: **0.2311** (vs 0.1662 in Experiment A, a **+39.0% relative improvement**)
- **Macro Precision**: **0.2039**
- **Macro Recall**: **0.3203**
- **QWK Score**: **-0.0255**
- **Test Loss**: **1.4193**

#### **Setting 4 Confusion Matrix (Fold 0, N=1,474)**:
```
                    Predicted Low (0)   Predicted Mid (1)   Predicted High (2)   Total True
Actual Low (0)             526                  0                  81               607
Actual Mid (1)             309                  0                 102               411
Actual High (2)            413                  0                  43               456
Total Predicted           1248                  0                 226              1474
```
- **Low Class Accuracy**: 86.66% (526/607)
- **Mid Class Accuracy**: 0.00% (0/411)
- **High Class Accuracy**: 9.43% (43/456)

---

## 4. Key Scientific Inferences & Analysis

```
                              EXPERIMENT B: IN-DOMAIN VS CROSS-DOMAIN
  100% ┌─────────────────────────────────────────────────────────────────────────┐
       │  99.04%                    91.04%                                       │
   80% │ ┌───────┐                 ┌───────┐                                     │
       │ │ OUC   │                 │ COLER │                                     │
   60% │ │ 3D In │                 │ 3D In │                                     │
       │ │ Domain│                 │ Domain│                                     │
   40% │ │       │                 │       │                        38.60%       │
       │ │       │                 │       │                       ┌───────┐     │
   20% │ │       │                 │       │            24.60%     │ COLER │     │
       │ │       │                 │       │           ┌───────┐   │ -> OUC│     │
       │ │       │                 │       │           │ OUC-> │   │ 3D    │     │
    0% └─┴───────┴─────────────────┴───────┴───────────┴───────┴───┴───────┴─────┘
             Setting 1                 Setting 2           Setting 3       Setting 4
```

### Inference 1: Temporal Dynamics Improve Reverse Transfer (+9.45% over 2D)
- In Setting 4 (COLER $\rightarrow$ OUC), 3D spatiotemporal modeling improved zero-shot accuracy from **29.15% (2D ResNet-50)** to **38.60% (3D R(2+1)D-18)**, and Macro F1 from **0.1662 to 0.2311**.
- **Explanation**: A 2D model trained on tight student crops relies heavily on static facial textures. When transferred to a group classroom with small, distant faces, 2D visual cues completely break down. In contrast, 3D convolutions capture coarse temporal motion signatures (e.g., student movement vs stillness), which partially translate from individual webcams to group classroom settings.

### Inference 2: The Spatiotemporal "Motion Stillness" Semantic Inversion
- In Setting 3 (OUC $\rightarrow$ COLER), the 3D model suffered a massive drop to **24.60% accuracy**, predicting **123 out of 126 samples as Low engagement**.
- **The Core Conflict**:
  - In OUC-CGE (distant classroom view), students who are motionless and looking straight down at their desk are labeled **Low engagement** (disengaged or sleeping).
  - In COLER (close-up webcam), an engaged student sits motionless and stares intently at the screen, which annotators labeled **Mid** or **High engagement**.
  - When the 3D model encounters this sustained lack of large-scale body motion, it triggers its learned feature representation for "classroom disengagement" and predicts `Low` for 97.6% of COLER sequences.

### Inference 3: Why Zero-Shot Spatiotemporal Transfer Fails Without Adaptation
- Both Experiment A and Experiment B confirm that **neither static 2D features nor unaligned 3D temporal features can bridge the cross-dataset gap zero-shot**.
- Without domain adaptation or explicit feature alignment, temporal models overfit to dataset-specific motion characteristics (e.g., group movement rates vs individual micro-head movements) just as severely as 2D models overfit to camera lighting and background textures.

### Inference 4: Direct Justification for Domain Adaptation & Experiment C
- The failure of direct zero-shot transfer provides the definitive empirical motivation for **Experiment C (Joint Multi-Dataset Dual-Head & Domain Adversarial Learning)**:
  - Models need domain-invariant shared representation layers coupled with dataset-aware normalization or adversarial domain discriminators to align conflicting motion priors.

---

## 5. Execution Commands for Reproduction

```powershell
# 1. Reproduce Setting 1 (OUC-CGE 3D 5-Fold Cross Validation)
python train.py --config configs/experiment_B/train_ouc_3d.yaml --all-folds

# 2. Reproduce Setting 2 (COLER 3D 5-Fold Cross Validation)
python train.py --config configs/experiment_B/train_coler_3d.yaml --all-folds

# 3. Reproduce Setting 3 (Direct Transfer OUC 3D -> COLER Test Set)
python evaluate.py --config configs/experiment_B/eval_ouc_to_coler_3d.yaml

# 4. Reproduce Setting 4 (Reverse Direct Transfer COLER 3D -> OUC Test Set)
python evaluate.py --config configs/experiment_B/eval_coler_to_ouc_3d.yaml
```

---

## 6. Output Artifact & Directory Directory

- **Setting 1 Output**: [outputs/exp_b_train_ouc_3d](file:///e:/Video_Engagement/outputs/exp_b_train_ouc_3d)
- **Setting 2 Output**: [outputs/exp_b_train_coler_3d](file:///e:/Video_Engagement/outputs/exp_b_train_coler_3d)
- **Setting 3 Output**: [outputs/exp_b_eval_ouc_to_coler_3d](file:///e:/Video_Engagement/outputs/exp_b_eval_ouc_to_coler_3d)
- **Setting 4 Output**: [outputs/exp_b_eval_coler_to_ouc_3d](file:///e:/Video_Engagement/outputs/exp_b_eval_coler_to_ouc_3d)
- **Setting 1 Master Report**: [experiment_B_setting1_master_report.md](file:///e:/Video_Engagement/experiment_B_setting1_master_report.md)
- **Setting 2 Master Report**: [experiment_B_setting2_master_report.md](file:///e:/Video_Engagement/experiment_B_setting2_master_report.md)
- **Master Aggregator**: [aggregate_5fold_exp_b_setting2.py](file:///e:/Video_Engagement/aggregate_5fold_exp_b_setting2.py)
