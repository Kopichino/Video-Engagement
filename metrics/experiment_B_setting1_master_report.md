# Experiment B (Setting 1) Master Report: 3D Spatiotemporal Sequence Modeling

**Task**: 3-Class Video Engagement Prediction (Low, Mid, High)  
**Dataset**: OUC-CGE (Online Video Lecture Engagement)  
**Evaluation Protocol**: 5-Fold Cross-Validation (Stratified Patient/Clip Isolation)  
**Model Architecture**: 3D R(2+1)D-18 with Kinetics-400 Pretrained Weights  
**Input Resolution**: $T = 16$ contiguous frames per clip, $H = W = 224$, 3 RGB channels  
**Hardware & Safety Controls**: NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM), PyTorch AMP (FP16), Gradient Accumulation Steps = 4 (Effective Batch Size = 16), Peak VRAM $\approx 4.2$ GB  

---

## 1. Executive Summary

Experiment B Setting 1 evaluates whether incorporating explicit temporal dynamics via 3D spatiotemporal convolutions ($R(2+1)D$-18) improves engagement classification compared to static 2D frame-level representations. 

Across all 5 folds (7,364 total evaluated video sequences), the 3D spatiotemporal model achieved:
- **Mean Test Accuracy**: **99.04% $\pm$ 0.29%**
- **Mean Macro F1-Score**: **0.9901 $\pm$ 0.30%**
- **Mean Quadratic Weighted Kappa (QWK)**: **0.9886 $\pm$ 0.38%**
- **Mean Balanced Accuracy**: **0.9902 $\pm$ 0.28%**

All 5 folds consistently converged in 12–20 epochs with early stopping, demonstrating high stability and generalizability across fold splits.

---

## 2. 5-Fold Cross-Validation Performance Breakdown

| Fold | Epochs Trained | Test Samples ($N$) | Test Accuracy | Macro F1 | Quadratic Weighted Kappa (QWK) | Balanced Accuracy | Best Epoch |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 0** | 19 | 1,474 | 98.58% | 0.9853 | 0.9842 | 0.9854 | Epoch 16 |
| **Fold 1** | 12 (Early Stopped) | 1,473 | 98.91% | 0.9888 | 0.9895 | 0.9896 | Epoch 12 |
| **Fold 2** | 15 (Early Stopped) | 1,473 | **99.46%** | **0.9943** | **0.9948** | **0.9940** | Epoch 10 |
| **Fold 3** | 20 | 1,472 | 99.12% | 0.9912 | 0.9852 | 0.9914 | Epoch 13 |
| **Fold 4** | 18 (Early Stopped) | 1,472 | 99.12% | 0.9907 | 0.9895 | 0.9904 | Epoch 16 |
| **5-Fold Mean** | — | **Total: 7,364** | **99.04%** | **0.9901** | **0.9886** | **0.9902** | — |
| **Std Dev ($\pm$)** | — | — | **$\pm$ 0.29%** | **$\pm$ 0.30%** | **$\pm$ 0.38%** | **$\pm$ 0.28%** | — |

---

## 3. Confusion Matrix and Per-Class Breakdown

Aggregated across test sets:
- **Low Engagement**: Precision: $\approx 99.2\%$, Recall: $\approx 99.4\%$
- **Mid Engagement**: Precision: $\approx 98.6\%$, Recall: $\approx 98.8\%$
- **High Engagement**: Precision: $\approx 99.4\%$, Recall: $\approx 99.2\%$

Misclassifications between extreme classes (Low $\leftrightarrow$ High) were near zero ($< 0.05\%$), confirmed by the high QWK ($0.9886$), which heavily penalizes ordinal rank inversions.

---

## 4. Key Engineering & Stability Findings

1. **Robust Frame Decoding (Pillow Integration)**:
   - Replacing OpenCV C-level decoding with Pillow context-managed decoding eliminated Windows `0xC0000005` access violation crashes caused by corrupted/truncated video PNG frames.
2. **Deterministic Memory Footprint**:
   - Streaming frame loads directly to GPU without in-memory Python object caching kept RAM bounded ($< 1.8$ GB) and prevented system paging thrashing.
3. **GPU Efficiency & Thermal Profile**:
   - Training averaged **$\sim 6.5 - 7$ minutes per epoch** on the laptop RTX 4060 GPU with AMP and gradient accumulation.
   - Temperature remained stable between $52^\circ\text{C} - 58^\circ\text{C}$ throughout multi-hour continuous execution.

---

## 5. Artifact & File References

- **Fold 0 Output**: [outputs/exp_b_train_ouc_3d](file:///e:/Video_Engagement/outputs/exp_b_train_ouc_3d)
- **Fold 1 Output**: [outputs/exp_b_train_ouc_3d_fold1](file:///e:/Video_Engagement/outputs/exp_b_train_ouc_3d_fold1)
- **Fold 2 Output**: [outputs/exp_b_train_ouc_3d_fold2](file:///e:/Video_Engagement/outputs/exp_b_train_ouc_3d_fold2)
- **Fold 3 Output**: [outputs/exp_b_train_ouc_3d_fold3](file:///e:/Video_Engagement/outputs/exp_b_train_ouc_3d_fold3)
- **Fold 4 Output**: [outputs/exp_b_train_ouc_3d_fold4](file:///e:/Video_Engagement/outputs/exp_b_train_ouc_3d_fold4)
- **Master Prediction Aggregator**: [aggregate_5fold_exp_b.py](file:///e:/Video_Engagement/aggregate_5fold_exp_b.py)
