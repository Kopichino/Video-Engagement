# Cross-Dataset Video Engagement Recognition Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, research-grade PyTorch framework for **Cross-Dataset Video Engagement Recognition**. This repository harmonizes two disparate classroom datasets (**OUC-CGE** from China and **COLER** from Vietnam) into a single unified taxonomy to evaluate cross-domain generalization and transfer learning.

---

## Executive Summary

Classroom engagement recognition models often suffer from **domain shift** when applied to new classroom environments due to variations in camera angles, seating arrangements, cultural contexts, and annotation criteria. 

This project provides a **4-tier modular pipeline** that automates:
1. **Dataset Scanning & Ingestion**: Robust handling of multi-byte Unicode paths and corrupted video file filtering.
2. **Taxonomy Harmonization**: Dynamic JSON-driven label mapping into a 3-class taxonomy (`Low`, `Mid`, `High`) and a binary taxonomy (`Not Engaged` vs `Engaged`).
3. **Space-Optimized Frame Extraction**: Extracting keyframes downscaled to $256 \times 256$ resolution, reducing disk usage by 99% without sacrificing feature accuracy.
4. **Reproducible Experiment A Framework**: ResNet-50 frame-level transfer learning across 4 settings with 5-Fold Stratified Cross-Validation, Automatic Mixed Precision (AMP), gradient clipping, TensorBoard logging, and automated checkpointing.

---

## 5-Fold Stratified Cross-Validation Benchmark Results

### Setting 1: In-Domain OUC-CGE Frame Training (ResNet-50)

| Fold | Accuracy | Macro F1 | QWK Score | Validation Loss |
|---|---|---|---|---|
| **Fold 0** | **99.52%** | **0.9950** | **0.9942** | 0.0181 |
| **Fold 1** | **99.00%** | **0.9895** | **0.9886** | 0.0266 |
| **Fold 2** | **99.09%** | **0.9905** | **0.9902** | 0.0268 |
| **Fold 3** | **99.02%** | **0.9898** | **0.9888** | 0.0309 |
| **Fold 4** | **99.61%** | **0.9961** | **0.9934** | 0.0209 |
| **5-Fold Mean $\pm$ Std** | **99.25% $\pm$ 0.26%** 🚀 | **0.9922 $\pm$ 0.0028** | **0.9910 $\pm$ 0.0023** | **0.0247** |

---

## Directory Architecture & Organization

```
E:\Video_Engagement\
├── configs/
│   └── experiment_A/
│       ├── train_ouc.yaml                  # Setting 1: Train & Evaluate on OUC-CGE
│       ├── train_coler.yaml                # Setting 2: Train & Evaluate on COLER
│       ├── eval_ouc_to_coler.yaml          # Setting 3: OUC Model -> COLER Test (Direct Transfer)
│       └── eval_coler_to_ouc.yaml          # Setting 4: COLER Model -> OUC Test (Direct Transfer)
│
├── cross_dataset/
│   ├── config/
│   │   ├── base_config.py                  # Master PipelineConfig dataclass
│   │   └── label_mappings.json             # Fallback label taxonomy definition
│   ├── datasets/
│   │   └── frame_dataset.py                # UnifiedFrameDataset PyTorch Class
│   ├── extraction/
│   │   ├── samplers.py                     # FrameSampler (Single, Three, Uniform, Center, Random)
│   │   ├── ouc_extractor.py                # Space-optimized OUC-CGE MP4 frame extractor
│   │   ├── coler_extractor.py              # COLER PNG sequence validator & normalizer
│   │   └── extract_all.py                  # Master frame extraction runner
│   ├── label_mapping/
│   │   └── mapper.py                       # ConfigurableLabelMapper class
│   ├── preprocessing/
│   │   ├── ouc_cge_parser.py               # OUC-CGE dataset scanner & OpenCV validator
│   │   ├── coler_parser.py                 # COLER dataset hierarchy parser
│   │   └── build_manifest.py               # 5-Fold Stratified manifest builder
│   ├── representation/
│   │   └── builder.py                      # UnifiedDatasetRepresentationBuilder
│   ├── training/
│   │   ├── config_parser.py               # YAML experiment configuration parser
│   │   ├── model_factory.py               # ResNet-50 model builder & layer freezer
│   │   ├── optimizer_factory.py           # AdamW & SGD optimizer factory
│   │   ├── scheduler_factory.py           # Cosine, Step, & Plateau scheduler factory
│   │   ├── dataset_factory.py             # DataLoader builder with cross-eval support
│   │   ├── metrics.py                     # MetricsCalculator (Acc, F1, QWK, ROC AUC)
│   │   ├── checkpoint_manager.py          # Checkpointing & Auto-Resume manager
│   │   ├── logger.py                      # TensorBoard, CSV History, & Plotter
│   │   ├── evaluator.py                   # Single-pass @torch.no_grad() evaluator
│   │   └── trainer.py                     # AMP Trainer module
│   └── validation/
│       └── pipeline_validator.py          # End-to-end pipeline integrity auditor
│
├── data/
│   ├── configs/
│   │   └── label_mappings.json             # Master JSON label taxonomy config
│   └── manifests/
│       ├── unified_manifest.csv            # Clip-level manifest (7,989 samples)
│       ├── extracted_frames_manifest.csv   # Frame extraction log manifest
│       └── unified_frame_dataset.csv       # Unified frame dataset (28,815 samples)
│
├── outputs/                                # Generated experiment checkpoints & logs
├── train.py                                # CLI Training Entry Point
├── evaluate.py                             # CLI Evaluation Entry Point
├── run_experiment_A.ps1                    # Master PowerShell Automation Script
├── master_cross_dataset_project_report.md  # Comprehensive Advisor Project Report
└── README.md                               # THIS MASTER README
```

---

## Step-by-Step Execution & Training Guide

### 1. Environment Setup
Activate the Python virtual environment in PowerShell:

```powershell
cd E:\Video_Engagement
.\OUC-CGE\ouc_env\Scripts\Activate.ps1
```

---

### 2. Preprocessing & Data Ingestion Pipeline

#### **Step 2.1: Build Unified Clip Manifest**
Scans raw dataset directories, validates OpenCV video streams, excludes corrupted samples, and assigns 5-fold stratified cross-validation splits:
```powershell
python -m cross_dataset.preprocessing.build_manifest
```
*Output*: `data/manifests/unified_manifest.csv` (7,989 clip samples).

#### **Step 2.2: Extract Keyframe Samples**
Extracts 3 keyframes per video downscaled to $256 \times 256$ pixels (99% disk space optimization):
```powershell
python -m cross_dataset.extraction.extract_all
```
*Output*: `data/manifests/extracted_frames_manifest.csv`.

#### **Step 2.3: Build Unified Frame Dataset Representation**
Generates the master frame-level dataset manifest across OUC-CGE and COLER:
```powershell
python -m cross_dataset.representation.builder
```
*Output*: `data/manifests/unified_frame_dataset.csv` (28,815 frame samples).

#### **Step 2.4: Verify Pipeline Integrity**
Audits file existence, byte integrity, and label consistency:
```powershell
python -m cross_dataset.validation.pipeline_validator
```

---

### 3. Running Experiment A Training Settings

#### **Automated One-Click Pipeline Execution**
Run all 4 settings sequentially across all 5 folds:
```powershell
.\run_experiment_A.ps1 -AllFolds
```

---

#### **Manual CLI Execution per Setting**

##### **Setting 1: Train & Evaluate on OUC-CGE (5-Fold Cross Validation)**
```powershell
python train.py --config configs/experiment_A/train_ouc.yaml --all-folds
```

##### **Setting 2: Train & Evaluate on COLER (5-Fold Cross Validation)**
```powershell
python train.py --config configs/experiment_A/train_coler.yaml --all-folds
```

##### **Setting 3: Direct Transfer (OUC-CGE Model $\rightarrow$ COLER Test Set)**
Evaluates the trained OUC-CGE model directly on the COLER test dataset without fine-tuning:
```powershell
python evaluate.py --config configs/experiment_A/eval_ouc_to_coler.yaml
```

##### **Setting 4: Direct Transfer (COLER Model $\rightarrow$ OUC-CGE Test Set)**
Evaluates the trained COLER model directly on the OUC-CGE test dataset without fine-tuning:
```powershell
python evaluate.py --config configs/experiment_A/eval_coler_to_ouc.yaml
```

---

## Experiment Output Structure (`outputs/`)

Each experiment run generates a dedicated output folder:

```
outputs/exp_a_train_ouc_fold0/
├── checkpoints/
│   ├── best_model.pth            # Model weights at peak validation Macro F1
│   └── last_model.pth            # Final epoch checkpoint (used for auto-resume)
├── metrics/
│   ├── history.csv               # Epoch-by-epoch training/val loss, accuracy, & F1
│   └── epoch_020_metrics.json    # JSON metric snapshot
├── predictions/
│   └── test_predictions.csv      # Per-sample predictions & probabilities
└── logs/                         # TensorBoard event logs
```

### Viewing TensorBoard Real-Time Training Logs
```powershell
tensorboard --logdir outputs/
```
Then open `http://localhost:6006` in your web browser.

---


## Koppesh P

## Citation & Reference Reports

For a complete technical analysis and presentation guide prepared for academic advisors, see:
- [master_cross_dataset_project_report.md](file:///e:/Video_Engagement/master_cross_dataset_project_report.md)
- [experiment_A_final_engineering_review.md](file:///e:/Video_Engagement/experiment_A_final_engineering_review.md)
