# Experiment A Training Framework — Technical Documentation

**Version**: 1.0.0  
**Scope**: Complete research-grade PyTorch training, evaluation, and cross-dataset transfer learning framework for **Experiment A (Frame-Level Transfer)**.

---

## 1. Overview & Architectural Design

The Experiment A Framework is a production-grade, highly modular deep-learning system built on PyTorch and `torchvision` ResNet-50. It enables four distinct evaluation settings through configuration files without requiring any code modifications or hardcoded paths.

### The Four Experiment A Settings

| Setting ID | Description | Config File | Data Flow |
|---|---|---|---|
| **Setting 1** | Train & Evaluate on OUC-CGE | `configs/experiment_A/train_ouc.yaml` | Train (OUC) $\rightarrow$ Val (OUC) $\rightarrow$ Test (OUC) |
| **Setting 2** | Train & Evaluate on COLER | `configs/experiment_A/train_coler.yaml` | Train (COLER) $\rightarrow$ Val (COLER) $\rightarrow$ Test (COLER) |
| **Setting 3** | Direct Transfer: OUC $\rightarrow$ COLER | `configs/experiment_A/eval_ouc_to_coler.yaml` | OUC Checkpoint $\rightarrow$ COLER Test Set |
| **Setting 4** | Direct Transfer: COLER $\rightarrow$ OUC | `configs/experiment_A/eval_coler_to_ouc.yaml` | COLER Checkpoint $\rightarrow$ OUC-CGE Test Set |

---

## 2. Directory Structure

```
.
├── configs/
│   └── experiment_A/
│       ├── train_ouc.yaml            # Setting 1 Config
│       ├── train_coler.yaml          # Setting 2 Config
│       ├── eval_ouc_to_coler.yaml    # Setting 3 Direct Transfer Config
│       └── eval_coler_to_ouc.yaml    # Setting 4 Direct Transfer Config
│
├── cross_dataset/
│   └── training/                     # ★ FRAMEWORK PACKAGE
│       ├── __init__.py
│       ├── config_parser.py          # Config loader & YAML snapshot manager
│       ├── model_factory.py          # ResNet-50 builder, freezing, & head config
│       ├── optimizer_factory.py      # AdamW / SGD builder
│       ├── scheduler_factory.py      # Cosine, Step, & ReduceLROnPlateau builder
│       ├── dataset_factory.py        # PyTorch DataLoaders builder
│       ├── metrics.py                # Acc, F1, Prec, Rec, QWK, CM, ROC AUC
│       ├── checkpoint_manager.py     # Checkpointing & Auto-Resume manager
│       ├── logger.py                 # TensorBoard, CSV History, & Plotting manager
│       ├── evaluator.py              # Single-pass evaluation loop
│       └── trainer.py                # AMP Trainer (Grad Clip, Accumulation, EarlyStop)
│
├── outputs/                           # ★ GENERATED EXPERIMENT ARTIFACTS
│   └── <experiment_name>/
│       ├── checkpoints/              # best_model.pth, last_model.pth
│       ├── metrics/                  # history.csv, epoch_XXX_metrics.json
│       ├── plots/                    # loss_acc_curves.png, confusion_matrix.png
│       ├── predictions/              # test_predictions.csv
│       ├── logs/                     # TensorBoard event logs
│       └── config_snapshot.yaml      # Saved config snapshot
│
├── train.py                          # CLI Training Entry Point
└── evaluate.py                       # CLI Evaluation Entry Point
```

---

## 3. Module Responsibilities & Features

### 3.1 Configuration System (`cross_dataset/training/config_parser.py`)
- **Responsibility**: Loads YAML configuration files, applies fallback default values, and exports readable configuration snapshots (`config_snapshot.yaml`).
- **Zero Hardcoding**: All hyperparameters (LR, weight decay, epochs, batch size, patience, dataset roots, fold indices) are strictly specified via YAML.

### 3.2 Model Factory (`cross_dataset/training/model_factory.py`)
- **Responsibility**: Builds `torchvision` ResNet-50 frame-level engagement classifiers.
- **Key Features**:
  - `pretrained`: Toggle ImageNet pretrained weights (`ResNet50_Weights.DEFAULT`) vs. training from scratch (`pretrained: false`).
  - Configurable Head: Linear $\rightarrow$ BatchNorm1d $\rightarrow$ ReLU $\rightarrow$ Dropout $\rightarrow$ Linear (`num_classes`).
  - Layer Freezing: `freeze_backbone: true` freezes feature extractor weights.
  - Selective Unfreezing: `unfreeze_blocks: ['layer4', 'fc']` unfreezes specified ResNet blocks for fine-tuning.

### 3.3 Optimizer & Scheduler Factories (`optimizer_factory.py`, `scheduler_factory.py`)
- **Optimizers**: AdamW, SGD (with momentum & nesterov).
- **Schedulers**:
  - `cosine`: CosineAnnealingLR with minimum learning rate `eta_min`.
  - `step`: StepLR with configurable `step_size` and `gamma`.
  - `plateau`: ReduceLROnPlateau monitoring validation loss or Macro F1.

### 3.4 Dataset Factory (`cross_dataset/training/dataset_factory.py`)
- **Responsibility**: Wraps `UnifiedFrameDataset` into standard PyTorch `DataLoader` objects.
- **Features**: Configures `batch_size`, `num_workers`, `pin_memory`, dataset filtering (OUC-CGE vs. COLER), class balancing, and image augmentation transforms.

### 3.5 Metrics Calculator (`cross_dataset/training/metrics.py`)
- **Responsibility**: Computes standard classification evaluation metrics after every epoch:
  - **Accuracy** (overall micro accuracy)
  - **Macro F1-Score** (harmonic mean of precision and recall)
  - **Macro Precision & Recall**
  - **Per-Class Metrics** (F1, Precision, Recall per class)
  - **Class Accuracy** (diagonal accuracy per class)
  - **Balanced Accuracy** (unweighted mean recall)
  - **Quadratic Weighted Kappa (QWK)** (ordinal agreement)
  - **ROC Curve & AUC** (computed automatically when operating in binary mode)

### 3.6 Checkpoint Manager (`cross_dataset/training/checkpoint_manager.py`)
- **Responsibility**: Serialization and automatic training resumption.
- **Features**:
  - Saves `best_model.pth` whenever the monitored validation metric (`val_macro_f1`) improves.
  - Saves `last_model.pth` at the end of every epoch.
  - Serializes complete state: `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `scaler_state_dict` (AMP), epoch index, and metrics.
  - `auto_resume: true`: Automatically detects `last_model.pth` and resumes seamlessly.

### 3.7 Experiment Logger & Plotter (`cross_dataset/training/logger.py`)
- **Responsibility**: Multi-channel logging and visualization.
- **Outputs**:
  - **TensorBoard**: Event logs written to `logs/` for loss, accuracy, Macro F1, QWK, and learning rate tracking.
  - **CSV History**: `metrics/history.csv` updated epoch-by-epoch.
  - **Plots**: Matplotlib/Seaborn figures generated under `plots/`:
    - `loss_acc_curves.png`: Training loss & accuracy vs. validation curves.
    - `learning_rate_schedule.png`: Learning rate decay curve.
    - `confusion_matrix.png`: Heatmap of final confusion matrix.
    - `roc_curve.png`: ROC curve with AUC score (binary mode).

### 3.8 Evaluator & Trainer (`evaluator.py`, `trainer.py`)
- **Evaluator**: Performs single-pass `@torch.no_grad()` evaluation over DataLoader splits with AMP autocast and generates detailed prediction CSV files (`predictions/test_predictions.csv`).
- **Trainer**: Orchestrates the multi-epoch training loop with:
  - **AMP (Automatic Mixed Precision)** via `torch.cuda.amp.autocast` and `GradScaler`.
  - **Gradient Accumulation** (`gradient_accumulation_steps`).
  - **Gradient Clipping** (`nn.utils.clip_grad_norm_`).
  - **Early Stopping** with configurable patience and minimum delta.

---

## 4. Execution Commands & Pipeline Flow

### Training Execution (Setting 1 & Setting 2)

```powershell
# Setting 1: Train on OUC-CGE
python train.py --config configs/experiment_A/train_ouc.yaml

# Setting 2: Train on COLER
python train.py --config configs/experiment_A/train_coler.yaml
```

### Direct Transfer Evaluation (Setting 3 & Setting 4)

```powershell
# Setting 3: Direct Transfer - Evaluate OUC model on COLER
python evaluate.py --config configs/experiment_A/eval_ouc_to_coler.yaml

# Setting 4: Direct Transfer - Evaluate COLER model on OUC-CGE
python evaluate.py --config configs/experiment_A/eval_coler_to_ouc.yaml
```

---

## 5. Summary of Output Artifacts

Every experiment run produces a clean, self-contained output directory:

```
outputs/exp_a_train_ouc/
├── checkpoints/
│   ├── best_model.pth
│   └── last_model.pth
├── metrics/
│   ├── history.csv
│   └── epoch_020_metrics.json
├── plots/
│   ├── loss_acc_curves.png
│   ├── learning_rate_schedule.png
│   └── test_confusion_matrix.png
├── predictions/
│   └── test_predictions.csv
├── logs/
│   └── events.out.tfevents...
└── config_snapshot.yaml
```

---

*Documentation complete for Experiment A Training Framework.*
