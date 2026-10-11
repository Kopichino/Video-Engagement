# Step-by-Step Execution Guide: Experiment A (Settings 1 to 4)

**Target Framework**: ResNet-50 Frame-Level Transfer Learning  
**Target Epochs**: 20 Epochs per Training Setting  
**Virtual Environment Path**: `E:\Video_Engagement\OUC-CGE\ouc_env`  
**Master Command File**: `run_experiment_A.ps1`

---

## 1. Overview of Experiment A Settings

Experiment A evaluates frame-level engagement classification across four distinct settings:

```
                  +-------------------------------------------------------+
                  |                 EXPERIMENT A SETTINGS                 |
                  +-------------------------------------------------------+
                                              |
       +-------------------+------------------+-------------------+-------------------+
       |                   |                                      |                   |
       v                   v                                      v                   v
  [Setting 1]         [Setting 2]                            [Setting 3]         [Setting 4]
Train: OUC-CGE      Train: COLER                           Train: OUC-CGE      Train: COLER
 Val: OUC-CGE        Val: COLER                             Eval: COLER         Eval: OUC-CGE
Test: OUC-CGE       Test: COLER                            (Zero-Shot)         (Zero-Shot)
```

1. **Setting 1 (`exp_a_train_ouc`)**: In-domain baseline trained & tested on OUC-CGE.
2. **Setting 2 (`exp_a_train_coler`)**: In-domain baseline trained & tested on COLER.
3. **Setting 3 (`exp_a_eval_ouc_to_coler`)**: Direct cross-dataset transfer (OUC-CGE model evaluated on COLER test set without fine-tuning).
4. **Setting 4 (`exp_a_eval_coler_to_ouc`)**: Direct cross-dataset transfer (COLER model evaluated on OUC-CGE test set without fine-tuning).

---

## 2. Prerequisites & Virtual Environment Setup

Open PowerShell or Command Prompt, navigate to the repository root directory `E:\Video_Engagement`, and activate the Python virtual environment:

```powershell
# Navigate to workspace root
cd E:\Video_Engagement

# Activate virtual environment (PowerShell)
.\OUC-CGE\ouc_env\Scripts\Activate.ps1
```

If using Command Prompt (`cmd.exe`):
```cmd
cd /d E:\Video_Engagement
OUC-CGE\ouc_env\Scripts\activate.bat
```

---

## 3. Manual Step-by-Step Execution Guide

### Step 1: Run Setting 1 (Train & Evaluate on OUC-CGE)
Trains ResNet-50 for 20 epochs on OUC-CGE dataset (Fold 0 split) with AMP mixed precision and AdamW optimizer. Saves best model weights to `outputs/exp_a_train_ouc/checkpoints/best_model.pth`.

```powershell
python train.py --config configs/experiment_A/train_ouc.yaml
```

*Expected Terminal Output*:
- TensorBoard SummaryWriter initialized at `outputs/exp_a_train_ouc/logs`
- Epoch progress logs showing `Train Loss`, `Val Loss`, `Val Macro F1`, and `QWK`
- Auto-saving `best_model.pth` and `last_model.pth`
- Final test evaluation metrics output on completion

---

### Step 2: Run Setting 2 (Train & Evaluate on COLER)
Trains ResNet-50 for 20 epochs on COLER dataset (Fold 0 split). Saves best model weights to `outputs/exp_a_train_coler/checkpoints/best_model.pth`.

```powershell
python train.py --config configs/experiment_A/train_coler.yaml
```

*Expected Terminal Output*:
- Epoch progress logs showing training across COLER student frame crops
- Best model saved at `outputs/exp_a_train_coler/checkpoints/best_model.pth`
- Final COLER in-domain test evaluation metrics output

---

### Step 3: Run Setting 3 (Direct Transfer: OUC Model $\rightarrow$ COLER Test Set)
Evaluates the trained OUC-CGE model (`outputs/exp_a_train_ouc/checkpoints/best_model.pth`) directly on the COLER test dataset **without any fine-tuning**.

```powershell
python evaluate.py --config configs/experiment_A/eval_ouc_to_coler.yaml
```

*Expected Terminal Output*:
- Loads best checkpoint from `outputs/exp_a_train_ouc/checkpoints/best_model.pth`
- Evaluates directly on COLER test samples
- Saves prediction CSV to `outputs/exp_a_eval_ouc_to_coler/predictions/transfer_test_predictions.csv`
- Saves confusion matrix plot to `outputs/exp_a_eval_ouc_to_coler/plots/transfer_confusion_matrix.png`

---

### Step 4: Run Setting 4 (Direct Transfer: COLER Model $\rightarrow$ OUC-CGE Test Set)
Evaluates the trained COLER model (`outputs/exp_a_train_coler/checkpoints/best_model.pth`) directly on the OUC-CGE test dataset **without any fine-tuning**.

```powershell
python evaluate.py --config configs/experiment_A/eval_coler_to_ouc.yaml
```

*Expected Terminal Output*:
- Loads best checkpoint from `outputs/exp_a_train_coler/checkpoints/best_model.pth`
- Evaluates directly on OUC-CGE test samples
- Saves prediction CSV to `outputs/exp_a_eval_coler_to_ouc/predictions/transfer_test_predictions.csv`
- Saves confusion matrix plot to `outputs/exp_a_eval_coler_to_ouc/plots/transfer_confusion_matrix.png`

---

## 4. Automated One-Click Execution Method

Instead of running each command manually, you can execute all four settings sequentially using the provided master PowerShell script `run_experiment_A.ps1`:

```powershell
# Run all 4 Settings sequentially in automated pipeline
.\run_experiment_A.ps1
```

This script handles python environment paths, error code checking after each setting, and logs step progress.

---

## 5. Output Directory Structure & Results Inspection

After running all 4 settings, the `outputs/` directory will be populated with all experimental artifacts:

```
outputs/
├── exp_a_train_ouc/                  # Setting 1 Artifacts
│   ├── checkpoints/
│   │   ├── best_model.pth            # Best model weights
│   │   └── last_model.pth            # Last epoch checkpoint
│   ├── metrics/
│   │   ├── history.csv               # Epoch-by-epoch training/val loss & F1 scores
│   │   └── epoch_020_metrics.json
│   ├── plots/
│   │   ├── loss_acc_curves.png       # Loss & accuracy visualization curves
│   │   ├── learning_rate_schedule.png# LR decay curve
│   │   └── test_confusion_matrix.png # Final test confusion matrix heatmap
│   ├── predictions/
│   │   └── test_predictions.csv      # Per-sample predictions & probabilities
│   └── logs/                         # TensorBoard event log files
│
├── exp_a_train_coler/                # Setting 2 Artifacts
├── exp_a_eval_ouc_to_coler/          # Setting 3 Direct Transfer Artifacts
└── exp_a_eval_coler_to_ouc/          # Setting 4 Direct Transfer Artifacts
```

### Viewing TensorBoard Real-Time Training Logs
To visualize real-time training curves in your web browser, run TensorBoard pointing to `outputs/`:

```powershell
tensorboard --logdir outputs/
```
Then open `http://localhost:6006` in your web browser.

---

## 6. Key Configuration Summary (`configs/experiment_A/`)

If you want to modify hyperparameters (e.g. change batch size or learning rate), edit the YAML config files directly:

| Config Field | Setting 1 & 2 Value | Description |
|---|---|---|
| `training.epochs` | `20` | Total training epochs |
| `dataset.batch_size` | `32` | Batch size per GPU step |
| `model.architecture` | `"resnet50"` | Backbone model architecture |
| `model.pretrained` | `true` | Use ImageNet pretrained weights |
| `optimizer.type` | `"adamw"` | AdamW optimizer |
| `optimizer.lr` | `0.0001` | Initial learning rate |
| `scheduler.type` | `"cosine"` | CosineAnnealingLR decay |
| `training.use_amp` | `true` | FP16 Automatic Mixed Precision |

---

*Execution Guide complete — Ready to run Experiment A across Settings 1, 2, 3, and 4.*
