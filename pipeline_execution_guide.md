# 🚀 Step-by-Step Execution Guide for 5-Fold Cross-Validation

This guide provides the exact PowerShell commands and procedures to generate splits, configure epochs, run the 5-fold training loop, and consolidate testing metrics.

---

## 📋 Prerequisites & Virtual Environment

Before running any script, make sure your working directory is set to `E:\Video_Engagement\OUC-CGE`. 

In PowerShell, the path to the Python executable in your virtual environment is:
```powershell
"E:\Video_Engagement\OUC-CGE\ouc_env\Scripts\python.exe"
```
All commands below use this path.

---

## 🚶 Step-by-Step Pipeline Execution

### Step 1: Generate Balanced Folds
Run the split generator. This will scan all dataset videos, ignore corrupted ones, and randomly distribute balanced classes across 5 folds.
```powershell
& "E:\Video_Engagement\OUC-CGE\ouc_env\Scripts\python.exe" "E:\Video_Engagement\OUC-CGE\generate_fold_splits.py"
```

---

### Step 2: Training the Model (Folds 0-4)
You can configure and run the training using the master runner script (`run_5fold.py`).

#### How to Modify the Number of Epochs
We implemented command-line arguments to allow customizing epochs and evaluation periods easily without modifying config files.

*   **Option A (Recommended): Command-line Parameter**
    Simply append the `--max-epochs` parameter to the runner command:
    ```powershell
    # Train all 5 folds for 10 epochs each, saving checkpoint and evaluating every epoch
    & "E:\Video_Engagement\OUC-CGE\ouc_env\Scripts\python.exe" "E:\Video_Engagement\OUC-CGE\run_5fold.py" --max-epochs 10 --eval-period 1 --checkpoint-period 1
    ```

*   **Option B: Modifying the Configuration File**
    Open [5fold_SLOW_8x8_R50.yaml](file:///E:/Video_Engagement/OUC-CGE/configs/Kinetics/5fold_SLOW_8x8_R50.yaml#L50) and modify line 50:
    ```yaml
    SOLVER:
      MAX_EPOCH: 10   # Change this to your desired number of epochs
    ```
    Then run the script without overrides:
    ```powershell
    & "E:\Video_Engagement\OUC-CGE\ouc_env\Scripts\python.exe" "E:\Video_Engagement\OUC-CGE\run_5fold.py"
    ```

#### Running Specific Folds (e.g., Target Folds)
If you want to train only a subset of folds (for example, if training got interrupted or you want to train folds one-by-one):
```powershell
# Train Fold 0 only for 10 epochs:
& "E:\Video_Engagement\OUC-CGE\ouc_env\Scripts\python.exe" "E:\Video_Engagement\OUC-CGE\run_5fold.py" --folds 0 --max-epochs 10

# Train Fold 1, 2, and 3 for 10 epochs:
& "E:\Video_Engagement\OUC-CGE\ouc_env\Scripts\python.exe" "E:\Video_Engagement\OUC-CGE\run_5fold.py" --folds 1 2 3 --max-epochs 10
```

---

### Step 3: Run Testing & Consolidate Metrics
After training is finished, run the consolidation script to load the best checkpoint of each fold, perform validation set inference, calculate metrics (Accuracy, Macro-F1, QWK), and write the report.
```powershell
& "E:\Video_Engagement\OUC-CGE\ouc_env\Scripts\python.exe" "E:\Video_Engagement\OUC-CGE\consolidate_metrics.py"
```
The final cross-validation results table will be printed in the terminal and saved to:
👉 [work_dirs/5fold_summary.txt](file:///E:/Video_Engagement/OUC-CGE/work_dirs/5fold_summary.txt)

---

## ⏱️ Training Duration Estimation (5 Folds × 10 Epochs)

Based on the actual execution logs from the dry-run of Fold 0, we can calculate a highly precise training time estimation:

### 1. Benchmark Metrics from Dry-Run:
*   **Time taken for 1 epoch on 1 fold**: **30.9 minutes**
*   **Dataset size per fold**: 4,926 train samples / 1,233 validation samples.

### 2. Time Calculations:
*   **1 Fold (10 Epochs)**:
    $$\text{Time} = 10 \text{ epochs} \times 30.9 \text{ minutes} \approx 309 \text{ minutes} \text{ (5.15 hours)}$$
*   **All 5 Folds (10 Epochs each)**:
    $$\text{Total Time} = 5 \text{ folds} \times 5.15 \text{ hours} \approx 25.75 \text{ hours}$$

### 3. Summary of Expected Training Times:
*   **1 Epoch (Across all 5 folds)**: $\approx 2.5 \text{ hours}$
*   **10 Epochs (For 1 fold)**: $\approx 5.15 \text{ hours}$
*   **Full 5-Fold Cross-Validation (10 Epochs per fold)**: **$\approx 25.75 \text{ hours}$ (approx. 1 day and 2 hours of continuous training)**.

> [!NOTE]
> Training time is highly dependent on batch size and dataloader optimization. By keeping the batch size at 8 (to avoid OOM errors) and workers at 4, the GPU is steadily saturated.
