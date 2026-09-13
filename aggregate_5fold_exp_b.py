import os
import pandas as pd
import numpy as np
from cross_dataset.training.metrics import MetricsCalculator

folds = [
    ("Fold 0", "outputs/exp_b_train_ouc_3d/predictions/test_predictions.csv"),
    ("Fold 1", "outputs/exp_b_train_ouc_3d_fold1/predictions/test_predictions.csv"),
    ("Fold 2", "outputs/exp_b_train_ouc_3d_fold2/predictions/test_predictions.csv"),
    ("Fold 3", "outputs/exp_b_train_ouc_3d_fold3/predictions/test_predictions.csv"),
    ("Fold 4", "outputs/exp_b_train_ouc_3d_fold4/predictions/test_predictions.csv"),
]

results = []
for name, p in folds:
    if os.path.exists(p):
        df = pd.read_csv(p)
        # Check column names
        gt_col = "ground_truth" if "ground_truth" in df.columns else "target"
        pred_col = "predicted_label" if "predicted_label" in df.columns else "prediction"
        y_true = df[gt_col].values
        y_pred = df[pred_col].values
        m = MetricsCalculator.compute_metrics(y_true, y_pred, num_classes=3)
        results.append({
            "fold": name,
            "samples": len(df),
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "qwk": m["qwk"],
            "balanced_accuracy": m["balanced_accuracy"],
            "class_accuracy": m["class_accuracy"],
            "confusion_matrix": m["confusion_matrix"],
        })
        print(f"{name} (N={len(df)}): Acc={m['accuracy']:.4f}, MacroF1={m['macro_f1']:.4f}, QWK={m['qwk']:.4f}, BalAcc={m['balanced_accuracy']:.4f}")

accs = [r["accuracy"] for r in results]
f1s = [r["macro_f1"] for r in results]
qwks = [r["qwk"] for r in results]
bal_accs = [r["balanced_accuracy"] for r in results]

print("\n==================================================")
print(" EXPERIMENT B (3D R(2+1)D-18) 5-FOLD CV SUMMARY")
print("==================================================")
print(f"Mean Accuracy:         {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
print(f"Mean Macro F1:         {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
print(f"Mean QWK:              {np.mean(qwks):.4f} +/- {np.std(qwks):.4f}")
print(f"Mean Balanced Acc:     {np.mean(bal_accs):.4f} +/- {np.std(bal_accs):.4f}")
print("==================================================")
