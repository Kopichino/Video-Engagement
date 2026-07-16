#!/usr/bin/env python3
"""
Consolidate 5-fold cross-validation metrics for OUC-CGE.

For each fold, loads the best checkpoint and runs inference on the validation
set to compute:
  - Accuracy
  - Macro F1-Score
  - Quadratic Weighted Kappa (QWK)

Then prints and saves a summary table with per-fold and averaged metrics.

Usage:
    python consolidate_metrics.py
"""

import json
import os
import sys
import time
from collections import Counter

# ---------------------------------------------------------------------------
# Setup paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = SCRIPT_DIR
TOOLS_DIR = os.path.join(REPO_ROOT, "tools")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, TOOLS_DIR)

import numpy as np
import torch
import torch.nn.functional as F

from slowfast.config.defaults import get_cfg, assert_and_infer_cfg
from slowfast.datasets import loader
from slowfast.models import build_model
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_FOLDS = 5
NUM_CLASSES = 3
CLASS_NAMES = ["Low", "Mid", "High"]
BASE_CONFIG = os.path.join(REPO_ROOT, "configs", "Kinetics", "5fold_SLOW_8x8_R50.yaml")
SPLITS_DIR = os.path.join(REPO_ROOT, "splits")
WORK_DIR = os.path.join(REPO_ROOT, "work_dirs")
SUMMARY_FILE = os.path.join(WORK_DIR, "5fold_summary.txt")


# ===================================================================
# Metric Computation (no sklearn dependency)
# ===================================================================

def compute_accuracy(y_true, y_pred):
    """Compute overall accuracy."""
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true) if y_true else 0.0


def compute_macro_f1(y_true, y_pred, num_classes=NUM_CLASSES):
    """Compute macro-averaged F1 score."""
    f1_scores = []
    for c in range(num_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


def compute_qwk(y_true, y_pred, num_classes=NUM_CLASSES):
    """
    Compute Quadratic Weighted Kappa (Cohen's kappa with quadratic weights).

    QWK = 1 - sum(W * O) / sum(W * E)
    where:
      W[i,j] = (i - j)^2 / (num_classes - 1)^2   (quadratic weight matrix)
      O = observed confusion matrix (normalized)
      E = expected confusion matrix under independence (normalized)
    """
    n = len(y_true)
    if n == 0:
        return 0.0

    # Build confusion matrix
    O = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        O[t][p] += 1

    # Weight matrix (quadratic)
    W = [[0.0] * num_classes for _ in range(num_classes)]
    for i in range(num_classes):
        for j in range(num_classes):
            W[i][j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)

    # Row and column marginals of the observed matrix
    row_sums = [sum(O[i]) for i in range(num_classes)]
    col_sums = [sum(O[i][j] for i in range(num_classes)) for j in range(num_classes)]

    # Expected matrix (outer product of marginals, normalized)
    E = [[0.0] * num_classes for _ in range(num_classes)]
    for i in range(num_classes):
        for j in range(num_classes):
            E[i][j] = (row_sums[i] * col_sums[j]) / n

    # Weighted sums
    weighted_observed = sum(W[i][j] * O[i][j] for i in range(num_classes) for j in range(num_classes))
    weighted_expected = sum(W[i][j] * E[i][j] for i in range(num_classes) for j in range(num_classes))

    if weighted_expected == 0:
        return 1.0  # Perfect agreement

    return 1.0 - (weighted_observed / weighted_expected)


def compute_confusion_matrix(y_true, y_pred, num_classes=NUM_CLASSES):
    """Compute confusion matrix."""
    cm = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


# ===================================================================
# Inference
# ===================================================================

def find_best_checkpoint(fold_output_dir):
    """Find the last checkpoint in the fold directory (best = last saved)."""
    if not os.path.isdir(fold_output_dir):
        return None
    ckpt_dir = os.path.join(fold_output_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        # SlowFast may save directly in output_dir
        ckpt_dir = fold_output_dir

    candidates = []
    for fname in os.listdir(ckpt_dir):
        if fname.endswith(".pyth"):
            # Parse epoch number from filename: checkpoint_epoch_XXXXX.pyth
            try:
                parts = fname.replace(".pyth", "").split("_")
                epoch = int(parts[-1])
                candidates.append((epoch, os.path.join(ckpt_dir, fname)))
            except (ValueError, IndexError):
                candidates.append((-1, os.path.join(ckpt_dir, fname)))

    if not candidates:
        return None

    # Return highest epoch checkpoint
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def run_inference_for_fold(fold_id):
    """
    Load best checkpoint for a fold and run inference on its val set.
    Returns (all_true_labels, all_pred_labels).
    """
    cfg = get_cfg()
    cfg.merge_from_file(BASE_CONFIG)

    fold_split_dir = os.path.join(SPLITS_DIR, f"fold_{fold_id}")
    fold_output_dir = os.path.join(WORK_DIR, f"fold_{fold_id}")

    cfg.merge_from_list([
        "DATA.PATH_TO_DATA_DIR", fold_split_dir,
        "DATA.PATH_PREFIX", "",
        "OUTPUT_DIR", fold_output_dir,
        "TRAIN.ENABLE", "False",
        "TEST.ENABLE", "True",
        "TEST.BATCH_SIZE", "8",
        "TEST.NUM_ENSEMBLE_VIEWS", "1",
        "TEST.NUM_SPATIAL_CROPS", "1",
        "NUM_GPUS", "1",
    ])
    cfg = assert_and_infer_cfg(cfg)

    # Find checkpoint
    ckpt_path = find_best_checkpoint(fold_output_dir)
    if ckpt_path is None:
        print(f"  WARNING: No checkpoint found for fold {fold_id} in {fold_output_dir}")
        return [], []

    print(f"  Loading checkpoint: {ckpt_path}")

    # Build model and load weights
    model = build_model(cfg)
    cu.load_checkpoint(ckpt_path, model, cfg.NUM_GPUS > 1)
    model.eval()

    # Build val loader
    val_loader = loader.construct_loader(cfg, "val")

    all_true = []
    all_pred = []

    with torch.no_grad():
        for inputs, labels, index, time_idx, meta in val_loader:
            if cfg.NUM_GPUS:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
                labels = labels.cuda()

            preds = model(inputs)

            # Get predicted class
            pred_classes = preds.argmax(dim=1).cpu().numpy().tolist()
            true_classes = labels.cpu().numpy().tolist()

            all_true.extend(true_classes)
            all_pred.extend(pred_classes)

    return all_true, all_pred


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 70)
    print("  OUC-CGE 5-Fold Cross-Validation — Metrics Consolidation")
    print("=" * 70)
    print()

    all_fold_metrics = []
    report_lines = []

    report_lines.append("=" * 70)
    report_lines.append("  OUC-CGE 5-Fold Cross-Validation Results")
    report_lines.append("  Model: SLOW 8x8 R50")
    report_lines.append(f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 70)
    report_lines.append("")

    for fold_id in range(NUM_FOLDS):
        print(f"\n--- Fold {fold_id} ---")
        y_true, y_pred = run_inference_for_fold(fold_id)

        if not y_true:
            print(f"  Skipping fold {fold_id} (no predictions)")
            all_fold_metrics.append({"fold": fold_id, "acc": None, "f1": None, "qwk": None})
            continue

        acc = compute_accuracy(y_true, y_pred)
        f1 = compute_macro_f1(y_true, y_pred)
        qwk = compute_qwk(y_true, y_pred)
        cm = compute_confusion_matrix(y_true, y_pred)

        metrics = {"fold": fold_id, "acc": acc, "f1": f1, "qwk": qwk, "n_samples": len(y_true)}
        all_fold_metrics.append(metrics)

        print(f"  Samples: {len(y_true)}")
        print(f"  Accuracy:  {acc * 100:.2f}%")
        print(f"  Macro-F1:  {f1 * 100:.2f}%")
        print(f"  QWK:       {qwk:.4f}")

        # Confusion matrix
        report_lines.append(f"Fold {fold_id}  (n={len(y_true)})")
        report_lines.append(f"  Accuracy:  {acc * 100:.2f}%")
        report_lines.append(f"  Macro-F1:  {f1 * 100:.2f}%")
        report_lines.append(f"  QWK:       {qwk:.4f}")
        report_lines.append(f"  Confusion Matrix:")
        header = "        " + "  ".join(f"{CLASS_NAMES[j]:>6s}" for j in range(NUM_CLASSES))
        report_lines.append(header)
        for i in range(NUM_CLASSES):
            row = f"  {CLASS_NAMES[i]:>4s}  " + "  ".join(f"{cm[i][j]:>6d}" for j in range(NUM_CLASSES))
            report_lines.append(row)
        report_lines.append("")

    # ---------------------------------------------------------------
    # Compute averages
    # ---------------------------------------------------------------
    valid_metrics = [m for m in all_fold_metrics if m["acc"] is not None]

    if valid_metrics:
        avg_acc = sum(m["acc"] for m in valid_metrics) / len(valid_metrics)
        avg_f1 = sum(m["f1"] for m in valid_metrics) / len(valid_metrics)
        avg_qwk = sum(m["qwk"] for m in valid_metrics) / len(valid_metrics)

        std_acc = (sum((m["acc"] - avg_acc) ** 2 for m in valid_metrics) / len(valid_metrics)) ** 0.5
        std_f1 = (sum((m["f1"] - avg_f1) ** 2 for m in valid_metrics) / len(valid_metrics)) ** 0.5
        std_qwk = (sum((m["qwk"] - avg_qwk) ** 2 for m in valid_metrics) / len(valid_metrics)) ** 0.5
    else:
        avg_acc = avg_f1 = avg_qwk = 0.0
        std_acc = std_f1 = std_qwk = 0.0

    # ---------------------------------------------------------------
    # Final Summary Table
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  FINAL 5-FOLD CROSS-VALIDATION SUMMARY")
    print("=" * 70)

    header = f"{'Fold':>6s}  {'Accuracy':>10s}  {'Macro-F1':>10s}  {'QWK':>8s}"
    sep = "-" * len(header)
    print(header)
    print(sep)

    report_lines.append("=" * 70)
    report_lines.append("  SUMMARY")
    report_lines.append("=" * 70)
    report_lines.append(header)
    report_lines.append(sep)

    for m in all_fold_metrics:
        if m["acc"] is not None:
            line = f"  {m['fold']:>4d}  {m['acc'] * 100:>9.2f}%  {m['f1'] * 100:>9.2f}%  {m['qwk']:>8.4f}"
        else:
            line = f"  {m['fold']:>4d}  {'N/A':>10s}  {'N/A':>10s}  {'N/A':>8s}"
        print(line)
        report_lines.append(line)

    print(sep)
    avg_line = f"  {'Avg':>4s}  {avg_acc * 100:>9.2f}%  {avg_f1 * 100:>9.2f}%  {avg_qwk:>8.4f}"
    std_line = f"  {'Std':>4s}  {std_acc * 100:>9.2f}%  {std_f1 * 100:>9.2f}%  {std_qwk:>8.4f}"
    print(avg_line)
    print(std_line)
    print("=" * 70)

    report_lines.append(sep)
    report_lines.append(avg_line)
    report_lines.append(std_line)
    report_lines.append("=" * 70)

    # ---------------------------------------------------------------
    # Save to file
    # ---------------------------------------------------------------
    os.makedirs(os.path.dirname(SUMMARY_FILE), exist_ok=True)
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"\nReport saved to: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
