#!/usr/bin/env python3
"""
5-Fold Cross-Validation Master Runner for OUC-CGE.

Loops through all 5 folds, configures the SlowFast training pipeline for each,
trains the SLOW 8x8 R50 model, and at the end runs metrics consolidation.

Usage:
    python run_5fold.py
"""

import json
import os
import sys
import time

# ---------------------------------------------------------------------------
# Setup paths so SlowFast imports work (same as tools/run_net.py)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = SCRIPT_DIR
TOOLS_DIR = os.path.join(REPO_ROOT, "tools")

# Add repo root and tools to path so that slowfast and tool modules are importable
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, TOOLS_DIR)

from slowfast.config.defaults import get_cfg, assert_and_infer_cfg
from slowfast.utils.misc import launch_job
from train_net import train

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_FOLDS = 5
BASE_CONFIG = os.path.join(REPO_ROOT, "configs", "Kinetics", "5fold_SLOW_8x8_R50.yaml")
SPLITS_DIR = os.path.join(REPO_ROOT, "splits")
WORK_DIR = os.path.join(REPO_ROOT, "work_dirs")


def build_cfg_for_fold(fold_id: int, max_epochs: int = None, eval_period: int = None, checkpoint_period: int = None):
    """Build a SlowFast CfgNode configured for a specific fold."""
    cfg = get_cfg()
    cfg.merge_from_file(BASE_CONFIG)

    # Per-fold overrides
    fold_split_dir = os.path.join(SPLITS_DIR, f"fold_{fold_id}")
    fold_output_dir = os.path.join(WORK_DIR, f"fold_{fold_id}")

    merge_list = [
        "DATA.PATH_TO_DATA_DIR", fold_split_dir,
        "DATA.PATH_PREFIX", "",          # CSVs contain absolute paths
        "OUTPUT_DIR", fold_output_dir,
        "RNG_SEED", str(fold_id),        # Different seed per fold for variety
    ]

    if max_epochs is not None:
        merge_list.extend(["SOLVER.MAX_EPOCH", max_epochs])
    if eval_period is not None:
        merge_list.extend(["TRAIN.EVAL_PERIOD", eval_period])
    if checkpoint_period is not None:
        merge_list.extend(["TRAIN.CHECKPOINT_PERIOD", checkpoint_period])

    cfg.merge_from_list(merge_list)

    cfg = assert_and_infer_cfg(cfg)
    return cfg


def extract_best_accuracy(fold_output_dir: str) -> dict:
    """
    Parse the SlowFast stdout.log in the fold output directory to extract
    the best (minimum) top-1 error from validation epochs.

    Returns a dict with best_top1_acc and best_epoch, or empty dict if not found.
    """
    log_path = os.path.join(fold_output_dir, "stdout.log")
    if not os.path.isfile(log_path):
        return {}

    best_top1_err = float("inf")
    best_epoch = -1

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            # Look for validation epoch stats lines
            # Format: "json_stats: {"_type": "val_epoch", ...}"
            if '"_type": "val_epoch"' in line:
                try:
                    # Extract the JSON portion
                    json_start = line.index("{")
                    json_str = line[json_start:]
                    stats = json.loads(json_str)
                    top1_err = stats.get("top1_err", float("inf"))
                    epoch_raw = stats.get("epoch", -1)
                    if isinstance(epoch_raw, str):
                        epoch = int(epoch_raw.split("/")[0]) if "/" in epoch_raw else int(epoch_raw)
                    else:
                        epoch = int(epoch_raw)
                    if top1_err < best_top1_err:
                        best_top1_err = top1_err
                        best_epoch = epoch
                except (ValueError, json.JSONDecodeError):
                    continue

    if best_epoch >= 0:
        return {
            "best_top1_acc": round(100.0 - best_top1_err, 2),
            "best_top1_err": round(best_top1_err, 2),
            "best_epoch": best_epoch,
        }
    return {}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="5-Fold Cross-Validation Master Runner")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="List of fold IDs to train (default: all 5 folds, 0-4)")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Override SOLVER.MAX_EPOCH")
    parser.add_argument("--eval-period", type=int, default=None,
                        help="Override TRAIN.EVAL_PERIOD")
    parser.add_argument("--checkpoint-period", type=int, default=None,
                        help="Override TRAIN.CHECKPOINT_PERIOD")
    args = parser.parse_args()

    print("=" * 70)
    print("  OUC-CGE 5-Fold Cross-Validation Training")
    print("  Model: SLOW 8x8 R50  |  Device: cuda:0")
    print(f"  Target Folds: {args.folds}")
    print("=" * 70)

    # Verify splits exist for target folds
    for fold_id in args.folds:
        fold_dir = os.path.join(SPLITS_DIR, f"fold_{fold_id}")
        train_csv = os.path.join(fold_dir, "train.csv")
        val_csv = os.path.join(fold_dir, "val.csv")
        if not os.path.isfile(train_csv) or not os.path.isfile(val_csv):
            print(f"ERROR: Split files not found for fold {fold_id} in {fold_dir}")
            print("Please run generate_fold_splits.py first!")
            sys.exit(1)
    print("Fold split files verified.\n")

    fold_results = {}
    total_start = time.time()

    for fold_id in args.folds:
        print("\n" + "=" * 70)
        print(f"  FOLD {fold_id} / 4")
        print("=" * 70)

        fold_start = time.time()
        cfg = build_cfg_for_fold(
            fold_id,
            max_epochs=args.max_epochs,
            eval_period=args.eval_period,
            checkpoint_period=args.checkpoint_period
        )

        print(f"  Split dir : {cfg.DATA.PATH_TO_DATA_DIR}")
        print(f"  Output dir: {cfg.OUTPUT_DIR}")
        print(f"  Max epochs: {cfg.SOLVER.MAX_EPOCH}")
        print(f"  Batch size: {cfg.TRAIN.BATCH_SIZE}")
        print(f"  Eval every: {cfg.TRAIN.EVAL_PERIOD} epochs")
        print()

        # Create output directory before starting training so log file creation doesn't fail
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

        # Launch training (single GPU, single process)
        launch_job(cfg=cfg, init_method="tcp://localhost:9999", func=train)

        fold_elapsed = time.time() - fold_start
        print(f"\n  Fold {fold_id} completed in {fold_elapsed / 60:.1f} minutes")

        # Extract best validation accuracy from logs
        best_info = extract_best_accuracy(cfg.OUTPUT_DIR)
        if best_info:
            print(f"  Best val accuracy: {best_info['best_top1_acc']:.2f}% "
                  f"(epoch {best_info['best_epoch']})")
            fold_results[fold_id] = best_info
        else:
            print("  WARNING: Could not parse validation accuracy from logs")
            fold_results[fold_id] = {}

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print("  TARGET FOLDS COMPLETED")
    print(f"  Total time: {total_elapsed / 3600:.1f} hours")
    print("=" * 70)

    # Print quick summary of per-fold accuracies
    print("\n  Quick Summary (from training logs):")
    print("  " + "-" * 40)
    accs = []
    for fold_id in args.folds:
        info = fold_results.get(fold_id, {})
        acc = info.get("best_top1_acc", "N/A")
        epoch = info.get("best_epoch", "N/A")
        print(f"  Fold {fold_id}: Best Acc = {acc}%  (epoch {epoch})")
        if isinstance(acc, (int, float)):
            accs.append(acc)
    if accs:
        avg_acc = sum(accs) / len(accs)
        print(f"\n  Average accuracy: {avg_acc:.2f}%")

    print("\n  For detailed metrics (Macro-F1, QWK), run:")
    print("  python consolidate_metrics.py")
    print()


if __name__ == "__main__":
    main()
