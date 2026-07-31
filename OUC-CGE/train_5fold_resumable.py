#!/usr/bin/env python3
"""
5-Fold Cross-Validation Trainer with Pause/Resume Support.

FEATURES:
    - Trains SLOW 8x8 R50 for 25 epochs across all 5 folds
    - GPU training with mixed precision (FP16)
    - Pause:  Create a file named "PAUSE" in the OUC-CGE directory to pause
    - Resume: Delete the "PAUSE" file to continue training
    - Smart resume: If checkpoints exist, picks up from the last completed epoch
    - Fresh start: Use --fresh flag to wipe old checkpoints and start from scratch
    - Checkpoints saved every 5 epochs (epochs 5, 10, 15, 20, 25)
    - Validation runs every 5 epochs

USAGE:
    # Fresh start (wipes all old checkpoints):
    python train_5fold_resumable.py --fresh

    # Resume from where it stopped:
    python train_5fold_resumable.py

    # To PAUSE training while it's running:
    #   Create a file called "PAUSE" in E:\\Video_Engagement\\OUC-CGE\\
    #   e.g.:  echo. > E:\\Video_Engagement\\OUC-CGE\\PAUSE

    # To RESUME after pausing:
    #   Delete the PAUSE file, then re-run:
    #   del E:\\Video_Engagement\\OUC-CGE\\PAUSE
    #   python train_5fold_resumable.py
"""

import argparse
import json
import os
import shutil
import signal
import sys
import time

# ---------------------------------------------------------------------------
# Setup paths so SlowFast imports work
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = SCRIPT_DIR
TOOLS_DIR = os.path.join(REPO_ROOT, "tools")

sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, TOOLS_DIR)

from slowfast.config.defaults import get_cfg, assert_and_infer_cfg
from slowfast.utils.misc import launch_job
from train_net import train

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_FOLDS = 5
MAX_EPOCHS = 20
BASE_CONFIG = os.path.join(REPO_ROOT, "configs", "Kinetics", "5fold_SLOW_8x8_R50.yaml")
SPLITS_DIR = os.path.join(REPO_ROOT, "splits")
WORK_DIR = os.path.join(REPO_ROOT, "work_dirs")
PAUSE_FILE = os.path.join(REPO_ROOT, "PAUSE")
PROGRESS_FILE = os.path.join(WORK_DIR, "training_progress.json")


def log(msg):
    """Timestamped logging."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def save_progress(fold_id, status, info=None):
    """Save training progress to a JSON file for tracking."""
    progress = load_progress()
    progress[f"fold_{fold_id}"] = {
        "status": status,  # "not_started", "in_progress", "completed", "paused"
        "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
        "info": info or {},
    }
    os.makedirs(WORK_DIR, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def load_progress():
    """Load training progress from JSON file."""
    if os.path.isfile(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {}


def check_pause():
    """Check if the PAUSE file exists. If so, wait until it's removed."""
    if not os.path.exists(PAUSE_FILE):
        return False

    log("=" * 60)
    log("PAUSE DETECTED! Training is paused.")
    log(f"To resume: delete the file '{PAUSE_FILE}'")
    log("  PowerShell: Remove-Item '{}'".format(PAUSE_FILE))
    log("  Or simply delete the PAUSE file manually.")
    log("=" * 60)

    while os.path.exists(PAUSE_FILE):
        time.sleep(5)  # Check every 5 seconds

    log("PAUSE file removed. Resuming training...")
    return True


def get_last_checkpoint_epoch(fold_output_dir):
    """
    Find the last checkpoint epoch number in a fold's checkpoint directory.
    Returns the epoch number (int) or -1 if no checkpoints found.
    """
    ckpt_dir = os.path.join(fold_output_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return -1

    max_epoch = -1
    for fname in os.listdir(ckpt_dir):
        if fname.startswith("checkpoint_epoch_") and fname.endswith(".pyth"):
            try:
                # checkpoint_epoch_00005.pyth -> 5
                epoch_str = fname.replace("checkpoint_epoch_", "").replace(".pyth", "")
                epoch = int(epoch_str)
                if epoch > max_epoch:
                    max_epoch = epoch
            except ValueError:
                continue

    return max_epoch


def is_fold_completed(fold_id):
    """Check if a fold has completed all MAX_EPOCHS epochs."""
    fold_dir = os.path.join(WORK_DIR, f"fold_{fold_id}")
    last_epoch = get_last_checkpoint_epoch(fold_dir)
    return last_epoch >= MAX_EPOCHS


def clean_fold(fold_id):
    """Remove all training artifacts for a specific fold."""
    fold_dir = os.path.join(WORK_DIR, f"fold_{fold_id}")
    if os.path.isdir(fold_dir):
        shutil.rmtree(fold_dir)
        log(f"  Cleaned fold {fold_id}: removed {fold_dir}")
    else:
        log(f"  Fold {fold_id}: nothing to clean")


def extract_best_accuracy(fold_output_dir):
    """Parse the json_stats.log to find the best validation accuracy."""
    log_path = os.path.join(fold_output_dir, "json_stats.log")
    if not os.path.isfile(log_path):
        return {}

    best_top1_err = float("inf")
    best_epoch = -1

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if '"_type": "val_epoch"' in line or '"_type":"val_epoch"' in line:
                try:
                    json_start = line.index("{")
                    json_str = line[json_start:].strip()
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


def build_cfg_for_fold(fold_id):
    """Build a SlowFast CfgNode configured for a specific fold."""
    cfg = get_cfg()
    cfg.merge_from_file(BASE_CONFIG)

    fold_split_dir = os.path.join(SPLITS_DIR, f"fold_{fold_id}")
    fold_output_dir = os.path.join(WORK_DIR, f"fold_{fold_id}")

    merge_list = [
        "DATA.PATH_TO_DATA_DIR", fold_split_dir,
        "DATA.PATH_PREFIX", "",
        "OUTPUT_DIR", fold_output_dir,
        "RNG_SEED", str(fold_id),
        "SOLVER.MAX_EPOCH", str(MAX_EPOCHS),
        "TRAIN.EVAL_PERIOD", "5",
        "TRAIN.CHECKPOINT_PERIOD", "5",
        "TRAIN.AUTO_RESUME", "True",  # Auto-resume from last checkpoint
    ]

    cfg.merge_from_list(merge_list)
    cfg = assert_and_infer_cfg(cfg)
    return cfg


def train_fold(fold_id):
    """Train a single fold with pause-checking integrated."""
    fold_output_dir = os.path.join(WORK_DIR, f"fold_{fold_id}")
    last_epoch = get_last_checkpoint_epoch(fold_output_dir)

    if last_epoch >= MAX_EPOCHS:
        log(f"Fold {fold_id}: Already completed ({last_epoch}/{MAX_EPOCHS} epochs). Skipping.")
        return True

    if last_epoch > 0:
        log(f"Fold {fold_id}: Resuming from epoch {last_epoch} (checkpoint found)")
    else:
        log(f"Fold {fold_id}: Starting fresh (no checkpoints found)")

    # Check for pause BEFORE starting training
    check_pause()

    cfg = build_cfg_for_fold(fold_id)

    log(f"  Config summary:")
    log(f"    Split dir : {cfg.DATA.PATH_TO_DATA_DIR}")
    log(f"    Output dir: {cfg.OUTPUT_DIR}")
    log(f"    Max epochs: {cfg.SOLVER.MAX_EPOCH}")
    log(f"    Batch size: {cfg.TRAIN.BATCH_SIZE}")
    log(f"    Eval every: {cfg.TRAIN.EVAL_PERIOD} epochs")
    log(f"    Mixed prec: {cfg.TRAIN.MIXED_PRECISION}")
    log(f"    Auto resume: {cfg.TRAIN.AUTO_RESUME}")

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    save_progress(fold_id, "in_progress", {
        "started_from_epoch": last_epoch + 1 if last_epoch > 0 else 0,
        "target_epochs": MAX_EPOCHS,
    })

    try:
        # Launch training (single GPU, single process)
        launch_job(cfg=cfg, init_method="tcp://localhost:9999", func=train)

        # Training completed for this fold
        best_info = extract_best_accuracy(cfg.OUTPUT_DIR)
        save_progress(fold_id, "completed", best_info)

        if best_info:
            log(f"Fold {fold_id} COMPLETED: Best val accuracy = {best_info['best_top1_acc']:.2f}% (epoch {best_info['best_epoch']})")
        else:
            log(f"Fold {fold_id} COMPLETED (could not parse val accuracy from logs)")

        return True

    except KeyboardInterrupt:
        log(f"Fold {fold_id}: Training interrupted by user (Ctrl+C)")
        save_progress(fold_id, "paused", {
            "last_checkpoint_epoch": get_last_checkpoint_epoch(fold_output_dir),
            "reason": "keyboard_interrupt",
        })
        return False

    except Exception as e:
        log(f"Fold {fold_id}: Training ERROR: {e}")
        save_progress(fold_id, "error", {
            "last_checkpoint_epoch": get_last_checkpoint_epoch(fold_output_dir),
            "error": str(e),
        })
        raise


def main():
    parser = argparse.ArgumentParser(
        description="5-Fold CV Training with Pause/Resume support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Fresh start (wipes old checkpoints):
  python train_5fold_resumable.py --fresh

  # Resume from where it stopped:
  python train_5fold_resumable.py

  # Train only specific folds:
  python train_5fold_resumable.py --folds 0 1 2

  # To PAUSE during training:
  #   Create a file: echo. > PAUSE
  # To RESUME:
  #   Delete it:     del PAUSE
  #   Then re-run:   python train_5fold_resumable.py
        """,
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Delete all old checkpoints and start training from scratch"
    )
    parser.add_argument(
        "--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
        help="Which folds to train (default: all 5 folds, 0-4)"
    )
    args = parser.parse_args()

    # Remove PAUSE file if it exists at startup (clean state)
    if os.path.exists(PAUSE_FILE):
        log(f"NOTE: Removing stale PAUSE file from previous run.")
        os.remove(PAUSE_FILE)

    log("=" * 70)
    log("  OUC-CGE 5-Fold Cross-Validation Training")
    log("  Model : SLOW 8x8 R50  |  Device: cuda:0")
    log(f"  Epochs: {MAX_EPOCHS} per fold  |  Folds: {args.folds}")
    log(f"  Mode  : {'FRESH START' if args.fresh else 'RESUME (smart)'}")
    log("=" * 70)

    # -----------------------------------------------------------------------
    # Step 1: Verify splits exist
    # -----------------------------------------------------------------------
    for fold_id in args.folds:
        fold_dir = os.path.join(SPLITS_DIR, f"fold_{fold_id}")
        train_csv = os.path.join(fold_dir, "train.csv")
        val_csv = os.path.join(fold_dir, "val.csv")
        if not os.path.isfile(train_csv) or not os.path.isfile(val_csv):
            log(f"ERROR: Split files not found for fold {fold_id} in {fold_dir}")
            log("Please run generate_fold_splits.py first!")
            sys.exit(1)
    log("Split files verified for all target folds.")

    # -----------------------------------------------------------------------
    # Step 2: Fresh start — clean old checkpoints if requested
    # -----------------------------------------------------------------------
    if args.fresh:
        log("")
        log("FRESH START: Cleaning old training artifacts...")
        for fold_id in args.folds:
            clean_fold(fold_id)
        # Also remove old progress file
        if os.path.isfile(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)
        log("All old checkpoints removed. Starting from epoch 0.")
    else:
        log("")
        log("SMART RESUME: Checking existing checkpoints...")
        for fold_id in args.folds:
            fold_dir = os.path.join(WORK_DIR, f"fold_{fold_id}")
            last_epoch = get_last_checkpoint_epoch(fold_dir)
            if last_epoch >= MAX_EPOCHS:
                log(f"  Fold {fold_id}: COMPLETED ({last_epoch}/{MAX_EPOCHS} epochs)")
            elif last_epoch > 0:
                log(f"  Fold {fold_id}: Will RESUME from epoch {last_epoch}")
            else:
                log(f"  Fold {fold_id}: Will START FRESH (no checkpoints)")

    # -----------------------------------------------------------------------
    # Step 3: Train each fold
    # -----------------------------------------------------------------------
    log("")
    log("PAUSE INSTRUCTIONS:")
    log(f"  To pause:  Create file '{PAUSE_FILE}'")
    log(f"  To resume: Delete that file, then re-run this script.")
    log("")

    fold_results = {}
    total_start = time.time()
    all_completed = True

    for fold_id in args.folds:
        log("")
        log("=" * 70)
        log(f"  FOLD {fold_id} / {len(args.folds) - 1}")
        log("=" * 70)

        fold_start = time.time()
        completed = train_fold(fold_id)

        fold_elapsed = time.time() - fold_start
        log(f"Fold {fold_id} took {fold_elapsed / 60:.1f} minutes")

        if not completed:
            all_completed = False
            log(f"Fold {fold_id} was interrupted. Re-run this script to resume.")
            break

        # Gather results
        fold_dir = os.path.join(WORK_DIR, f"fold_{fold_id}")
        best_info = extract_best_accuracy(fold_dir)
        fold_results[fold_id] = best_info

    total_elapsed = time.time() - total_start

    # -----------------------------------------------------------------------
    # Step 4: Summary
    # -----------------------------------------------------------------------
    log("")
    log("=" * 70)
    if all_completed:
        log("  ALL TARGET FOLDS COMPLETED!")
    else:
        log("  TRAINING PAUSED/INTERRUPTED")
    log(f"  Total time this session: {total_elapsed / 3600:.2f} hours")
    log("=" * 70)

    log("")
    log("  Per-Fold Results:")
    log("  " + "-" * 50)
    accs = []
    for fold_id in args.folds:
        info = fold_results.get(fold_id, {})
        acc = info.get("best_top1_acc", "N/A")
        epoch = info.get("best_epoch", "N/A")
        fold_dir = os.path.join(WORK_DIR, f"fold_{fold_id}")
        last_ckpt = get_last_checkpoint_epoch(fold_dir)
        log(f"  Fold {fold_id}: Best Acc = {acc}%  (epoch {epoch})  |  Last checkpoint: epoch {last_ckpt}")
        if isinstance(acc, (int, float)):
            accs.append(acc)

    if accs:
        avg_acc = sum(accs) / len(accs)
        log(f"\n  Average accuracy: {avg_acc:.2f}%")

    if all_completed:
        log("\n  For detailed metrics (Macro-F1, QWK), run:")
        log("  python consolidate_metrics.py")
    else:
        log("\n  To resume training, simply re-run:")
        log("  python train_5fold_resumable.py")

    log("")


if __name__ == "__main__":
    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))
    main()
