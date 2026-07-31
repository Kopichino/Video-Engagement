#!/usr/bin/env python3
"""
Generate 5-fold train/val split CSV files for the SlowFast Kinetics data loader.

This script replicates the full preprocessing.ipynb pipeline:
  1. Scans all .mp4 files under high/, mid/, low/ in the dataset directory
  2. Validates each video with OpenCV to detect corrupted files
  3. Applies StratifiedKFold (5 splits) to create train/val partitions
  4. Balances each fold by downsampling majority classes to the minority count
  5. Writes per-fold train.csv / val.csv in SlowFast Kinetics format:
       <absolute_video_path> <integer_label>  (space-separated, no header)

Each run uses a different random seed (based on current time) so that the
random balancing produces different sample selections across runs.
To reproduce a specific split, set RANDOM_SEED below to a fixed integer.

Label mapping: Low -> 0, Medium -> 1, High -> 2

Dependencies: cv2 (opencv-python), numpy, sklearn
"""

import os
import sys
import random
import time

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = SCRIPT_DIR  # e.g. E:\Video_Engagement\OUC-CGE

# Where the actual video files live
DATASET_ROOT = os.path.normpath(REPO_ROOT + "-dataset")  # E:\Video_Engagement\OUC-CGE-dataset

# Output directory for per-fold splits
SPLITS_DIR = os.path.join(REPO_ROOT, "splits")

# Class folder name -> (display label, integer label)
CLASS_MAPPING = {
    "high": ("High", 2),
    "mid":  ("Medium", 1),
    "low":  ("Low", 0),
}

NUM_FOLDS = 5

# Set to None for a time-based random seed (different every run),
# or set to a fixed integer (e.g. 42) for reproducibility.
RANDOM_SEED = 37247


def collect_all_mp4(root_dir):
    """Recursively find all .mp4 files under root_dir."""
    mp4_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(".mp4"):
                mp4_files.append(os.path.join(dirpath, fname))
    return mp4_files


def check_video_readable(video_path):
    """
    Return True if OpenCV can read frames at multiple points throughout the video.
    This helps detect truncated or corrupted files that fail mid-stream or at the end.
    """
    import cv2
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            return False
        
        # Get frame count
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            return False

        # Check key frames across the duration (e.g. 0%, 25%, 50%, 75%, 95%)
        check_points = [0, int(frame_count * 0.25), int(frame_count * 0.50), int(frame_count * 0.75), max(0, frame_count - 5)]
        
        # Deduplicate and sort check points
        check_points = sorted(list(set(check_points)))
        
        for pt in check_points:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pt)
            ret, _ = cap.read()
            if not ret:
                cap.release()
                return False
                
        cap.release()
        return True
    except Exception:
        return False


def scan_and_verify_videos():
    """
    Scan dataset directory, validate each video, return list of
    (absolute_path, class_label_str, int_label) for valid videos.
    Also returns a list of corrupted file paths.
    """
    valid_videos = []
    corrupted_videos = []

    for folder_name, (class_label, int_label) in CLASS_MAPPING.items():
        class_dir = os.path.join(DATASET_ROOT, folder_name)
        if not os.path.isdir(class_dir):
            print(f"  WARNING: Class directory not found: {class_dir}")
            continue

        mp4_files = collect_all_mp4(class_dir)
        print(f"  Checking {class_label} class: {len(mp4_files)} files found...")

        valid_count = 0
        corrupt_count = 0
        for video_path in mp4_files:
            if check_video_readable(video_path):
                valid_videos.append((video_path, class_label, int_label))
                valid_count += 1
            else:
                corrupted_videos.append((video_path, class_label))
                corrupt_count += 1
                try:
                    print(f"    Corrupted file ignored: {os.path.basename(video_path)}")
                except UnicodeEncodeError:
                    print(f"    Corrupted file ignored: {os.path.basename(video_path).encode('ascii', 'replace').decode()}")

        print(f"    -> {valid_count} valid, {corrupt_count} corrupted")

    return valid_videos, corrupted_videos


def stratified_kfold_split(videos, n_splits, seed):
    """
    Apply StratifiedKFold to split videos into n_splits folds.
    Returns a list of (train_indices, val_indices) tuples.

    Uses sklearn.model_selection.StratifiedKFold.
    """
    from sklearn.model_selection import StratifiedKFold

    labels = [v[1] for v in videos]  # class_label strings for stratification
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X_dummy = list(range(len(videos)))

    folds = []
    for train_idx, val_idx in skf.split(X_dummy, labels):
        folds.append((train_idx.tolist(), val_idx.tolist()))

    return folds


def balance_subset(subset, seed):
    """
    Given a list of (path, class_label, int_label), downsample majority
    classes to match the minority class count.
    Returns a balanced list.
    """
    # Group by class
    by_class = {}
    for item in subset:
        cls = item[1]
        by_class.setdefault(cls, []).append(item)

    # Find minimum class count
    min_count = min(len(items) for items in by_class.values())

    # Sample each class down to min_count
    rng = random.Random(seed)
    balanced = []
    for cls in sorted(by_class.keys()):
        items = by_class[cls]
        if len(items) > min_count:
            sampled = rng.sample(items, min_count)
        else:
            sampled = items[:]
        balanced.extend(sampled)

    # Shuffle the final list
    rng.shuffle(balanced)
    return balanced


def main():
    # Determine random seed
    if RANDOM_SEED is not None:
        seed = RANDOM_SEED
    else:
        seed = int(time.time()) % 100000

    print("=" * 70)
    print("  OUC-CGE 5-Fold Split Generator")
    print(f"  Dataset root: {DATASET_ROOT}")
    print(f"  Random seed:  {seed}")
    print("=" * 70)
    print()

    # ------------------------------------------------------------------
    # Step 1: Scan and validate all videos
    # ------------------------------------------------------------------
    print("Step 1: Scanning and verifying videos...")
    valid_videos, corrupted_videos = scan_and_verify_videos()
    print()

    if not valid_videos:
        print("ERROR: No valid videos found! Check DATASET_ROOT path.")
        sys.exit(1)

    # Count per class
    class_counts = {}
    for _, cls, _ in valid_videos:
        class_counts[cls] = class_counts.get(cls, 0) + 1

    print(f"Total valid videos: {len(valid_videos)}")
    print(f"Total corrupted:    {len(corrupted_videos)}")
    for cls in ["Low", "Medium", "High"]:
        print(f"  {cls}: {class_counts.get(cls, 0)}")
    print()

    # Save corrupted file list for reference
    corrupt_log = os.path.join(SPLITS_DIR, "corrupted_files.txt")
    os.makedirs(SPLITS_DIR, exist_ok=True)
    with open(corrupt_log, "w", encoding="utf-8") as f:
        f.write(f"# Corrupted files detected (seed={seed})\n")
        f.write(f"# Total: {len(corrupted_videos)}\n\n")
        for path, cls in corrupted_videos:
            f.write(f"{cls}: {path}\n")
    print(f"Corrupted file list saved to: {corrupt_log}")
    print()

    # ------------------------------------------------------------------
    # Step 2: StratifiedKFold split
    # ------------------------------------------------------------------
    print("Step 2: Applying StratifiedKFold split...")
    folds = stratified_kfold_split(valid_videos, NUM_FOLDS, seed)
    print()

    # ------------------------------------------------------------------
    # Step 3: Balance and write per-fold CSVs
    # ------------------------------------------------------------------
    print("Step 3: Balancing and writing fold CSVs...")
    print()

    for fold_id, (train_idx, val_idx) in enumerate(folds):
        train_subset = [valid_videos[i] for i in train_idx]
        val_subset = [valid_videos[i] for i in val_idx]

        # Balance each subset (use fold_id + seed for per-fold randomness)
        train_balanced = balance_subset(train_subset, seed=seed + fold_id)
        val_balanced = balance_subset(val_subset, seed=seed + fold_id + NUM_FOLDS)

        # Write CSVs
        fold_dir = os.path.join(SPLITS_DIR, f"fold_{fold_id}")
        os.makedirs(fold_dir, exist_ok=True)

        for phase, entries in [("train", train_balanced), ("val", val_balanced)]:
            out_path = os.path.join(fold_dir, f"{phase}.csv")
            with open(out_path, "w", encoding="utf-8", newline="") as f:
                for video_path, class_label, int_label in entries:
                    f.write(f"{video_path} {int_label}\n")

            # Count distribution
            label_counts = {}
            for _, _, lbl in entries:
                label_counts[lbl] = label_counts.get(lbl, 0) + 1

            print(
                f"  Fold {fold_id} {phase:5s}: {len(entries):5d} samples  "
                f"(Low={label_counts.get(0, 0)}, Mid={label_counts.get(1, 0)}, "
                f"High={label_counts.get(2, 0)})  -> {out_path}"
            )
        print()

    # ------------------------------------------------------------------
    # Step 4: Spot-check
    # ------------------------------------------------------------------
    print("--- Spot-check: verifying sample video paths exist ---")
    fold_0_train_csv = os.path.join(SPLITS_DIR, "fold_0", "train.csv")
    check_count = 0
    missing_count = 0
    with open(fold_0_train_csv, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if line_no >= 6:
                break
            parts = line.strip().rsplit(" ", 1)
            if len(parts) != 2:
                continue
            video_path = parts[0]
            check_count += 1
            exists = os.path.isfile(video_path)
            status = "OK" if exists else "MISSING"
            if not exists:
                missing_count += 1
            try:
                print(f"  [{status}] {video_path}")
            except UnicodeEncodeError:
                print(f"  [{status}] {video_path.encode('ascii', 'replace').decode()}")

    if missing_count > 0:
        print(f"\nWARNING: {missing_count}/{check_count} checked paths are MISSING!")
    else:
        print(f"\nAll {check_count} checked paths exist on disk.")

    # Save the seed used for this run
    seed_file = os.path.join(SPLITS_DIR, "seed.txt")
    with open(seed_file, "w") as f:
        f.write(f"{seed}\n")
    print(f"\nRandom seed saved to: {seed_file}")
    print(f"(To reproduce this exact split, set RANDOM_SEED = {seed} in the script)")

    print("\nDone! Split files generated in:", SPLITS_DIR)


if __name__ == "__main__":
    main()
