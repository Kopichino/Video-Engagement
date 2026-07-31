"""
Master Manifest Builder for Cross-Dataset Pipeline.

Scans OUC-CGE and COLER datasets, extracts standardized metadata, performs
5-fold stratified splitting per dataset, and generates unified_manifest.csv.
"""

import csv
import os
import random
import sys
from typing import List, Dict, Any

from cross_dataset.config.base_config import load_config
from cross_dataset.preprocessing.coler_parser import COLERParser
from cross_dataset.preprocessing.ouc_cge_parser import OUCCGEParser
from cross_dataset.utils.logger import get_logger

logger = get_logger("Build_Manifest")


def assign_stratified_folds(
    records: List[Dict[str, Any]], num_folds: int = 5, seed: int = 37247
) -> List[Dict[str, Any]]:
    """
    Assigns stratified fold indices (0 to num_folds - 1) to dataset records
    based on their 3-class target label to ensure class balance across folds.

    Args:
        records: List of sample metadata records.
        num_folds: Number of folds (default: 5).
        seed: Random seed for reproducibility.

    Returns:
        List of records with updated 'split_fold' field.
    """
    random.seed(seed)

    # Group record indices by class label
    class_groups: Dict[int, List[int]] = {}
    for idx, rec in enumerate(records):
        label = rec["label_3class"]
        if label not in class_groups:
            class_groups[label] = []
        class_groups[label].append(idx)

    # Shuffle each class group independently and assign fold indices
    for label, indices in class_groups.items():
        random.shuffle(indices)
        for rank, idx in enumerate(indices):
            fold = rank % num_folds
            records[idx]["split_fold"] = fold

    return records


def build_unified_manifest(config_path: str = None) -> str:
    """
    Main function to parse both datasets, assign folds, and export unified_manifest.csv.

    Args:
        config_path: Optional path to custom JSON config.

    Returns:
        str: Absolute path to the generated unified_manifest.csv.
    """
    cfg = load_config(config_path)

    manifest_path = cfg.paths.manifest_path
    manifest_dir = os.path.dirname(manifest_path)
    os.makedirs(manifest_dir, exist_ok=True)

    logger.info("Starting Unified Manifest Construction...")

    # 1. Parse OUC-CGE Dataset
    ouc_parser = OUCCGEParser(
        dataset_root=cfg.paths.ouc_cge_root,
        validate_videos=cfg.preprocessing.validate_videos,
    )
    ouc_records, ouc_corrupted = ouc_parser.scan()

    # 2. Parse COLER Dataset
    coler_parser = COLERParser(dataset_root=cfg.paths.coler_root)
    coler_records, coler_empty = coler_parser.scan()

    if not ouc_records and not coler_records:
        logger.error("No valid records found in either dataset!")
        return ""

    # 3. Assign 5-Fold Stratified Splits independently per dataset
    ouc_records = assign_stratified_folds(
        ouc_records,
        num_folds=cfg.preprocessing.num_folds,
        seed=cfg.preprocessing.random_seed,
    )
    coler_records = assign_stratified_folds(
        coler_records,
        num_folds=cfg.preprocessing.num_folds,
        seed=cfg.preprocessing.random_seed,
    )

    combined_records = ouc_records + coler_records

    # 4. Export to CSV Manifest
    csv_headers = [
        "sample_id",
        "dataset_name",
        "relative_path",
        "file_type",
        "num_frames",
        "fps",
        "width",
        "height",
        "duration_sec",
        "raw_label",
        "label_3class",
        "label_binary",
        "split_fold",
    ]

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(combined_records)

    logger.info(f"Unified Manifest successfully written to: {manifest_path}")
    logger.info(f"Total Combined Samples: {len(combined_records)}")
    logger.info(f"  - OUC-CGE Samples: {len(ouc_records)}")
    logger.info(f"  - COLER Samples:   {len(coler_records)}")

    # Save corrupted files log if any
    if ouc_corrupted:
        corrupted_log = os.path.join(manifest_dir, "corrupted_files.txt")
        with open(corrupted_log, "w", encoding="utf-8") as f:
            f.write("\n".join(ouc_corrupted))
        logger.info(f"Corrupted video list saved to: {corrupted_log}")

    return manifest_path


if __name__ == "__main__":
    build_unified_manifest()
