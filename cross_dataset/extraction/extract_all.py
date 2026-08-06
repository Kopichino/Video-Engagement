"""
Master Frame Extraction & Metadata Generation Runner.

Orchestrates frame extraction for OUC-CGE videos and sequence validation
for COLER frame trees based on unified_manifest.csv.
"""

import csv
import json
import os
import sys
import time
from typing import List, Dict, Any

from cross_dataset.config.base_config import load_config
from cross_dataset.extraction.coler_extractor import COLERFrameExtractor
from cross_dataset.extraction.ouc_extractor import OUCCGEFrameExtractor
from cross_dataset.extraction.samplers import SamplingMode
from cross_dataset.utils.logger import get_logger

logger = get_logger("Extract_All")


def run_extraction_pipeline(
    num_frames: int = 3,
    sampling_mode: str = "uniform",
    max_samples: int = 0,
    config_path: str = None,
    log_interval: int = 250,
) -> str:
    """
    Executes frame extraction and sequence validation across datasets.

    Args:
        num_frames: Number of frames N to extract per OUC-CGE sample.
        sampling_mode: Sampling mode ('single', 'three', 'uniform', 'center', 'random').
        max_samples: Maximum samples per dataset to process (0 = all).
        config_path: Optional path to JSON config.
        log_interval: Interval count for progress logging to stdout.

    Returns:
        str: Absolute path to the generated extracted_frames_manifest.csv.
    """
    cfg = load_config(config_path)
    manifest_path = cfg.paths.manifest_path

    if not os.path.exists(manifest_path):
        logger.error(f"Unified manifest not found at: {manifest_path}. Run build_manifest first.")
        return ""

    mode_enum = SamplingMode(sampling_mode.lower())

    out_manifest_dir = cfg.paths.manifest_dir
    out_manifest_path = os.path.join(out_manifest_dir, "extracted_frames_manifest.csv")
    os.makedirs(out_manifest_dir, exist_ok=True)

    ouc_extractor = OUCCGEFrameExtractor()
    coler_extractor = COLERFrameExtractor()

    extracted_records: List[Dict[str, Any]] = []

    logger.info(f"Reading manifest: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        samples = list(reader)

    total_samples = len(samples)
    logger.info(f"Total samples in manifest: {total_samples}")
    logger.info(f"Configured OUC-CGE Sampling: num_frames={num_frames}, mode={mode_enum.value}")

    ouc_count = 0
    coler_count = 0
    start_time = time.time()

    fieldnames = [
        "sample_id",
        "dataset_name",
        "status",
        "total_frames",
        "extracted_count",
        "sampling_mode",
        "frame_indices",
        "frame_paths",
        "missing_frames_count",
        "label_3class",
        "split_fold",
    ]

    for idx, sample in enumerate(samples, 1):
        dataset_name = sample["dataset_name"]
        sample_id = sample["sample_id"]

        if dataset_name == "OUC-CGE":
            if max_samples > 0 and ouc_count >= max_samples:
                continue

            video_path = sample["abs_path"] if "abs_path" in sample and os.path.exists(sample.get("abs_path", "")) else os.path.join(cfg.paths.ouc_cge_root, sample["relative_path"])

            meta = ouc_extractor.extract_from_video(
                video_abs_path=video_path,
                sample_id=sample_id,
                num_frames=num_frames,
                sampling_mode=mode_enum,
                save_to_disk=True,
            )

            record = {
                "sample_id": sample_id,
                "dataset_name": "OUC-CGE",
                "status": meta["status"],
                "total_frames": meta["total_frames"],
                "extracted_count": meta.get("num_extracted", 0),
                "sampling_mode": meta["sampling_mode"],
                "frame_indices": json.dumps(meta["indices_extracted"]),
                "frame_paths": json.dumps(meta["extracted_frame_paths"]),
                "missing_frames_count": 0,
                "label_3class": sample["label_3class"],
                "split_fold": sample["split_fold"],
            }
            extracted_records.append(record)
            ouc_count += 1

        elif dataset_name == "COLER":
            if max_samples > 0 and coler_count >= max_samples:
                continue

            sample_dir = sample["abs_path"] if "abs_path" in sample and os.path.exists(sample.get("abs_path", "")) else os.path.join(cfg.paths.coler_root, sample["relative_path"])

            meta = coler_extractor.validate_and_normalize_sequence(
                sample_dir_abs_path=sample_dir,
                sample_id=sample_id,
                expected_total_frames=15,
            )

            record = {
                "sample_id": sample_id,
                "dataset_name": "COLER",
                "status": meta["status"],
                "total_frames": meta["raw_file_count"],
                "extracted_count": len(meta["ordered_frame_paths"]),
                "sampling_mode": "coler_sequence",
                "frame_indices": json.dumps(meta["sorted_frame_indices"]),
                "frame_paths": json.dumps(meta["ordered_frame_paths"]),
                "missing_frames_count": len(meta["missing_frames"]),
                "label_3class": sample["label_3class"],
                "split_fold": sample["split_fold"],
            }
            extracted_records.append(record)
            coler_count += 1

        # Live Terminal Progress Logging
        if idx % log_interval == 0 or idx == total_samples:
            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 else 0
            logger.info(
                f"Progress: [{idx:04d}/{total_samples:04d}] ({idx/total_samples*100:.1f}%) | "
                f"OUC-CGE: {ouc_count} | COLER: {coler_count} | Speed: {rate:.1f} samples/sec"
            )

            # Periodically write manifest checkpoint
            with open(out_manifest_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(extracted_records)

    # Final Manifest Write
    with open(out_manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(extracted_records)

    elapsed_total = time.time() - start_time
    logger.info("==================================================")
    logger.info(" FRAME EXTRACTION PIPELINE COMPLETE!             ")
    logger.info(f" Total Processed: OUC-CGE = {ouc_count}, COLER = {coler_count}")
    logger.info(f" Total Time Elapsed: {elapsed_total:.2f} seconds")
    logger.info(f" Manifest Saved To: {out_manifest_path}")
    logger.info("==================================================")

    return out_manifest_path


if __name__ == "__main__":
    run_extraction_pipeline(num_frames=3, sampling_mode="uniform", max_samples=0)
