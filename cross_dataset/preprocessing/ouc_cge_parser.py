"""
Dataset Scanner and Metadata Extractor for OUC-CGE Dataset.
"""

import os
from typing import List, Dict, Any, Tuple

from cross_dataset.preprocessing.label_mapper import LabelMapper
from cross_dataset.utils.validation import validate_video_file
from cross_dataset.utils.logger import get_logger

logger = get_logger("OUC_CGE_Parser")


class OUCCGEParser:
    """
    Scans OUC-CGE video dataset, validates MP4 files, extracts metadata,
    and formats records for the unified manifest.
    """

    CLASS_FOLDERS = ["high", "mid", "low"]

    def __init__(self, dataset_root: str, validate_videos: bool = True):
        """
        Args:
            dataset_root: Path to OUC-CGE-dataset directory.
            validate_videos: Whether to check video decoding integrity with OpenCV.
        """
        self.dataset_root = os.path.abspath(dataset_root)
        self.validate_videos = validate_videos

    def scan(self) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Executes dataset scan and metadata extraction.

        Returns:
            Tuple of (valid_records, corrupted_file_paths).
        """
        if not os.path.exists(self.dataset_root):
            logger.error(f"OUC-CGE dataset root directory not found: {self.dataset_root}")
            return [], []

        valid_records: List[Dict[str, Any]] = []
        corrupted_files: List[str] = []

        logger.info(f"Scanning OUC-CGE dataset at: {self.dataset_root}")

        sample_counter = 0
        for class_folder in self.CLASS_FOLDERS:
            class_path = os.path.join(self.dataset_root, class_folder)
            if not os.path.exists(class_path):
                logger.warning(f"Class folder missing: {class_path}")
                continue

            for root, _, files in os.walk(class_path):
                for file_name in files:
                    if not file_name.lower().endswith((".mp4", ".avi", ".mov")):
                        continue

                    abs_path = os.path.join(root, file_name)
                    rel_path = os.path.relpath(abs_path, self.dataset_root).replace("\\", "/")

                    # Map raw label to standardized targets
                    label_3c, label_bin = LabelMapper.get_labels(class_folder)

                    # Metadata defaults
                    metadata = {
                        "frame_count": 300,
                        "fps": 30.0,
                        "width": 1092,
                        "height": 614,
                        "duration_sec": 10.0,
                    }

                    is_valid = True
                    err_msg = None

                    if self.validate_videos:
                        is_valid, meta, err_msg = validate_video_file(abs_path)
                        if is_valid:
                            metadata.update(meta)
                        else:
                            corrupted_files.append(abs_path)
                            logger.debug(f"Corrupted video skipped [{err_msg}]: {rel_path}")
                            continue

                    sample_counter += 1
                    sample_id = f"ouc_{class_folder}_{sample_counter:05d}"

                    record = {
                        "sample_id": sample_id,
                        "dataset_name": "OUC-CGE",
                        "relative_path": rel_path,
                        "abs_path": abs_path,
                        "file_type": "mp4",
                        "num_frames": metadata["frame_count"],
                        "fps": metadata["fps"],
                        "width": metadata["width"],
                        "height": metadata["height"],
                        "duration_sec": metadata["duration_sec"],
                        "raw_label": class_folder.capitalize(),
                        "label_3class": label_3c,
                        "label_binary": label_bin,
                        "is_valid": True,
                    }

                    valid_records.append(record)

        logger.info(
            f"OUC-CGE scan complete: {len(valid_records)} valid samples, "
            f"{len(corrupted_files)} corrupted files excluded."
        )
        return valid_records, corrupted_files
