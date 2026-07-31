"""
Dataset Scanner and Metadata Extractor for COLER Dataset.
"""

import os
from typing import List, Dict, Any, Tuple

from cross_dataset.preprocessing.label_mapper import LabelMapper
from cross_dataset.utils.logger import get_logger

logger = get_logger("COLER_Parser")


class COLERParser:
    """
    Scans COLER dataset (Ver4 directory tree), resolves directory anomalies,
    extracts per-sample frame metadata, pose/skeleton presence, and formats records.
    """

    CLASS_NAMES = ["Disengagement", "Normal", "Engagement", "Very engagement", "Very Engagement"]

    def __init__(self, dataset_root: str):
        """
        Args:
            dataset_root: Path to COLER Ver4 directory (e.g. .../Ver4).
        """
        self.dataset_root = os.path.abspath(dataset_root)

    def scan(self) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Executes directory tree scan across Goc/Table hierarchies.

        Returns:
            Tuple of (valid_records, empty_or_invalid_directories).
        """
        if not os.path.exists(self.dataset_root):
            logger.error(f"COLER dataset root directory not found: {self.dataset_root}")
            return [], []

        valid_records: List[Dict[str, Any]] = []
        invalid_dirs: List[str] = []

        logger.info(f"Scanning COLER dataset at: {self.dataset_root}")

        sample_counter = 0

        # Traverse Goc1 to Goc5
        for goc_dir in sorted(os.listdir(self.dataset_root)):
            if not goc_dir.startswith("Goc"):
                continue
            goc_path = os.path.join(self.dataset_root, goc_dir)
            if not os.path.isdir(goc_path):
                continue

            # Traverse Table1 to Table3
            for table_dir in sorted(os.listdir(goc_path)):
                if not table_dir.startswith("Table"):
                    continue
                table_path = os.path.join(goc_path, table_dir)
                if not os.path.isdir(table_path):
                    continue

                # Traverse Class Folders
                for class_dir in os.listdir(table_path):
                    if class_dir not in self.CLASS_NAMES:
                        continue
                    class_path = os.path.join(table_path, class_dir)
                    if not os.path.isdir(class_path):
                        continue

                    # Determine class label and normalized key
                    raw_label = class_dir
                    label_3c, label_bin = LabelMapper.get_labels(raw_label)

                    # Traverse X0_GocN, X1_GocN, etc.
                    for x_dir in os.listdir(class_path):
                        if x_dir == "Emotion":
                            continue  # Emotion branch handled separately if needed
                        x_path = os.path.join(class_path, x_dir)
                        if not os.path.isdir(x_path):
                            continue

                        # Traverse Video{ID} folders
                        for video_dir in os.listdir(x_path):
                            if not video_dir.startswith("Video"):
                                continue
                            video_path = os.path.join(x_path, video_dir)
                            if not os.path.isdir(video_path):
                                continue

                            # Traverse id_{N} folders
                            for id_dir in os.listdir(video_path):
                                if not id_dir.startswith("id_"):
                                    continue
                                id_path = os.path.join(video_path, id_dir)
                                if not os.path.isdir(id_path):
                                    continue

                                # Count RGB frames and skeleton CSV presence
                                files = os.listdir(id_path)
                                rgb_frames = [f for f in files if f.lower().endswith(".png")]
                                skeleton_csv = "skeletons.csv" in files

                                if len(files) == 0:
                                    invalid_dirs.append(id_path)
                                    continue

                                sample_counter += 1
                                rel_path = os.path.relpath(id_path, self.dataset_root).replace("\\", "/")
                                sample_id = f"coler_{goc_dir.lower()}_{table_dir.lower()}_{video_dir.lower()}_{id_dir.lower()}"

                                record = {
                                    "sample_id": sample_id,
                                    "dataset_name": "COLER",
                                    "relative_path": rel_path,
                                    "abs_path": id_path,
                                    "file_type": "png_sequence",
                                    "num_frames": len(rgb_frames) if rgb_frames else 1,
                                    "fps": 3.0,  # 15 frames per 5s clip -> ~3fps sampled
                                    "width": 224,  # Standard crop size
                                    "height": 224,
                                    "duration_sec": 5.0,
                                    "raw_label": raw_label,
                                    "label_3class": label_3c,
                                    "label_binary": label_bin,
                                    "has_skeleton": skeleton_csv,
                                    "is_valid": True,
                                }

                                valid_records.append(record)

        logger.info(
            f"COLER scan complete: {len(valid_records)} valid student samples, "
            f"{len(invalid_dirs)} empty directories flagged."
        )
        return valid_records, invalid_dirs
