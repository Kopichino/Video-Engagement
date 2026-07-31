"""
Unified Dataset Representation Builder.

Generates granular frame-level dataset representations for OUC-CGE and COLER
containing dataset, image_path, clip_id, frame_id, mapped_label, original_label,
and serialized metadata. Outputs unified CSV manifests.
"""

import csv
import json
import os
from typing import List, Dict, Any, Optional

from cross_dataset.config.base_config import load_config
from cross_dataset.label_mapping.mapper import ConfigurableLabelMapper
from cross_dataset.utils.logger import get_logger

logger = get_logger("Unified_Representation_Builder")


class UnifiedDatasetRepresentationBuilder:
    """
    Builds a unified, standardized frame-level dataset representation across OUC-CGE and COLER.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: Path to configuration file.
        """
        self.cfg = load_config(config_path)
        self.label_mapper = ConfigurableLabelMapper(config_path=self.cfg.paths.label_mappings_config)

    def parse_coler_metadata_from_path(self, relative_path: str) -> Dict[str, Any]:
        """
        Extracts structural metadata (goc, table, seat_x, video_id, student_id) from COLER path.
        """
        parts = relative_path.replace("\\", "/").split("/")
        meta = {
            "goc": None,
            "table": None,
            "seat_position": None,
            "video_id": None,
            "student_id": None,
        }

        for part in parts:
            p_lower = part.lower()
            if p_lower.startswith("goc"):
                meta["goc"] = part
            elif p_lower.startswith("table"):
                meta["table"] = part
            elif p_lower.startswith("x") and "_goc" in p_lower:
                meta["seat_position"] = part
            elif p_lower.startswith("video"):
                meta["video_id"] = part
            elif p_lower.startswith("id_"):
                meta["student_id"] = part

        return meta

    def build_representation(self) -> str:
        """
        Processes manifests and frame extractions to generate the unified frame-level CSV.

        Returns:
            str: Path to generated CSV.
        """
        manifest_path = self.cfg.paths.manifest_path
        extracted_manifest_path = os.path.join(self.cfg.paths.manifest_dir, "extracted_frames_manifest.csv")

        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Unified manifest not found at: {manifest_path}. Run build_manifest first.")

        logger.info(f"Building Unified Dataset Representation from manifest: {manifest_path}")

        # Read clip-level manifest
        clip_manifest: Dict[str, Dict[str, Any]] = {}
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                clip_manifest[row["sample_id"]] = row

        # Read extracted frames manifest if available
        extracted_manifest: Dict[str, Dict[str, Any]] = {}
        if os.path.exists(extracted_manifest_path):
            with open(extracted_manifest_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    extracted_manifest[row["sample_id"]] = row

        frame_records: List[Dict[str, Any]] = []

        for sample_id, clip_info in clip_manifest.items():
            dataset_name = clip_info["dataset_name"]
            raw_label = clip_info["raw_label"]
            split_fold = int(clip_info.get("split_fold", 0))

            # Map labels using ConfigurableLabelMapper
            label_3c, label_bin_int = self.label_mapper.map_all(raw_label)
            primary_label_str = self.label_mapper.get_primary_display_name(label_3c)
            binary_label_str = self.label_mapper.get_binary_display_name(label_bin_int)

            rel_path = clip_info["relative_path"]

            # Determine frame image paths
            frame_image_entries: List[Tuple[int, str]] = []

            if sample_id in extracted_manifest:
                ext_info = extracted_manifest[sample_id]
                try:
                    f_paths = json.loads(ext_info.get("frame_paths", "[]"))
                    f_indices = json.loads(ext_info.get("frame_indices", "[]"))

                    if f_paths and len(f_paths) == len(f_indices):
                        frame_image_entries = list(zip(f_indices, f_paths))
                    elif f_paths:
                        frame_image_entries = list(enumerate(f_paths))
                except Exception:
                    pass

            if not frame_image_entries:
                # Fallback: if frame extraction manifest not run for sample, inspect directory or path
                abs_p = clip_info.get("abs_path", "")
                if not abs_p or not os.path.exists(abs_p):
                    base_root = self.cfg.paths.ouc_cge_root if dataset_name == "OUC-CGE" else self.cfg.paths.coler_root
                    abs_p = os.path.join(base_root, rel_path)

                if os.path.isdir(abs_p):
                    files = sorted([f for f in os.listdir(abs_p) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
                    for f_idx, fname in enumerate(files):
                        frame_image_entries.append((f_idx, os.path.join(abs_p, fname)))
                elif os.path.isfile(abs_p) and abs_p.lower().endswith((".png", ".jpg", ".jpeg")):
                    frame_image_entries.append((0, abs_p))

            # Construct Metadata dictionary for sample
            base_meta = {
                "width": int(clip_info.get("width", 224)),
                "height": int(clip_info.get("height", 224)),
                "fps": float(clip_info.get("fps", 30.0)),
                "duration_sec": float(clip_info.get("duration_sec", 0.0)),
                "split_fold": split_fold,
                "file_type": clip_info.get("file_type", "unknown"),
            }

            if dataset_name == "COLER":
                coler_meta = self.parse_coler_metadata_from_path(rel_path)
                base_meta.update(coler_meta)

            # Build row per frame sample
            for frame_idx, img_path in frame_image_entries:
                rel_img_path = os.path.relpath(img_path, self.cfg.paths.root_dir).replace("\\", "/") if os.path.isabs(img_path) else img_path

                frame_record = {
                    "dataset": dataset_name,
                    "image_path": rel_img_path,
                    "clip_id": sample_id,
                    "frame_id": frame_idx,
                    "mapped_label": primary_label_str,
                    "mapped_label_3class": label_3c,
                    "mapped_label_binary": binary_label_str,
                    "mapped_label_binary_int": label_bin_int,
                    "original_label": raw_label,
                    "metadata": json.dumps(base_meta),
                }
                frame_records.append(frame_record)

        # Output CSV Manifest
        output_csv_path = os.path.join(self.cfg.paths.manifest_dir, "unified_frame_dataset.csv")
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

        fieldnames = [
            "dataset",
            "image_path",
            "clip_id",
            "frame_id",
            "mapped_label",
            "mapped_label_3class",
            "mapped_label_binary",
            "mapped_label_binary_int",
            "original_label",
            "metadata",
        ]

        with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(frame_records)

        logger.info(f"Unified Frame-Level Dataset Representation written to: {output_csv_path}")
        logger.info(f"Total Frame-Level Samples Represented: {len(frame_records)}")

        # Also generate per-fold split manifests
        self.generate_split_csvs(frame_records)

        return output_csv_path

    def generate_split_csvs(self, frame_records: List[Dict[str, Any]]) -> None:
        """
        Generates per-fold train and val CSV files under data/manifests/splits/fold_N/.
        """
        splits_base_dir = os.path.join(self.cfg.paths.manifest_dir, "splits")

        fieldnames = [
            "dataset",
            "image_path",
            "clip_id",
            "frame_id",
            "mapped_label",
            "mapped_label_3class",
            "mapped_label_binary",
            "mapped_label_binary_int",
            "original_label",
            "metadata",
        ]

        for fold in range(self.cfg.preprocessing.num_folds):
            fold_dir = os.path.join(splits_base_dir, f"fold_{fold}")
            os.makedirs(fold_dir, exist_ok=True)

            train_rows = []
            val_rows = []

            for rec in frame_records:
                meta = json.loads(rec["metadata"])
                rec_fold = meta.get("split_fold", 0)

                if rec_fold == fold:
                    val_rows.append(rec)
                else:
                    train_rows.append(rec)

            train_csv = os.path.join(fold_dir, "train_frames.csv")
            val_csv = os.path.join(fold_dir, "val_frames.csv")

            with open(train_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(train_rows)

            with open(val_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(val_rows)

        logger.info(f"Per-fold split CSVs successfully generated under: {splits_base_dir}")


def build_unified_dataset_representation(config_path: str = None) -> str:
    """
    Runner entry point.
    """
    builder = UnifiedDatasetRepresentationBuilder(config_path=config_path)
    return builder.build_representation()


if __name__ == "__main__":
    build_unified_dataset_representation()
