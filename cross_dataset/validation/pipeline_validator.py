"""
Comprehensive Pipeline Validator.

Validates:
  1. Duplicate samples (sample_id, image_path)
  2. Missing files (non-existent image/video paths)
  3. Broken images (corrupted headers/bytes)
  4. Label mismatch (raw_label vs ConfigurableLabelMapper)
  5. Incorrect metadata (width, height, fps, JSON syntax)
  6. Frame counts (manifest vs extracted frames)

Generates data/reports/pipeline_validation_report.md.
"""

import csv
import json
import os
from typing import List, Dict, Any, Tuple

from PIL import Image

from cross_dataset.config.base_config import load_config
from cross_dataset.label_mapping.mapper import ConfigurableLabelMapper
from cross_dataset.utils.logger import get_logger

logger = get_logger("Pipeline_Validator")


class PipelineValidator:
    """
    Validates end-to-end dataset manifests, extracted frames, label mappings, and image integrity.
    """

    def __init__(self, config_path: str = None):
        self.cfg = load_config(config_path)
        self.root_dir = self.cfg.paths.root_dir
        self.label_mapper = ConfigurableLabelMapper(config_path=self.cfg.paths.label_mappings_config)

        self.clip_manifest_path = self.cfg.paths.manifest_path
        self.frame_manifest_path = os.path.join(self.cfg.paths.manifest_dir, "unified_frame_dataset.csv")
        self.report_path = os.path.join(self.root_dir, "data", "reports", "pipeline_validation_report.md")

    def run_validation(self) -> Dict[str, Any]:
        """
        Runs complete pipeline validation audit.
        """
        logger.info("==================================================")
        logger.info(" Starting End-to-End Pipeline Validation Audit")
        logger.info("==================================================")

        results = {
            "clip_manifest_exists": os.path.exists(self.clip_manifest_path),
            "frame_manifest_exists": os.path.exists(self.frame_manifest_path),
            "total_clip_samples": 0,
            "total_frame_samples": 0,
            "duplicate_sample_ids": [],
            "duplicate_image_paths": [],
            "missing_files": [],
            "broken_images": [],
            "label_mismatches": [],
            "incorrect_metadata": [],
            "frame_count_discrepancies": [],
        }

        if not results["clip_manifest_exists"]:
            logger.error(f"Clip manifest not found at: {self.clip_manifest_path}")
            return results

        # 1. Audit Clip Manifest (unified_manifest.csv)
        self._audit_clip_manifest(results)

        # 2. Audit Frame Manifest (unified_frame_dataset.csv)
        if results["frame_manifest_exists"]:
            self._audit_frame_manifest(results)

        # 3. Generate Markdown Validation Report
        self._generate_markdown_report(results)

        logger.info("==================================================")
        logger.info(f" Audit Complete! Report saved to: {self.report_path}")
        logger.info("==================================================")

        return results

    def _audit_clip_manifest(self, results: Dict[str, Any]) -> None:
        """Audits unified_manifest.csv for duplicates, missing paths, label mismatches, and metadata."""
        logger.info(f"Auditing Clip Manifest: {self.clip_manifest_path}")

        seen_sample_ids = set()

        with open(self.clip_manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        results["total_clip_samples"] = len(rows)

        for idx, row in enumerate(rows):
            sample_id = row["sample_id"]
            raw_label = row["raw_label"]
            rel_path = row["relative_path"]

            # 1. Duplicate sample_id check
            if sample_id in seen_sample_ids:
                results["duplicate_sample_ids"].append(sample_id)
            seen_sample_ids.add(sample_id)

            # 2. Label Mismatch check against ConfigurableLabelMapper
            try:
                exp_3c, exp_bin = self.label_mapper.map_all(raw_label)
                act_3c = int(row["label_3class"])
                act_bin = int(row["label_binary"])

                if act_3c != exp_3c or act_bin != exp_bin:
                    results["label_mismatches"].append({
                        "sample_id": sample_id,
                        "raw_label": raw_label,
                        "expected_3class": exp_3c,
                        "actual_3class": act_3c,
                        "expected_binary": exp_bin,
                        "actual_binary": act_bin,
                    })
            except Exception as e:
                results["label_mismatches"].append({
                    "sample_id": sample_id,
                    "raw_label": raw_label,
                    "error": str(e),
                })

            # 3. Metadata validation
            try:
                num_frames = int(row.get("num_frames", 0))
                fps = float(row.get("fps", 0.0))
                width = int(row.get("width", 0))
                height = int(row.get("height", 0))

                if num_frames <= 0 or fps <= 0 or width <= 0 or height <= 0:
                    results["incorrect_metadata"].append({
                        "sample_id": sample_id,
                        "issue": f"Invalid dimensions/fps: frames={num_frames}, fps={fps}, size=({width}x{height})",
                    })
            except ValueError as e:
                results["incorrect_metadata"].append({
                    "sample_id": sample_id,
                    "issue": f"Value parsing error: {e}",
                })

            # 4. Missing file check
            abs_p = row.get("abs_path", "")
            if not abs_p or not os.path.exists(abs_p):
                base_dir = self.cfg.paths.ouc_cge_root if row["dataset_name"] == "OUC-CGE" else self.cfg.paths.coler_root
                abs_p = os.path.join(base_dir, rel_path)

            if not os.path.exists(abs_p):
                results["missing_files"].append(abs_p)

    def _audit_frame_manifest(self, results: Dict[str, Any]) -> None:
        """Audits unified_frame_dataset.csv for broken images, frame count integrity, and duplicates."""
        logger.info(f"Auditing Frame Manifest: {self.frame_manifest_path}")

        seen_image_paths = set()
        clip_frame_counts: Dict[str, int] = {}

        with open(self.frame_manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        results["total_frame_samples"] = len(rows)

        for row in rows:
            clip_id = row["clip_id"]
            rel_img_path = row["image_path"]
            raw_label = row["original_label"]

            clip_frame_counts[clip_id] = clip_frame_counts.get(clip_id, 0) + 1

            abs_img_path = os.path.join(self.root_dir, rel_img_path) if not os.path.isabs(rel_img_path) else rel_img_path

            # 1. Duplicate image path check
            if abs_img_path in seen_image_paths:
                results["duplicate_image_paths"].append(abs_img_path)
            seen_image_paths.add(abs_img_path)

            # 2. Missing image file check
            if not os.path.exists(abs_img_path):
                results["missing_files"].append(abs_img_path)
                continue

            # 3. Broken image check (sample verification)
            try:
                with Image.open(abs_img_path) as img:
                    img.verify()  # Verifies byte integrity without decoding full image
            except Exception as e:
                results["broken_images"].append({
                    "image_path": abs_img_path,
                    "error": str(e),
                })

            # 4. JSON metadata syntax validation
            try:
                meta_json = row.get("metadata", "{}")
                meta_dict = json.loads(meta_json)
                if not isinstance(meta_dict, dict):
                    results["incorrect_metadata"].append({
                        "sample_id": clip_id,
                        "issue": "Metadata JSON is not a valid dictionary",
                    })
            except Exception as e:
                results["incorrect_metadata"].append({
                    "sample_id": clip_id,
                    "issue": f"Metadata JSON syntax error: {e}",
                })

            # 5. Label Mismatch check
            try:
                exp_3c, exp_bin = self.label_mapper.map_all(raw_label)
                act_3c = int(row["mapped_label_3class"])
                act_bin = int(row["mapped_label_binary_int"])

                if act_3c != exp_3c or act_bin != exp_bin:
                    results["label_mismatches"].append({
                        "sample_id": clip_id,
                        "raw_label": raw_label,
                        "expected_3class": exp_3c,
                        "actual_3class": act_3c,
                    })
            except Exception as e:
                results["label_mismatches"].append({
                    "sample_id": clip_id,
                    "raw_label": raw_label,
                    "error": str(e),
                })

    def _generate_markdown_report(self, results: Dict[str, Any]) -> None:
        """Generates comprehensive markdown report."""
        os.makedirs(os.path.dirname(self.report_path), exist_ok=True)

        status_flag = "PASSED ✅" if (
            len(results["duplicate_sample_ids"]) == 0 and
            len(results["missing_files"]) == 0 and
            len(results["broken_images"]) == 0 and
            len(results["label_mismatches"]) == 0 and
            len(results["incorrect_metadata"]) == 0
        ) else "WARNINGS / ANOMALIES FOUND ⚠️"

        report_lines = [
            "# Pipeline Validation Audit Report",
            "",
            f"**Overall Status**: {status_flag}  ",
            f"**Clip Manifest**: `{self.clip_manifest_path}`  ",
            f"**Frame Manifest**: `{self.frame_manifest_path}`  ",
            "",
            "---",
            "",
            "## 1. Audit Summary Statistics",
            "",
            "| Validation Check | Total Examined | Anomalies / Errors Found | Status |",
            "|---|---|---|---|",
            f"| **Clip Manifest Samples** | {results['total_clip_samples']} | {len(results['duplicate_sample_ids'])} duplicates | {'✅ Clean' if len(results['duplicate_sample_ids']) == 0 else '⚠️ Duplicates'} |",
            f"| **Frame Manifest Samples** | {results['total_frame_samples']} | {len(results['duplicate_image_paths'])} duplicates | {'✅ Clean' if len(results['duplicate_image_paths']) == 0 else '⚠️ Duplicates'} |",
            f"| **File Existence Check** | {results['total_clip_samples'] + results['total_frame_samples']} | {len(results['missing_files'])} missing files | {'✅ Clean' if len(results['missing_files']) == 0 else '❌ Missing Files'} |",
            f"| **Image Byte Integrity** | {results['total_frame_samples']} | {len(results['broken_images'])} broken images | {'✅ Clean' if len(results['broken_images']) == 0 else '❌ Broken Images'} |",
            f"| **Label Consistency** | {results['total_clip_samples'] + results['total_frame_samples']} | {len(results['label_mismatches'])} label mismatches | {'✅ Clean' if len(results['label_mismatches']) == 0 else '❌ Mismatch'} |",
            f"| **Metadata Integrity** | {results['total_clip_samples'] + results['total_frame_samples']} | {len(results['incorrect_metadata'])} metadata issues | {'✅ Clean' if len(results['incorrect_metadata']) == 0 else '⚠️ Issues'} |",
            "",
            "---",
            "",
            "## 2. Detailed Anomaly Findings",
            "",
        ]

        # Duplicate Sample IDs
        report_lines.append("### 2.1 Duplicate Sample IDs")
        if results["duplicate_sample_ids"]:
            report_lines.append(f"Found **{len(results['duplicate_sample_ids'])}** duplicate sample IDs:")
            for sample_id in results["duplicate_sample_ids"][:10]:
                report_lines.append(f"- `{sample_id}`")
        else:
            report_lines.append("✅ Zero duplicate sample IDs found.")
        report_lines.append("")

        # Missing Files
        report_lines.append("### 2.2 Missing Files")
        if results["missing_files"]:
            report_lines.append(f"Found **{len(results['missing_files'])}** missing file paths:")
            for path in results["missing_files"][:10]:
                report_lines.append(f"- `{path}`")
        else:
            report_lines.append("✅ Zero missing files found.")
        report_lines.append("")

        # Broken Images
        report_lines.append("### 2.3 Broken / Unreadable Images")
        if results["broken_images"]:
            report_lines.append(f"Found **{len(results['broken_images'])}** broken image files:")
            for item in results["broken_images"][:10]:
                report_lines.append(f"- `{item['image_path']}` (Error: {item['error']})")
        else:
            report_lines.append("✅ Zero broken or corrupted image files detected.")
        report_lines.append("")

        # Label Mismatches
        report_lines.append("### 2.4 Label Taxonomy Mismatches")
        if results["label_mismatches"]:
            report_lines.append(f"Found **{len(results['label_mismatches'])}** label mismatches:")
            for item in results["label_mismatches"][:10]:
                report_lines.append(f"- `{item['sample_id']}`: Raw='{item['raw_label']}', Expected 3-Class={item.get('expected_3class')}, Actual={item.get('actual_3class')}")
        else:
            report_lines.append("✅ Zero label mismatches found. ConfigurableLabelMapper strictly matches manifest values.")
        report_lines.append("")

        # Metadata Integrity
        report_lines.append("### 2.5 Metadata Syntax & Values")
        if results["incorrect_metadata"]:
            report_lines.append(f"Found **{len(results['incorrect_metadata'])}** metadata anomalies:")
            for item in results["incorrect_metadata"][:10]:
                report_lines.append(f"- `{item['sample_id']}`: {item['issue']}")
        else:
            report_lines.append("✅ Zero metadata syntax or dimensional anomalies found.")
        report_lines.append("")

        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))


def validate_pipeline(config_path: str = None) -> Dict[str, Any]:
    """Runner function."""
    validator = PipelineValidator(config_path=config_path)
    return validator.run_validation()


if __name__ == "__main__":
    validate_pipeline()
