"""
Frame Extractor and Sequence Validator for COLER Dataset.
"""

import os
import re
from typing import List, Dict, Any, Tuple, Optional

from cross_dataset.utils.logger import get_logger

logger = get_logger("COLER_Frame_Extractor")


class COLERFrameExtractor:
    """
    Validates, sorts, normalizes, and verifies missing frames for COLER PNG frame sequences.
    """

    # Regex patterns for frame index extraction
    FRAME_PATTERNS = [
        re.compile(r"frame_?(\d+)\.png$", re.IGNORECASE),
        re.compile(r"_frame(\d+)\.png$", re.IGNORECASE),
        re.compile(r"frame(\d+)\.png$", re.IGNORECASE),
    ]

    def __init__(self, output_base_dir: Optional[str] = None):
        """
        Args:
            output_base_dir: Directory where normalized frames can optionally be copied/linked.
        """
        if output_base_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            output_base_dir = os.path.join(project_root, "data", "extracted_frames", "COLER")

        self.output_base_dir = os.path.abspath(output_base_dir)

    @classmethod
    def parse_frame_index(cls, filename: str) -> Optional[int]:
        """
        Extracts integer frame index from filename using regex patterns.

        Args:
            filename: Image filename (e.g., 'Video1113_rgb_3_frame107.png' or 'frame_005.png').

        Returns:
            Optional[int]: Extracted 1-based or 0-based frame index, or None if unparseable.
        """
        for pattern in cls.FRAME_PATTERNS:
            match = pattern.search(filename)
            if match:
                return int(match.group(1))
        return None

    def validate_and_normalize_sequence(
        self,
        sample_dir_abs_path: str,
        sample_id: str,
        expected_total_frames: int = 15,
    ) -> Dict[str, Any]:
        """
        Validates PNG sequence ordering, extracts indices, detects missing frames,
        and generates normalized file metadata.

        Args:
            sample_dir_abs_path: Path to COLER student sample directory (containing PNGs).
            sample_id: Unique sample identifier.
            expected_total_frames: Expected number of sampled frames in a full sequence (default: 15).

        Returns:
            Dict[str, Any]: Detailed sequence metadata dictionary.
        """
        result_meta = {
            "sample_id": sample_id,
            "sample_dir": sample_dir_abs_path,
            "raw_file_count": 0,
            "sorted_frame_indices": [],
            "ordered_frame_paths": [],
            "normalized_filenames": [],
            "missing_frames": [],
            "is_sequential": False,
            "status": "FAILED",
            "error_msg": None,
        }

        if not os.path.exists(sample_dir_abs_path):
            result_meta["error_msg"] = f"Directory not found: {sample_dir_abs_path}"
            return result_meta

        try:
            # Gather all PNG files in directory
            all_files = os.listdir(sample_dir_abs_path)
            png_files = [f for f in all_files if f.lower().endswith(".png")]
            result_meta["raw_file_count"] = len(png_files)

            if not png_files:
                result_meta["error_msg"] = "No PNG frame files found in directory"
                return result_meta

            # Extract indices and pair with original file paths
            indexed_frames: List[Tuple[int, str]] = []
            unparsed_files: List[str] = []

            for fname in png_files:
                idx = self.parse_frame_index(fname)
                if idx is not None:
                    indexed_frames.append((idx, os.path.join(sample_dir_abs_path, fname)))
                else:
                    unparsed_files.append(fname)

            if not indexed_frames:
                result_meta["error_msg"] = "Could not parse frame indices from filenames"
                return result_meta

            # Sort by frame index to validate order
            indexed_frames.sort(key=lambda x: x[0])

            sorted_indices = [item[0] for item in indexed_frames]
            ordered_paths = [item[1] for item in indexed_frames]

            # Normalize filenames format
            normalized_names = [f"frame_{idx:04d}.png" for idx in range(1, len(ordered_paths) + 1)]

            # Detect missing frames / sequence gaps
            missing_frames = []
            if len(sorted_indices) > 1:
                # Check for step gaps in sorted indices
                first_idx = sorted_indices[0]
                last_idx = sorted_indices[-1]

                # Check if steps between frames are uniform
                for i in range(len(sorted_indices) - 1):
                    gap = sorted_indices[i + 1] - sorted_indices[i]
                    if gap > 1:
                        missing_frames.extend(list(range(sorted_indices[i] + 1, sorted_indices[i + 1])))

            # Check sequence continuity
            is_seq = (len(missing_frames) == 0) and (len(png_files) == expected_total_frames)

            result_meta["sorted_frame_indices"] = sorted_indices
            result_meta["ordered_frame_paths"] = ordered_paths
            result_meta["normalized_filenames"] = normalized_names
            result_meta["missing_frames"] = missing_frames
            result_meta["is_sequential"] = is_seq
            result_meta["status"] = "SUCCESS"

            return result_meta

        except Exception as e:
            result_meta["error_msg"] = f"Exception during COLER sequence validation: {str(e)}"
            return result_meta
