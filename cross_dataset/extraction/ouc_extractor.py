"""
Frame Extractor for OUC-CGE Video Dataset.
"""

import os
from typing import List, Dict, Any, Optional

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

from cross_dataset.extraction.samplers import FrameSampler, SamplingMode
from cross_dataset.utils.logger import get_logger

logger = get_logger("OUC_Frame_Extractor")


class OUCCGEFrameExtractor:
    """
    Extracts frames from OUC-CGE MP4 video clips according to configurable sampling strategies.
    """

    def __init__(self, output_base_dir: Optional[str] = None):
        """
        Args:
            output_base_dir: Directory where extracted frame PNGs will be saved.
                             If None, defaults to 'data/extracted_frames/OUC-CGE'.
        """
        if output_base_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            output_base_dir = os.path.join(project_root, "data", "extracted_frames", "OUC-CGE")

        self.output_base_dir = os.path.abspath(output_base_dir)

    def extract_from_video(
        self,
        video_abs_path: str,
        sample_id: str,
        num_frames: int = 3,
        sampling_mode: SamplingMode = SamplingMode.UNIFORM,
        save_to_disk: bool = True,
    ) -> Dict[str, Any]:
        """
        Extracts requested frames from an OUC-CGE video file.

        Args:
            video_abs_path: Absolute path to the MP4 file.
            sample_id: Unique sample identifier.
            num_frames: Number of frames N to sample.
            sampling_mode: Sampling strategy (SINGLE, THREE, UNIFORM, CENTER, RANDOM).
            save_to_disk: If True, writes extracted frames to PNG files.

        Returns:
            Dict[str, Any]: Extraction metadata dictionary.
        """
        result_meta = {
            "sample_id": sample_id,
            "video_path": video_abs_path,
            "sampling_mode": str(sampling_mode.value if isinstance(sampling_mode, SamplingMode) else sampling_mode),
            "num_requested": num_frames,
            "total_frames": 0,
            "indices_extracted": [],
            "extracted_frame_paths": [],
            "status": "FAILED",
            "error_msg": None,
        }

        if not os.path.exists(video_abs_path):
            result_meta["error_msg"] = f"File not found: {video_abs_path}"
            return result_meta

        if not OPENCV_AVAILABLE:
            result_meta["error_msg"] = "OpenCV (cv2) is not installed"
            return result_meta

        try:
            cap = cv2.VideoCapture(video_abs_path)
            if not cap.isOpened():
                result_meta["error_msg"] = "Failed to open video file stream"
                return result_meta

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            result_meta["total_frames"] = total_frames

            if total_frames <= 0:
                cap.release()
                result_meta["error_msg"] = "Video has 0 frames"
                return result_meta

            # Calculate target frame indices using sampler
            target_indices = FrameSampler.get_sample_indices(
                total_frames=total_frames,
                num_frames=num_frames,
                mode=sampling_mode,
            )

            result_meta["indices_extracted"] = target_indices

            sample_out_dir = os.path.join(self.output_base_dir, sample_id)
            if save_to_disk:
                os.makedirs(sample_out_dir, exist_ok=True)

            extracted_paths = []
            target_set = set(target_indices)

            current_frame_idx = 0
            while cap.isOpened() and len(extracted_paths) < len(target_indices):
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                if current_frame_idx in target_set:
                    out_name = f"frame_{current_frame_idx:04d}.png"
                    out_path = os.path.join(sample_out_dir, out_name)

                    if save_to_disk:
                        cv2.imwrite(out_path, frame)

                    extracted_paths.append(out_path)

                current_frame_idx += 1

            cap.release()

            result_meta["extracted_frame_paths"] = extracted_paths
            result_meta["num_extracted"] = len(extracted_paths)
            result_meta["status"] = "SUCCESS"

            return result_meta

        except Exception as e:
            result_meta["error_msg"] = f"Exception during frame extraction: {str(e)}"
            return result_meta
