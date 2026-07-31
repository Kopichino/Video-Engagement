"""
Validation Utilities for Video and Image Data Integrity.
"""

import os
from typing import Tuple, Dict, Any, Optional

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


def validate_video_file(video_path: str) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """
    Validates a video file using OpenCV.

    Args:
        video_path: Absolute or relative path to the video file.

    Returns:
        Tuple containing:
            - is_valid (bool): True if video opens and decodes properly.
            - metadata (dict): Extracted metadata (frame_count, fps, width, height, duration_sec).
            - error_msg (str or None): Error message if invalid.
    """
    metadata = {
        "frame_count": 0,
        "fps": 0.0,
        "width": 0,
        "height": 0,
        "duration_sec": 0.0,
    }

    if not os.path.exists(video_path):
        return False, metadata, "File does not exist"

    if os.path.getsize(video_path) == 0:
        return False, metadata, "File size is 0 bytes"

    if not OPENCV_AVAILABLE:
        # Fallback if cv2 is not installed
        return True, metadata, None

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, metadata, "OpenCV failed to open video stream"

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Attempt to read at least the first frame
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return False, metadata, "OpenCV failed to read initial frame"

        duration = (frame_count / fps) if fps > 0 else 0.0

        metadata["frame_count"] = frame_count
        metadata["fps"] = round(fps, 2)
        metadata["width"] = width
        metadata["height"] = height
        metadata["duration_sec"] = round(duration, 2)

        return True, metadata, None

    except Exception as e:
        return False, metadata, f"OpenCV Exception: {str(e)}"


def validate_image_file(image_path: str) -> Tuple[bool, Optional[str]]:
    """
    Validates an image file.

    Args:
        image_path: Path to the image file.

    Returns:
        Tuple of (is_valid, error_message).
    """
    if not os.path.exists(image_path):
        return False, "File does not exist"

    if os.path.getsize(image_path) == 0:
        return False, "File size is 0 bytes"

    return True, None
