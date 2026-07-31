"""
Sampling Strategies for Video Frame Extraction.
"""

import random
from enum import Enum
from typing import List


class SamplingMode(Enum):
    SINGLE = "single"       # 1 frame (center frame)
    THREE = "three"         # 3 frames (start, middle, end)
    UNIFORM = "uniform"     # N frames uniformly spaced
    CENTER = "center"       # N frames centered around middle
    RANDOM = "random"       # N frames randomly selected (sorted)


class FrameSampler:
    """
    Computes frame indices to extract based on configurable sampling strategies.
    """

    @staticmethod
    def get_sample_indices(
        total_frames: int,
        num_frames: int = 1,
        mode: SamplingMode = SamplingMode.UNIFORM,
        seed: int = 42,
    ) -> List[int]:
        """
        Calculates 0-indexed frame positions to extract from a video sequence.

        Args:
            total_frames: Total number of frames in the video clip.
            num_frames: Desired number of frames N to sample.
            mode: Sampling strategy (SamplingMode enum or string name).
            seed: Random seed for random sampling mode.

        Returns:
            List[int]: Sorted list of 0-based frame indices.
        """
        if total_frames <= 0:
            return []

        if isinstance(mode, str):
            mode = SamplingMode(mode.lower())

        if total_frames <= num_frames:
            # If total frames is less than or equal to requested N, return all available indices
            return list(range(total_frames))

        if mode == SamplingMode.SINGLE:
            # 1 frame: middle frame
            return [total_frames // 2]

        elif mode == SamplingMode.THREE:
            # 3 frames: start, middle, end
            if total_frames < 3:
                return list(range(total_frames))
            mid = total_frames // 2
            return [0, mid, total_frames - 1]

        elif mode == SamplingMode.UNIFORM:
            # N frames uniformly spaced
            if num_frames <= 1:
                return [total_frames // 2]
            step = (total_frames - 1) / (num_frames - 1)
            indices = [int(round(i * step)) for i in range(num_frames)]
            # Ensure unique and sorted
            indices = sorted(list(set(indices)))
            return indices

        elif mode == SamplingMode.CENTER:
            # N frames centered around middle frame
            mid = total_frames // 2
            half = num_frames // 2
            start = max(0, mid - half)
            end = min(total_frames, start + num_frames)
            if (end - start) < num_frames:
                start = max(0, end - num_frames)
            return list(range(start, end))

        elif mode == SamplingMode.RANDOM:
            # N frames randomly selected without replacement, sorted chronologically
            rng = random.Random(seed)
            indices = rng.sample(range(total_frames), k=min(num_frames, total_frames))
            return sorted(indices)

        else:
            raise ValueError(f"Unsupported sampling mode: {mode}")
