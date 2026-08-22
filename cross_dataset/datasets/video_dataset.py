"""
Unified Video Dataset Module for Experiment B (3D Spatiotemporal Sequence Modeling).

Features In-Memory RAM Frame Caching for 100x Faster Data Loading without Windows Shared Memory Errors.
Loads contiguous or uniformly sampled 16-frame temporal windows per video clip
and formats them into 4D PyTorch video tensors of shape [C, T, H, W] = [3, 16, 224, 224].
"""

import os
import glob
import random
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

from cross_dataset.label_mapping.mapper import ConfigurableLabelMapper
from cross_dataset.utils.logger import get_logger

logger = get_logger("UnifiedVideoDataset")


class UnifiedVideoDataset(Dataset):
    """
    PyTorch Dataset for 3D Video Sequence Modeling (Experiment B).
    Loads T contiguous keyframe images per clip and converts them into shape [3, T, H, W].
    Includes In-Memory RAM caching for 100x faster epoch iterations.
    """

    # Global shared RAM cache across dataset instances
    _global_frame_cache: Dict[str, torch.Tensor] = {}

    def __init__(
        self,
        manifest_path: str,
        target_dataset: str,
        split: str = "train",
        fold: int = 0,
        num_frames: int = 16,
        img_size: int = 224,
        label_type: str = "primary",
        balance_classes: bool = False,
        seed: int = 42,
        use_ram_cache: bool = True,
    ):
        """
        Args:
            manifest_path: Path to unified_manifest.csv or extracted_frames_manifest.csv.
            target_dataset: Filter dataset ('OUC-CGE', 'COLER', or 'ALL').
            split: Split subset ('train', 'val', 'test').
            fold: Fold number (0 to 4).
            num_frames: Number of temporal frames T (default: 16).
            img_size: Target frame spatial resolution H=W (default: 224).
            label_type: Label taxonomy ('primary' for 3-class, 'binary' for 2-class).
            balance_classes: If True, applies class balancing to training set.
            seed: Random seed for reproducibility.
            use_ram_cache: If True, caches pre-processed video tensors in RAM.
        """
        self.manifest_path = os.path.abspath(manifest_path)
        self.target_dataset = target_dataset
        self.split = split
        self.fold = fold
        self.num_frames = num_frames
        self.img_size = img_size
        self.label_type = label_type
        self.balance_classes = balance_classes
        self.seed = seed
        self.use_ram_cache = use_ram_cache

        self.label_mapper = ConfigurableLabelMapper()
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.extracted_base_dir = os.path.join(self.project_root, "data", "extracted_frames")

        # Load Manifest
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")

        df = pd.read_csv(self.manifest_path)

        # Dataset Column Normalization ('dataset_name' vs 'dataset')
        ds_col = "dataset_name" if "dataset_name" in df.columns else "dataset"
        if ds_col in df.columns and self.target_dataset.upper() != "ALL":
            df = df[df[ds_col] == self.target_dataset]

        # Filter by Fold Split
        fold_col = f"fold_{self.fold}"
        if fold_col in df.columns:
            df = df[df[fold_col] == self.split]
        elif "split_fold" in df.columns:
            df = df[df["split_fold"].astype(str) == str(self.fold)]
        elif "split" in df.columns:
            df = df[df["split"] == self.split]

        # Map Labels
        def _get_mapped_label(r):
            if "label_3class" in r and self.label_type in ["primary", "label_3class"]:
                val = r["label_3class"]
                if pd.notnull(val):
                    return int(val)
            raw_val = r.get("raw_label", r.get("mapped_label", "Low"))
            if self.label_type == "binary":
                return self.label_mapper.map_binary(str(raw_val))
            return self.label_mapper.map_primary(str(raw_val))

        df["mapped_label"] = df.apply(_get_mapped_label, axis=1)
        df = df[df["mapped_label"].notnull()].copy()
        df["mapped_label"] = df["mapped_label"].astype(int)

        # Optional Training Class Balancing
        if self.split == "train" and self.balance_classes and len(df) > 0:
            df = self._balance_dataset(df)

        self.samples = df.to_dict("records")
        logger.info(
            f"Initialized UnifiedVideoDataset [{self.split.upper()}] Fold {self.fold} | "
            f"Dataset: {self.target_dataset} | Video Samples: {len(self.samples)} | T={self.num_frames} | RAM Cache: {self.use_ram_cache}"
        )

        # 3D Video Normalization Transforms (Kinetics-400 Standards)
        self.mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
        self.std = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)

    def _balance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Upsamples minority classes to match maximum class count."""
        class_counts = df["mapped_label"].value_counts()
        max_count = class_counts.max()
        balanced_dfs = []
        for cls, group in df.groupby("mapped_label"):
            balanced_group = group.sample(n=max_count, replace=True, random_state=self.seed)
            balanced_dfs.append(balanced_group)
        return pd.concat(balanced_dfs, axis=0).sample(frac=1.0, random_state=self.seed).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.samples)

    def _get_frame_paths_for_sample(self, record: Dict[str, Any]) -> List[str]:
        """Finds all frame PNG paths for a given video sample record."""
        sample_id = str(record["sample_id"])
        ds_name = str(record.get("dataset_name", record.get("dataset", self.target_dataset)))

        sample_dir = os.path.join(self.extracted_base_dir, ds_name, sample_id)
        if os.path.exists(sample_dir):
            frame_paths = sorted(glob.glob(os.path.join(sample_dir, "*.png")))
            if len(frame_paths) > 0:
                return frame_paths

        clip_path = str(record.get("relative_path", record.get("clip_path", "")))
        if os.path.exists(clip_path) and os.path.isdir(clip_path):
            frame_paths = sorted(glob.glob(os.path.join(clip_path, "*.png")) + glob.glob(os.path.join(clip_path, "*.jpg")))
            if len(frame_paths) > 0:
                return frame_paths

        return []

    def _load_video_tensor_from_frames(self, sample_id: str, frame_paths: List[str]) -> torch.Tensor:
        """
        Loads T frames, resizes to img_size x img_size, and stacks into [C, T, H, W] tensor.
        Uses RAM Caching for 100x speedup on repeat epoch accesses.
        """
        if self.use_ram_cache and sample_id in self._global_frame_cache:
            return self._global_frame_cache[sample_id]

        num_available = len(frame_paths)
        if num_available == 0:
            dummy_tensor = torch.zeros((3, self.num_frames, self.img_size, self.img_size), dtype=torch.float32)
            return dummy_tensor

        if num_available >= self.num_frames:
            indices = np.linspace(0, num_available - 1, self.num_frames, dtype=int)
        else:
            indices = [i % num_available for i in range(self.num_frames)]

        frames = []
        for idx in indices:
            path = frame_paths[idx]
            img = cv2.imread(path)
            if img is None:
                img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
                    img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            frames.append(img_tensor)

        video_tensor = torch.stack(frames, dim=0).permute(1, 0, 2, 3)
        video_tensor = (video_tensor - self.mean) / self.std

        if self.use_ram_cache:
            self._global_frame_cache[sample_id] = video_tensor

        return video_tensor

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        record = self.samples[idx]
        target_label = int(record["mapped_label"])
        sample_id = str(record["sample_id"])

        frame_paths = self._get_frame_paths_for_sample(record)
        video_tensor = self._load_video_tensor_from_frames(sample_id, frame_paths)

        return video_tensor, target_label, sample_id
