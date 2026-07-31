"""
Reusable PyTorch Dataset for Cross-Dataset Video Engagement Recognition.

Features:
    - Train / Validation / Test phase partitioning
    - Cross-dataset loading (OUC-CGE, COLER, or Combined)
    - Image loading & torchvision transforms
    - Class balancing via stratified downsampling
    - Dynamic attribute filtering
    - Config-driven initialization
"""

import csv
import json
import os
import random
from typing import List, Dict, Any, Optional, Tuple, Callable

from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Fallback dummy class if torch is missing
    class Dataset:
        pass

from cross_dataset.config.base_config import PipelineConfig, load_config
from cross_dataset.utils.logger import get_logger

logger = get_logger("UnifiedFrameDataset")


class UnifiedFrameDataset(Dataset):
    """
    Reusable PyTorch Dataset for single-frame engagement recognition.
    Supports train/val/test splits, cross-dataset loading, balancing, filtering, and transforms.
    """

    def __init__(
        self,
        phase: str = "train",
        fold: int = 0,
        target_dataset: Optional[str] = None,
        label_type: str = "primary",  # 'primary' (3-class) or 'binary' (2-class)
        transform: Optional[Callable] = None,
        balance_classes: bool = False,
        filters: Optional[Dict[str, Any]] = None,
        manifest_path: Optional[str] = None,
        config: Optional[PipelineConfig] = None,
    ):
        """
        Args:
            phase: Dataset phase ('train', 'val', 'test').
            fold: Target cross-validation fold (0 to 4).
            target_dataset: Filter by dataset ('OUC-CGE', 'COLER', or None for both).
            label_type: Target taxonomy ('primary' for 3-class, 'binary' for 2-class).
            transform: PyTorch/torchvision image transformation pipeline.
            balance_classes: If True, downsamples majority classes to equal frequency.
            filters: Optional dict of key-value filters (e.g. {'goc': 'Goc1'}).
            manifest_path: Path to unified_frame_dataset.csv (defaults to config location).
            config: PipelineConfig instance.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch and Torchvision must be installed to use UnifiedFrameDataset.")

        self.phase = phase.lower()
        self.fold = fold
        self.target_dataset = target_dataset
        self.label_type = label_type.lower()
        self.balance_classes = balance_classes
        self.filters = filters or {}
        self.cfg = config or load_config()

        if manifest_path is None:
            manifest_path = os.path.join(self.cfg.paths.manifest_dir, "unified_frame_dataset.csv")

        self.manifest_path = os.path.abspath(manifest_path)
        self.root_dir = self.cfg.paths.root_dir

        # Setup image transforms
        self.transform = transform or self._get_default_transforms(self.phase)

        # Load and process samples from manifest
        self.samples = self._load_and_filter_samples()

        if self.balance_classes and self.phase == "train":
            self.samples = self._balance_samples(self.samples)

        logger.info(
            f"Initialized UnifiedFrameDataset [{self.phase.upper()}] Fold {self.fold} | "
            f"Dataset: {self.target_dataset or 'COMBINED'} | "
            f"Samples: {len(self.samples)} | Balanced: {self.balance_classes}"
        )

    def _get_default_transforms(self, phase: str) -> Callable:
        """
        Returns default torchvision image preprocessing and normalization transforms.
        """
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        if phase == "train":
            return T.Compose([
                T.Resize((256, 256)),
                T.RandomCrop((224, 224)),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.Normalize(mean=imagenet_mean, std=imagenet_std),
            ])
        else:
            return T.Compose([
                T.Resize((256, 256)),
                T.CenterCrop((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=imagenet_mean, std=imagenet_std),
            ])

    def _load_and_filter_samples(self) -> List[Dict[str, Any]]:
        """
        Loads CSV manifest and applies phase, fold, dataset, and attribute filtering.
        """
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"Frame manifest CSV not found at: {self.manifest_path}")

        filtered_samples = []

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                meta = json.loads(row.get("metadata", "{}"))
                sample_fold = int(meta.get("split_fold", 0))

                # 1. Dataset Filtering
                if self.target_dataset and row["dataset"].upper() != self.target_dataset.upper():
                    continue

                # 2. Phase & Fold Partitioning
                if self.phase == "train":
                    if sample_fold == self.fold:
                        continue  # Validation fold excluded from train
                elif self.phase in ("val", "validation"):
                    if sample_fold != self.fold:
                        continue  # Only target validation fold included
                elif self.phase == "test":
                    # Test split can use designated fold or un-seen split
                    if sample_fold != self.fold:
                        continue

                # 3. Dynamic Metadata Attribute Filtering
                skip = False
                for f_key, f_val in self.filters.items():
                    val_in_meta = meta.get(f_key, row.get(f_key, None))
                    if val_in_meta != f_val:
                        skip = True
                        break
                if skip:
                    continue

                # Resolve Absolute Image Path
                img_path = row["image_path"]
                if not os.path.isabs(img_path):
                    img_path = os.path.join(self.root_dir, img_path)

                row["abs_image_path"] = img_path
                row["parsed_metadata"] = meta
                filtered_samples.append(row)

        return filtered_samples

    def _balance_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Balances sample class frequencies via downsampling majority classes to minority class count.
        """
        if not samples:
            return samples

        label_key = "mapped_label_3class" if self.label_type == "primary" else "mapped_label_binary_int"
        
        class_groups: Dict[int, List[Dict[str, Any]]] = {}
        for s in samples:
            lbl = int(s[label_key])
            if lbl not in class_groups:
                class_groups[lbl] = []
            class_groups[lbl].append(s)

        min_count = min(len(grp) for grp in class_groups.values())
        
        rng = random.Random(self.cfg.preprocessing.random_seed)
        balanced = []

        for lbl, grp in class_groups.items():
            sampled_grp = rng.sample(grp, k=min_count)
            balanced.extend(sampled_grp)

        rng.shuffle(balanced)
        return balanced

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Retrieves transformed frame tensor, integer target label, and sample metadata dict.

        Returns:
            Tuple[Tensor, Tensor, Dict]:
                - image_tensor: Image tensor of shape [C, H, W]
                - target_label: Integer class label Tensor (0, 1, or 2)
                - info: Sample details dictionary
        """
        sample_info = self.samples[idx]
        img_path = sample_info["abs_image_path"]

        # Safe Image Loading
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image at {img_path}: {e}")
            # Dummy black image fallback to prevent crash during training
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        # Apply transforms
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = T.ToTensor()(image)

        # Target label extraction
        if self.label_type == "primary":
            target_val = int(sample_info["mapped_label_3class"])
        else:
            target_val = int(sample_info["mapped_label_binary_int"])

        target_tensor = torch.tensor(target_val, dtype=torch.long)

        info = {
            "dataset": sample_info["dataset"],
            "clip_id": sample_info["clip_id"],
            "frame_id": int(sample_info["frame_id"]),
            "image_path": sample_info["image_path"],
            "original_label": sample_info["original_label"],
            "mapped_label": sample_info["mapped_label"],
            "split_fold": sample_info["parsed_metadata"].get("split_fold", 0),
        }

        return image_tensor, target_tensor, info
