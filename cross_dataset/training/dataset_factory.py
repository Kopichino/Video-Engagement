"""
DataLoader Factory for Experiment A (2D Frame-Level) and Experiment B (3D Video Sequence Level).
Builds PyTorch DataLoaders for train, validation, and test splits.
"""

from typing import Dict, Any, Tuple, Optional

import torch
from torch.utils.data import DataLoader

from cross_dataset.datasets.frame_dataset import UnifiedFrameDataset
from cross_dataset.datasets.video_dataset import UnifiedVideoDataset
from cross_dataset.utils.logger import get_logger

logger = get_logger("DatasetFactory")


def build_dataloaders(
    dataset_config: Dict[str, Any]
) -> Dict[str, DataLoader]:
    """
    Builds PyTorch DataLoaders for training, validation, and testing.

    Args:
        dataset_config: Dataset section dictionary from experiment config.

    Returns:
        Dict[str, DataLoader]: Dictionary containing 'train', 'val', and 'test' DataLoaders.
    """
    is_video_3d = bool(dataset_config.get("is_video_3d", False) or dataset_config.get("is_3d", False))
    
    if is_video_3d:
        default_manifest = "data/manifests/unified_manifest.csv"
    else:
        default_manifest = "data/manifests/unified_frame_dataset.csv"

    manifest_path = dataset_config.get("manifest_path", default_manifest)
    target_dataset = dataset_config.get("target_dataset", None)  # OUC-CGE, COLER, or None
    eval_target_dataset = dataset_config.get("eval_target_dataset", target_dataset)
    label_type = dataset_config.get("label_type", "primary")
    fold = int(dataset_config.get("fold", 0))
    balance_classes = bool(dataset_config.get("balance_classes", True))

    batch_size = int(dataset_config.get("batch_size", 16 if is_video_3d else 32))
    eval_batch_size = int(dataset_config.get("eval_batch_size", batch_size))
    num_workers = int(dataset_config.get("num_workers", 0))
    pin_memory = bool(dataset_config.get("pin_memory", True))

    is_cross_eval = (target_dataset is not None) and (eval_target_dataset is not None) and (target_dataset.upper() != eval_target_dataset.upper())

    logger.info(
        f"Building DataLoaders | Mode: {'3D Video' if is_video_3d else '2D Frame'} | "
        f"Target Train Dataset: {target_dataset or 'COMBINED'} | "
        f"Eval Target Dataset: {eval_target_dataset or 'COMBINED'} | Fold: {fold} | CrossEval: {is_cross_eval}"
    )

    if is_video_3d:
        num_frames = int(dataset_config.get("num_frames", 16))
        img_size = int(dataset_config.get("img_size", 224))

        use_ram_cache = bool(dataset_config.get("use_ram_cache", False))

        train_dataset = UnifiedVideoDataset(
            manifest_path=manifest_path,
            target_dataset=target_dataset,
            split="train",
            fold=fold,
            num_frames=num_frames,
            img_size=img_size,
            label_type=label_type,
            balance_classes=balance_classes,
            use_ram_cache=use_ram_cache,
        )

        val_dataset = UnifiedVideoDataset(
            manifest_path=manifest_path,
            target_dataset=target_dataset,
            split="val",
            fold=fold,
            num_frames=num_frames,
            img_size=img_size,
            label_type=label_type,
            balance_classes=False,
            use_ram_cache=use_ram_cache,
        )

        test_dataset = UnifiedVideoDataset(
            manifest_path=manifest_path,
            target_dataset=eval_target_dataset,
            split="test",
            fold=fold,
            num_frames=num_frames,
            img_size=img_size,
            label_type=label_type,
            balance_classes=False,
            use_ram_cache=use_ram_cache,
        )
    else:
        train_dataset = UnifiedFrameDataset(
            phase="train",
            fold=fold,
            target_dataset=target_dataset,
            label_type=label_type,
            balance_classes=balance_classes,
            manifest_path=manifest_path,
        )

        val_dataset = UnifiedFrameDataset(
            phase="val",
            fold=fold,
            target_dataset=target_dataset,
            label_type=label_type,
            balance_classes=False,
            manifest_path=manifest_path,
        )

        test_dataset = UnifiedFrameDataset(
            phase="test",
            fold=fold,
            target_dataset=eval_target_dataset,
            label_type=label_type,
            balance_classes=False,
            manifest_path=manifest_path,
            is_cross_dataset_eval=is_cross_eval,
        )

    # Create PyTorch DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory if torch.cuda.is_available() else False,
        drop_last=True if len(train_dataset) > batch_size else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory if torch.cuda.is_available() else False,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
