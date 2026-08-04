"""
DataLoader Factory for Experiment A.
Builds PyTorch DataLoaders for train, validation, and test splits.
"""

from typing import Dict, Any, Tuple, Optional

import torch
from torch.utils.data import DataLoader

from cross_dataset.datasets.frame_dataset import UnifiedFrameDataset
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
    manifest_path = dataset_config.get("manifest_path", "data/manifests/unified_frame_dataset.csv")
    target_dataset = dataset_config.get("target_dataset", None)  # OUC-CGE, COLER, or None
    eval_target_dataset = dataset_config.get("eval_target_dataset", target_dataset)
    label_type = dataset_config.get("label_type", "primary")
    fold = int(dataset_config.get("fold", 0))
    balance_classes = bool(dataset_config.get("balance_classes", True))

    batch_size = int(dataset_config.get("batch_size", 32))
    eval_batch_size = int(dataset_config.get("eval_batch_size", batch_size))
    num_workers = int(dataset_config.get("num_workers", 0))
    pin_memory = bool(dataset_config.get("pin_memory", True))

    is_cross_eval = (target_dataset is not None) and (eval_target_dataset is not None) and (target_dataset.upper() != eval_target_dataset.upper())

    logger.info(
        f"Building DataLoaders | Target Train Dataset: {target_dataset or 'COMBINED'} | "
        f"Eval Target Dataset: {eval_target_dataset or 'COMBINED'} | Fold: {fold} | CrossEval: {is_cross_eval}"
    )

    # 1. Train Dataset
    train_dataset = UnifiedFrameDataset(
        phase="train",
        fold=fold,
        target_dataset=target_dataset,
        label_type=label_type,
        balance_classes=balance_classes,
        manifest_path=manifest_path,
    )

    # 2. Validation Dataset
    val_dataset = UnifiedFrameDataset(
        phase="val",
        fold=fold,
        target_dataset=target_dataset,
        label_type=label_type,
        balance_classes=False,
        manifest_path=manifest_path,
    )

    # 3. Test Dataset (Evaluates on target or cross-dataset transfer target)
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
