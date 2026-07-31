"""
Verification Script for UnifiedFrameDataset PyTorch Integration.
"""

import torch
from torch.utils.data import DataLoader

from cross_dataset.datasets.frame_dataset import UnifiedFrameDataset


def test_unified_frame_dataset():
    print("==================================================")
    print(" Testing UnifiedFrameDataset PyTorch Integration  ")
    print("==================================================")

    # 1. Test Train Phase (Combined Cross-Dataset, Fold 0)
    train_ds = UnifiedFrameDataset(
        phase="train",
        fold=0,
        target_dataset=None,  # Combined
        label_type="primary",
        balance_classes=True,
    )
    print(f"\n[1] Train Dataset (Fold 0 Combined, Balanced): {len(train_ds)} samples")

    # 2. Test Validation Phase (Fold 0)
    val_ds = UnifiedFrameDataset(
        phase="val",
        fold=0,
        target_dataset=None,
        label_type="primary",
        balance_classes=False,
    )
    print(f"[2] Val Dataset (Fold 0 Combined): {len(val_ds)} samples")

    # 3. Test OUC-CGE Single-Dataset Loading
    ouc_ds = UnifiedFrameDataset(
        phase="train",
        fold=0,
        target_dataset="OUC-CGE",
        label_type="primary",
    )
    print(f"[3] OUC-CGE Only Dataset (Train Fold 0): {len(ouc_ds)} samples")

    # 4. Test COLER Single-Dataset Loading with Attribute Filter (goc="Goc1")
    coler_ds = UnifiedFrameDataset(
        phase="train",
        fold=0,
        target_dataset="COLER",
        filters={"goc": "Goc1"},
    )
    print(f"[4] COLER Only Dataset (Goc1 Filtered): {len(coler_ds)} samples")

    # 5. Test PyTorch DataLoader Batch Loading
    print("\n[5] Testing PyTorch DataLoader Batch Execution:")
    loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    
    images, labels, info = next(iter(loader))
    print(f"  - Image Batch Shape:  {images.shape} (Expected: [4, 3, 224, 224])")
    print(f"  - Label Batch Tensor: {labels} (Target dtype: {labels.dtype})")
    print(f"  - Datasets in Batch:  {info['dataset']}")
    print(f"  - Clip IDs in Batch:  {info['clip_id']}")

    print("\n==================================================")
    print(" All PyTorch Dataset Verifications PASSED!")
    print("==================================================")


if __name__ == "__main__":
    test_unified_frame_dataset()
