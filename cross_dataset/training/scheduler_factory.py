"""
Learning Rate Scheduler Factory for Experiment A.
Supports CosineAnnealingLR, StepLR, and ReduceLROnPlateau.
"""

from typing import Dict, Any, Optional

import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    ReduceLROnPlateau,
    _LRScheduler,
)

from cross_dataset.utils.logger import get_logger

logger = get_logger("SchedulerFactory")


def build_scheduler(
    optimizer: optim.Optimizer, sched_config: Dict[str, Any], total_epochs: int = 20
) -> Optional[Any]:
    """
    Builds PyTorch learning rate scheduler instance from configuration.

    Args:
        optimizer: PyTorch optimizer instance.
        sched_config: Scheduler configuration dictionary.
        total_epochs: Total training epochs.

    Returns:
        Optional PyTorch LR scheduler instance.
    """
    sched_type = sched_config.get("type", "cosine").lower()

    if not sched_type or sched_type == "none":
        logger.info("No learning rate scheduler requested.")
        return None

    if sched_type == "cosine":
        t_max = int(sched_config.get("t_max", total_epochs))
        eta_min = float(sched_config.get("eta_min", 1e-6))
        logger.info(f"Building CosineAnnealingLR Scheduler (T_max={t_max}, eta_min={eta_min})")
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=eta_min,
        )

    elif sched_type in ("step", "steplr"):
        step_size = int(sched_config.get("step_size", 5))
        gamma = float(sched_config.get("gamma", 0.5))
        logger.info(f"Building StepLR Scheduler (step_size={step_size}, gamma={gamma})")
        scheduler = StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )

    elif sched_type in ("plateau", "reducelronplateau"):
        mode = sched_config.get("mode", "max")
        factor = float(sched_config.get("factor", 0.5))
        patience = int(sched_config.get("patience", 3))
        min_lr = float(sched_config.get("min_lr", 1e-6))
        logger.info(f"Building ReduceLROnPlateau Scheduler (mode={mode}, factor={factor}, patience={patience})")
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )

    else:
        raise ValueError(f"Unsupported scheduler type: '{sched_type}'. Choose 'cosine', 'step', or 'plateau'.")

    return scheduler
