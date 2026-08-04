"""
Optimizer Factory for Experiment A.
Supports AdamW and SGD with configurable learning rates and weight decay.
"""

from typing import Dict, Any, Iterable

import torch
import torch.nn as nn
import torch.optim as optim

from cross_dataset.utils.logger import get_logger

logger = get_logger("OptimizerFactory")


def build_optimizer(model: nn.Module, opt_config: Dict[str, Any]) -> optim.Optimizer:
    """
    Builds PyTorch optimizer instance from configuration.

    Args:
        model: PyTorch model instance whose parameters will be optimized.
        opt_config: Optimizer configuration dictionary.

    Returns:
        optim.Optimizer: Configured PyTorch optimizer.
    """
    opt_type = opt_config.get("type", "adamw").lower()
    lr = float(opt_config.get("lr", 0.0001))
    weight_decay = float(opt_config.get("weight_decay", 0.01))

    # Filter trainable parameters (requires_grad = True)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if not trainable_params:
        logger.warning("No trainable parameters found in model! Check layer freezing configurations.")
        trainable_params = list(model.parameters())

    if opt_type == "adamw":
        logger.info(f"Building AdamW Optimizer (lr={lr}, weight_decay={weight_decay})")
        optimizer = optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
        )

    elif opt_type == "sgd":
        momentum = float(opt_config.get("momentum", 0.9))
        nesterov = bool(opt_config.get("nesterov", True))
        logger.info(f"Building SGD Optimizer (lr={lr}, momentum={momentum}, nesterov={nesterov}, weight_decay={weight_decay})")
        optimizer = optim.SGD(
            trainable_params,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
        )

    elif opt_type == "adam":
        logger.info(f"Building Adam Optimizer (lr={lr}, weight_decay={weight_decay})")
        optimizer = optim.Adam(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
        )

    else:
        raise ValueError(f"Unsupported optimizer type: '{opt_type}'. Choose 'adamw', 'sgd', or 'adam'.")

    return optimizer
