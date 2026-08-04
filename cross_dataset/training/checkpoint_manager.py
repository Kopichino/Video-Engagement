"""
Checkpoint Manager for Experiment A Training Framework.

Handles:
    - Model & training state serialization (best_model.pth, last_model.pth)
    - AMP scaler, optimizer, and scheduler state saving
    - Automatic training resumption from checkpoints
    - Metric monitoring & best model selection
"""

import os
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from cross_dataset.utils.logger import get_logger

logger = get_logger("CheckpointManager")


class CheckpointManager:
    """
    Manages saving, loading, and automatic resumption of model checkpoints.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        monitor_metric: str = "val_macro_f1",
        mode: str = "max",
    ):
        """
        Args:
            checkpoint_dir: Path to checkpoints directory.
            monitor_metric: Metric key to monitor for best model selection.
            mode: 'max' (higher is better) or 'min' (lower is better).
        """
        self.checkpoint_dir = os.path.abspath(checkpoint_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.monitor_metric = monitor_metric
        self.mode = mode.lower()

        self.best_metric_val = float("-inf") if self.mode == "max" else float("inf")
        self.best_epoch = 0

        self.best_model_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        self.last_model_path = os.path.join(self.checkpoint_dir, "last_model.pth")

    def is_better(self, current_val: float) -> bool:
        """Determines if the current metric value outperforms the best value recorded."""
        if self.mode == "max":
            return current_val > self.best_metric_val
        else:
            return current_val < self.best_metric_val

    def save_checkpoint(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        metrics: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
    ) -> str:
        """
        Saves a complete training state checkpoint.

        Args:
            epoch: Current epoch number (1-indexed).
            model: PyTorch model.
            optimizer: PyTorch optimizer.
            scheduler: PyTorch LR scheduler (optional).
            scaler: PyTorch AMP GradScaler (optional).
            metrics: Current epoch validation metrics dictionary.
            is_best: True if this checkpoint represents the best model so far.

        Returns:
            str: Path to saved checkpoint file.
        """
        state_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "best_metric_val": self.best_metric_val,
            "metrics": metrics or {},
        }

        # Always update last_model.pth
        torch.save(state_dict, self.last_model_path)
        logger.debug(f"Saved last model checkpoint to: {self.last_model_path}")

        # If best model, save best_model.pth
        if is_best:
            if metrics:
                clean_metric_key = self.monitor_metric.replace("val_", "")
                if clean_metric_key in metrics:
                    self.best_metric_val = float(metrics[clean_metric_key])
                elif self.monitor_metric in metrics:
                    self.best_metric_val = float(metrics[self.monitor_metric])

            self.best_epoch = epoch
            state_dict["best_metric_val"] = self.best_metric_val
            torch.save(state_dict, self.best_model_path)
            logger.info(
                f"[BEST MODEL] Epoch {epoch} | "
                f"{self.monitor_metric}: {self.best_metric_val:.4f} -> {self.best_model_path}"
            )

        return self.last_model_path

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Loads state dicts from a checkpoint file into model, optimizer, and scheduler.

        Args:
            checkpoint_path: Path to checkpoint .pth file.
            model: Model to load weights into.
            optimizer: Optional optimizer to restore state.
            scheduler: Optional scheduler to restore state.
            scaler: Optional AMP scaler to restore state.

        Returns:
            Tuple[int, Dict[str, Any]]: (resumed_epoch, saved_metrics).
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint state from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Restore Model Weights
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Restored model parameter weights.")

        # Restore Optimizer State if provided
        if optimizer is not None and "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"]:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info("Restored optimizer state.")

        # Restore Scheduler State if provided
        if scheduler is not None and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            logger.info("Restored learning rate scheduler state.")

        # Restore Scaler State if provided
        if scaler is not None and "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"]:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            logger.info("Restored AMP GradScaler state.")

        epoch = checkpoint.get("epoch", 0)
        self.best_metric_val = checkpoint.get("best_metric_val", self.best_metric_val)
        saved_metrics = checkpoint.get("metrics", {})

        return epoch, saved_metrics

    def auto_resume_if_available(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
    ) -> Tuple[int, bool]:
        """
        Attempts to automatically resume from last_model.pth if it exists.

        Returns:
            Tuple[int, bool]: (start_epoch, is_resumed).
        """
        if os.path.exists(self.last_model_path):
            try:
                resumed_epoch, _ = self.load_checkpoint(
                    self.last_model_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                )
                logger.info(f"Auto-resume successful! Resuming training from epoch {resumed_epoch + 1}.")
                return resumed_epoch + 1, True
            except Exception as e:
                logger.warning(f"Auto-resume failed from {self.last_model_path}: {e}")

        return 1, False
