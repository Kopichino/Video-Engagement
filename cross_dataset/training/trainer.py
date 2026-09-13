"""
Reusable Trainer Class for Experiment A Framework.

Supports:
    - Automatic Mixed Precision (AMP)
    - Gradient Clipping & Gradient Accumulation
    - Early Stopping & Best Checkpoint Selection
    - Automatic Training Resumption
    - Multi-Scheduler (Cosine, Step, ReduceLROnPlateau) & Multi-Optimizer (AdamW, SGD)
    - Full Metric Logging & Plot Generation
"""

import gc
import json
import os
import random
import time
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cross_dataset.training.checkpoint_manager import CheckpointManager
from cross_dataset.training.config_parser import ExperimentConfig
from cross_dataset.training.dataset_factory import build_dataloaders
from cross_dataset.training.evaluator import Evaluator
from cross_dataset.training.logger import ExperimentLogger
from cross_dataset.training.metrics import MetricsCalculator
from cross_dataset.training.model_factory import build_model
from cross_dataset.training.optimizer_factory import build_optimizer
from cross_dataset.training.scheduler_factory import build_scheduler
from cross_dataset.utils.logger import get_logger

logger = get_logger("Trainer")


class EarlyStopping:
    """
    Early stopping handler to stop training when monitored validation metric stops improving.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        monitor: str = "val_macro_f1",
        mode: str = "max",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode.lower()

        self.counter = 0
        self.best_score = float("-inf") if self.mode == "max" else float("inf")
        self.early_stop = False

    def check(self, current_val: float) -> bool:
        """
        Updates early stopping status based on current metric value.

        Returns:
            bool: True if training should be stopped early.
        """
        if self.mode == "max":
            improved = current_val > (self.best_score + self.min_delta)
        else:
            improved = current_val < (self.best_score - self.min_delta)

        if improved:
            self.best_score = current_val
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} / {self.patience} (Best {self.monitor}: {self.best_score:.4f})")
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class Trainer:
    """
    Main Trainer class executing training, validation, checkpointing, and evaluation.
    """

    def __init__(self, config: ExperimentConfig):
        """
        Args:
            config: ExperimentConfig instance parsed from YAML config file.
        """
        self.config = config

        # Extract config sections
        self.exp_cfg = config["experiment"]
        self.ds_cfg = config["dataset"]
        self.model_cfg = config["model"]
        self.train_cfg = config["training"]
        self.opt_cfg = config["optimizer"]
        self.sched_cfg = config["scheduler"]

        # Setup Compute Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using compute device: {self.device}")

        # Reproducibility Setup
        seed = self.exp_cfg.get("seed", 42)
        self._set_deterministic_seed(seed)

        # Output Directory & Logger Setup
        self.output_dir = os.path.abspath(self.exp_cfg["output_dir"])
        self.exp_logger = ExperimentLogger(output_dir=self.output_dir, experiment_name=self.exp_cfg["name"])

        # Save Configuration Snapshot
        snapshot_path = os.path.join(self.output_dir, "config_snapshot.yaml")
        config.save_snapshot(snapshot_path)
        logger.info(f"Saved experiment configuration snapshot to: {snapshot_path}")

        # Build Model
        self.model = build_model(self.model_cfg).to(self.device)

        # Build Optimizer
        self.optimizer = build_optimizer(self.model, self.opt_cfg)

        # Build Scheduler
        self.total_epochs = int(self.train_cfg.get("epochs", 20))
        self.scheduler = build_scheduler(self.optimizer, self.sched_cfg, total_epochs=self.total_epochs)

        # Build Loss Criterion
        self.criterion = nn.CrossEntropyLoss()

        # Build DataLoaders
        self.dataloaders = build_dataloaders(self.ds_cfg)
        self.train_loader = self.dataloaders["train"]
        self.val_loader = self.dataloaders["val"]
        self.test_loader = self.dataloaders["test"]

        # Setup AMP Scaler (Modern torch.amp API)
        self.use_amp = bool(self.train_cfg.get("use_amp", True)) and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        # Training Hyperparameters
        self.gradient_clipping = float(self.train_cfg.get("gradient_clipping", 1.0))
        self.gradient_accumulation_steps = int(self.train_cfg.get("gradient_accumulation_steps", 1))

        # Setup Evaluator
        num_classes = int(self.model_cfg.get("num_classes", 3))
        self.evaluator = Evaluator(
            model=self.model,
            criterion=self.criterion,
            device=self.device,
            use_amp=self.use_amp,
            num_classes=num_classes,
        )

        # Setup Checkpoint Manager
        es_cfg = self.train_cfg.get("early_stopping", {})
        monitor_metric = es_cfg.get("monitor", "val_macro_f1")
        monitor_mode = es_cfg.get("mode", "max")

        self.checkpoint_mgr = CheckpointManager(
            checkpoint_dir=self.exp_logger.checkpoints_dir,
            monitor_metric=monitor_metric,
            mode=monitor_mode,
        )

        # Setup Early Stopping
        self.early_stopping_enabled = bool(es_cfg.get("enabled", True))
        if self.early_stopping_enabled:
            self.early_stopper = EarlyStopping(
                patience=int(es_cfg.get("patience", 5)),
                min_delta=float(es_cfg.get("min_delta", 0.001)),
                monitor=monitor_metric,
                mode=monitor_mode,
            )

        # Auto-Resume Checkpoint
        self.start_epoch = 1
        if self.train_cfg.get("auto_resume", True):
            resumed_epoch, is_resumed = self.checkpoint_mgr.auto_resume_if_available(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
            )
            if is_resumed:
                if resumed_epoch > self.total_epochs:
                    logger.info(
                        f"Old checkpoint completed {resumed_epoch - 1} epochs (>= total_epochs {self.total_epochs}). "
                        f"Resetting start_epoch to 1 for clean training."
                    )
                    self.start_epoch = 1
                else:
                    self.start_epoch = resumed_epoch

    def _set_deterministic_seed(self, seed: int):
        """Sets deterministic random seeds across Python, NumPy, and PyTorch."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def train_epoch(self, epoch: int) -> Dict[str, Any]:
        """
        Executes one training epoch with AMP, gradient accumulation, clipping, and memory optimization.

        Returns:
            Dict[str, Any]: Training metrics dictionary for the epoch.
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        total_samples = 0
        y_true_list = []
        y_pred_list = []

        start_time = time.time()

        for batch_idx, (images, targets, _) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # AMP Forward Pass
            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    logits = self.model(images)
                    loss = self.criterion(logits, targets)
                    loss_scaled = loss / self.gradient_accumulation_steps
            else:
                logits = self.model(images)
                loss = self.criterion(logits, targets)
                loss_scaled = loss / self.gradient_accumulation_steps

            # Backward Pass with GradScaler
            self.scaler.scale(loss_scaled).backward()

            # Step Optimizer after accumulation steps
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                if self.gradient_clipping > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = torch.argmax(logits, dim=-1)
            y_true_list.extend(targets.cpu().numpy())
            y_pred_list.extend(preds.cpu().numpy())

        elapsed = time.time() - start_time
        avg_loss = (running_loss / total_samples) if total_samples > 0 else 0.0

        y_true_arr = np.array(y_true_list, dtype=np.int64)
        y_pred_arr = np.array(y_pred_list, dtype=np.int64)
        num_classes = int(self.model_cfg.get("num_classes", 3))

        metrics = MetricsCalculator.compute_metrics(y_true_arr, y_pred_arr, num_classes=num_classes)
        metrics["loss"] = round(avg_loss, 4)
        metrics["epoch_time_sec"] = round(elapsed, 2)

        return metrics

    def fit(self) -> Dict[str, Any]:
        """
        Executes the full multi-epoch training loop.

        Returns:
            Dict[str, Any]: Final evaluation metrics dictionary on test set.
        """
        logger.info(f"Starting Training: Epochs {self.start_epoch} -> {self.total_epochs}")

        for epoch in range(self.start_epoch, self.total_epochs + 1):
            # 1. Train Epoch
            train_metrics = self.train_epoch(epoch)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 2. Validation Epoch
            val_metrics, _ = self.evaluator.evaluate(self.val_loader)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Get current learning rate
            current_lr = float(self.optimizer.param_groups[0]["lr"])

            # 3. Update Learning Rate Scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    monitor_val = val_metrics.get("loss", 0.0) if self.scheduler.mode == "min" else val_metrics.get("macro_f1", 0.0)
                    self.scheduler.step(monitor_val)
                else:
                    self.scheduler.step()

            # 4. Log Epoch Metrics to TensorBoard and CSV
            self.exp_logger.log_epoch_metrics(epoch, train_metrics, val_metrics, current_lr)
            self.exp_logger.generate_plots()

            # Log Progress to Stdout
            logger.info(
                f"Epoch [{epoch:02d}/{self.total_epochs:02d}] | "
                f"Train Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.4f} || "
                f"Val Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | "
                f"Macro F1: {val_metrics['macro_f1']:.4f} | QWK: {val_metrics['qwk']:.4f} | LR: {current_lr:.6e}"
            )

            # 5. Save Checkpoints
            monitor_metric = self.checkpoint_mgr.monitor_metric
            val_score = val_metrics.get(monitor_metric.replace("val_", ""), val_metrics.get(monitor_metric, 0.0))
            is_best = self.checkpoint_mgr.is_better(val_score)

            self.checkpoint_mgr.save_checkpoint(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                metrics=val_metrics,
                is_best=is_best,
            )

            # Clean memory at epoch end
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 6. Early Stopping Check
            if self.early_stopping_enabled and self.early_stopper.check(val_score):
                logger.info(f"Early stopping triggered at epoch {epoch}. Terminating training loop.")
                break

        # Load Best Checkpoint for Final Test Set Evaluation
        logger.info("Training complete. Loading best model for final evaluation on test set...")
        best_path = self.checkpoint_mgr.best_model_path
        if os.path.exists(best_path):
            self.checkpoint_mgr.load_checkpoint(best_path, self.model)

        test_pred_path = os.path.join(self.exp_logger.predictions_dir, "test_predictions.csv")
        test_metrics, _ = self.evaluator.evaluate(self.test_loader, save_predictions_path=test_pred_path)

        # Plot Confusion Matrix & ROC Curve for Test Evaluation
        class_names = ["Low", "Mid", "High"] if self.evaluator.num_classes == 3 else ["Not Engaged", "Engaged"]
        self.exp_logger.plot_confusion_matrix(
            cm=np.array(test_metrics["confusion_matrix"]),
            class_names=class_names,
            filename="test_confusion_matrix.png",
        )

        if "fpr" in test_metrics and "tpr" in test_metrics:
            self.exp_logger.plot_roc_curve(
                fpr=test_metrics["fpr"],
                tpr=test_metrics["tpr"],
                roc_auc=test_metrics.get("roc_auc", 0.0),
                filename="test_roc_curve.png",
            )

        logger.info(f"Final Test Evaluation Metrics: {json.dumps(test_metrics, indent=2)}")

        self.exp_logger.close()
        return test_metrics
