"""
Experiment Logger & Plotter for Experiment A Framework.

Manages:
    - Directory hierarchy creation (checkpoints, metrics, plots, predictions, logs)
    - TensorBoard SummaryWriter integration
    - CSV history tracking (history.csv)
    - Matplotlib/Seaborn visualization generation (loss/acc curves, LR curves, confusion matrix, ROC curve)
"""

import csv
import json
import os
from typing import Dict, Any, List, Optional

import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from cross_dataset.utils.logger import get_logger

logger = get_logger("ExperimentLogger")


class ExperimentLogger:
    """
    Manages output directories, TensorBoard logging, CSV metric history, and plot generation.
    """

    def __init__(self, output_dir: str, experiment_name: str = "exp_a"):
        self.output_dir = os.path.abspath(output_dir)
        self.experiment_name = experiment_name

        # Create Output Subdirectories
        self.checkpoints_dir = os.path.join(self.output_dir, "checkpoints")
        self.metrics_dir = os.path.join(self.output_dir, "metrics")
        self.plots_dir = os.path.join(self.output_dir, "plots")
        self.predictions_dir = os.path.join(self.output_dir, "predictions")
        self.logs_dir = os.path.join(self.output_dir, "logs")

        for d in [
            self.output_dir,
            self.checkpoints_dir,
            self.metrics_dir,
            self.plots_dir,
            self.predictions_dir,
            self.logs_dir,
        ]:
            os.makedirs(d, exist_ok=True)

        # TensorBoard Writer
        if TENSORBOARD_AVAILABLE:
            self.tb_writer = SummaryWriter(log_dir=self.logs_dir)
            logger.info(f"TensorBoard SummaryWriter initialized at: {self.logs_dir}")
        else:
            self.tb_writer = None
            logger.warning("TensorBoard not available. Install 'tensorboard' for event logging.")

        # CSV History File
        self.csv_history_path = os.path.join(self.metrics_dir, "history.csv")
        self.history_fields = [
            "epoch",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "val_macro_f1",
            "val_macro_precision",
            "val_macro_recall",
            "val_qwk",
            "lr",
        ]
        self._init_csv_history()

    def _init_csv_history(self):
        """Initializes history.csv header if file doesn't exist."""
        if not os.path.exists(self.csv_history_path):
            with open(self.csv_history_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.history_fields)

    def log_epoch_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, Any],
        val_metrics: Dict[str, Any],
        current_lr: float,
    ):
        """
        Logs epoch metrics to TensorBoard and appends to history.csv.
        """
        tr_loss = train_metrics.get("loss", 0.0)
        tr_acc = train_metrics.get("accuracy", 0.0)
        val_loss = val_metrics.get("loss", 0.0)
        val_acc = val_metrics.get("accuracy", 0.0)
        val_f1 = val_metrics.get("macro_f1", 0.0)
        val_prec = val_metrics.get("macro_precision", 0.0)
        val_rec = val_metrics.get("macro_recall", 0.0)
        val_qwk = val_metrics.get("qwk", 0.0)

        # 1. TensorBoard Logging
        if self.tb_writer:
            self.tb_writer.add_scalar("Loss/Train", tr_loss, epoch)
            self.tb_writer.add_scalar("Loss/Val", val_loss, epoch)
            self.tb_writer.add_scalar("Accuracy/Train", tr_acc, epoch)
            self.tb_writer.add_scalar("Accuracy/Val", val_acc, epoch)
            self.tb_writer.add_scalar("Metrics/Val_Macro_F1", val_f1, epoch)
            self.tb_writer.add_scalar("Metrics/Val_QWK", val_qwk, epoch)
            self.tb_writer.add_scalar("LearningRate", current_lr, epoch)
            self.tb_writer.flush()

        # 2. CSV History Logging
        row = [
            epoch,
            round(tr_loss, 4),
            round(tr_acc, 4),
            round(val_loss, 4),
            round(val_acc, 4),
            round(val_f1, 4),
            round(val_prec, 4),
            round(val_rec, 4),
            round(val_qwk, 4),
            f"{current_lr:.6e}",
        ]
        with open(self.csv_history_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        # 3. Save JSON Metrics Snapshot
        epoch_metrics_json = os.path.join(self.metrics_dir, f"epoch_{epoch:03d}_metrics.json")
        with open(epoch_metrics_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "epoch": epoch,
                    "train": train_metrics,
                    "val": val_metrics,
                    "lr": current_lr,
                },
                f,
                indent=2,
            )

    def generate_plots(self, history_data: Optional[Dict[str, List[float]]] = None):
        """
        Generates matplotlib/seaborn visualization plots from history.csv.
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib/Seaborn not installed. Skipping plot generation.")
            return

        if not os.path.exists(self.csv_history_path):
            return

        # Load history data from CSV
        epochs, tr_loss, val_loss, tr_acc, val_acc, val_f1, lrs = [], [], [], [], [], [], []

        with open(self.csv_history_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row["epoch"]))
                tr_loss.append(float(row["train_loss"]))
                val_loss.append(float(row["val_loss"]))
                tr_acc.append(float(row["train_acc"]))
                val_acc.append(float(row["val_acc"]))
                val_f1.append(float(row["val_macro_f1"]))
                lrs.append(float(row["lr"]))

        if not epochs:
            return

        # 1. Loss & Accuracy History Curves Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(epochs, tr_loss, "b-o", label="Train Loss")
        ax1.plot(epochs, val_loss, "r-s", label="Val Loss")
        ax1.set_title("Training & Validation Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend()

        ax2.plot(epochs, tr_acc, "b-o", label="Train Acc")
        ax2.plot(epochs, val_acc, "r-s", label="Val Acc")
        ax2.plot(epochs, val_f1, "g-^", label="Val Macro F1")
        ax2.set_title("Training & Validation Metrics")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Score")
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend()

        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, "loss_acc_curves.png")
        plt.savefig(plot_path, dpi=300)
        plt.close(fig)

        # 2. Learning Rate Schedule Plot
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, lrs, "m-d", label="Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

        plt.tight_layout()
        lr_plot_path = os.path.join(self.plots_dir, "learning_rate_schedule.png")
        plt.savefig(lr_plot_path, dpi=300)
        plt.close(fig)

        logger.debug(f"Updated experiment plots saved under: {self.plots_dir}")

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        filename: str = "confusion_matrix.png",
    ):
        """Generates and saves a confusion matrix heatmap plot."""
        if not MATPLOTLIB_AVAILABLE:
            return

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

        plt.tight_layout()
        cm_path = os.path.join(self.plots_dir, filename)
        plt.savefig(cm_path, dpi=300)
        plt.close(fig)
        logger.info(f"Saved confusion matrix plot to: {cm_path}")

    def plot_roc_curve(
        self,
        fpr: List[float],
        tpr: List[float],
        roc_auc: float,
        filename: str = "roc_curve.png",
    ):
        """Generates and saves ROC curve plot for binary classification mode."""
        if not MATPLOTLIB_AVAILABLE:
            return

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic (ROC)")
        ax.legend(loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        roc_path = os.path.join(self.plots_dir, filename)
        plt.savefig(roc_path, dpi=300)
        plt.close(fig)
        logger.info(f"Saved ROC curve plot to: {roc_path}")

    def close(self):
        """Flushes and closes TensorBoard summary writer."""
        if self.tb_writer:
            self.tb_writer.close()
