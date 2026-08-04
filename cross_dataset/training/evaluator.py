"""
Evaluator Module for Experiment A Framework.

Performs single-pass evaluation over validation, testing, and direct cross-dataset
transfer splits without gradient computation.
"""

import csv
import json
import os
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cross_dataset.training.metrics import MetricsCalculator
from cross_dataset.utils.logger import get_logger

logger = get_logger("Evaluator")


class Evaluator:
    """
    Evaluates a trained neural network model on a given dataset split.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        use_amp: bool = True,
        num_classes: int = 3,
    ):
        """
        Args:
            model: Trained PyTorch model.
            criterion: Loss function (defaults to nn.CrossEntropyLoss).
            device: Target torch device (CPU or CUDA).
            use_amp: Whether to use Automatic Mixed Precision.
            num_classes: Number of target classes.
        """
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model = model.to(self.device)
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.use_amp = use_amp and (self.device.type == "cuda")
        self.num_classes = num_classes

    def _extract_info_item(self, container: Any, index: int) -> Any:
        """Safely extracts item at index from Tensor, list, tuple, or scalar."""
        if isinstance(container, torch.Tensor):
            return container[index].item()
        elif isinstance(container, (list, tuple)):
            return container[index]
        else:
            return container

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        save_predictions_path: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Runs single-pass evaluation over the dataloader.

        Args:
            dataloader: Target PyTorch DataLoader.
            save_predictions_path: Optional path to save detailed per-sample predictions CSV.

        Returns:
            Tuple[Dict[str, Any], List[Dict[str, Any]]]:
                - Computed evaluation metrics dictionary.
                - List of prediction record dictionaries.
        """
        self.model.eval()

        running_loss = 0.0
        total_samples = 0

        y_true_list: List[int] = []
        y_pred_list: List[int] = []
        y_prob_list: List[np.ndarray] = []
        prediction_records: List[Dict[str, Any]] = []

        for batch_idx, (images, targets, info) in enumerate(dataloader):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    logits = self.model(images)
                    loss = self.criterion(logits, targets)
            else:
                logits = self.model(images)
                loss = self.criterion(logits, targets)

            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            probs_np = probs.cpu().numpy()
            preds_np = preds.cpu().numpy()
            targets_np = targets.cpu().numpy()

            y_true_list.extend(targets_np)
            y_pred_list.extend(preds_np)
            y_prob_list.append(probs_np)

            # Record per-sample prediction details
            for i in range(batch_size):
                rec = {
                    "dataset": str(self._extract_info_item(info.get("dataset"), i)),
                    "clip_id": str(self._extract_info_item(info.get("clip_id"), i)),
                    "frame_id": int(self._extract_info_item(info.get("frame_id"), i)),
                    "image_path": str(self._extract_info_item(info.get("image_path"), i)),
                    "target": int(targets_np[i]),
                    "prediction": int(preds_np[i]),
                    "probabilities": probs_np[i].tolist(),
                }
                prediction_records.append(rec)

        avg_loss = (running_loss / total_samples) if total_samples > 0 else 0.0

        y_true_arr = np.array(y_true_list, dtype=np.int64)
        y_pred_arr = np.array(y_pred_list, dtype=np.int64)
        y_prob_arr = np.vstack(y_prob_list) if y_prob_list else None

        # Compute full metrics using MetricsCalculator
        metrics = MetricsCalculator.compute_metrics(
            y_true=y_true_arr,
            y_pred=y_pred_arr,
            y_prob=y_prob_arr,
            num_classes=self.num_classes,
        )
        metrics["loss"] = round(avg_loss, 4)

        # Optionally save predictions to CSV
        if save_predictions_path and prediction_records:
            self._save_predictions(prediction_records, save_predictions_path)

        return metrics, prediction_records

    def _save_predictions(
        self, records: List[Dict[str, Any]], output_path: str
    ) -> None:
        """Saves detailed prediction records to CSV."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fieldnames = ["dataset", "clip_id", "frame_id", "image_path", "target", "prediction", "probabilities"]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                row_dict = dict(r)
                row_dict["probabilities"] = json.dumps(row_dict["probabilities"])
                writer.writerow(row_dict)

        logger.info(f"Saved evaluation prediction records to: {output_path}")
