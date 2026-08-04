"""
Metrics Calculator for Experiment A.

Computes:
    - Accuracy
    - Macro F1
    - Precision (Macro & Per-Class)
    - Recall (Macro & Per-Class)
    - Confusion Matrix
    - Class Accuracy
    - Balanced Accuracy
    - Quadratic Weighted Kappa (QWK)
    - ROC Curve & AUC (Binary mode only)
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np

try:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        confusion_matrix,
        cohen_kappa_score,
        balanced_accuracy_score,
        roc_curve,
        auc,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from cross_dataset.utils.logger import get_logger

logger = get_logger("MetricsCalculator")


class MetricsCalculator:
    """
    Computes comprehensive classification metrics for engagement recognition evaluation.
    """

    @staticmethod
    def compute_qwk(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3) -> float:
        """
        Computes Quadratic Weighted Kappa (QWK) using Cohen's kappa with quadratic weights.
        """
        if SKLEARN_AVAILABLE:
            try:
                return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
            except Exception:
                pass

        # Fallback NumPy QWK implementation
        cm = np.zeros((num_classes, num_classes), dtype=np.float64)
        for t, p in zip(y_true, y_pred):
            if 0 <= t < num_classes and 0 <= p < num_classes:
                cm[t, p] += 1.0

        total_obs = np.sum(cm)
        if total_obs == 0:
            return 0.0

        hist_true = np.sum(cm, axis=1)
        hist_pred = np.sum(cm, axis=0)

        expected = np.outer(hist_true, hist_pred) / total_obs
        w = np.zeros((num_classes, num_classes), dtype=np.float64)

        for i in range(num_classes):
            for j in range(num_classes):
                w[i, j] = float((i - j) ** 2) / float((num_classes - 1) ** 2)

        numerator = np.sum(w * cm)
        denominator = np.sum(w * expected)

        if denominator == 0:
            return 1.0

        return float(1.0 - (numerator / denominator))

    @classmethod
    def compute_metrics(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        num_classes: int = 3,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Computes all evaluation metrics given true labels, predicted labels, and probabilities.

        Args:
            y_true: Ground truth integer target array [N].
            y_pred: Predicted class integer array [N].
            y_prob: Predicted probability array [N, C] (optional, for ROC AUC).
            num_classes: Total classes (3 for primary, 2 for binary).
            class_names: Optional list of human-readable class names.

        Returns:
            Dict[str, Any]: Dictionary containing all metric scores and confusion matrix.
        """
        y_true = np.asarray(y_true, dtype=np.int64)
        y_pred = np.asarray(y_pred, dtype=np.int64)

        if len(y_true) == 0 or len(y_pred) == 0:
            return {
                "accuracy": 0.0,
                "macro_f1": 0.0,
                "macro_precision": 0.0,
                "macro_recall": 0.0,
                "qwk": 0.0,
                "balanced_accuracy": 0.0,
                "confusion_matrix": np.zeros((num_classes, num_classes), dtype=int).tolist(),
            }

        if class_names is None:
            if num_classes == 3:
                class_names = ["Low", "Mid", "High"]
            elif num_classes == 2:
                class_names = ["Not Engaged", "Engaged"]
            else:
                class_names = [f"Class_{i}" for i in range(num_classes)]

        # Accuracy
        acc = float(np.mean(y_true == y_pred))

        # Confusion Matrix
        if SKLEARN_AVAILABLE:
            cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
            macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
            macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
            macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
            balanced_acc = float(balanced_accuracy_score(y_true, y_pred))

            # Per-class metrics
            per_class_f1 = f1_score(y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0).tolist()
            per_class_prec = precision_score(y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0).tolist()
            per_class_rec = recall_score(y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0).tolist()
        else:
            # Fallback manual calculation
            cm = np.zeros((num_classes, num_classes), dtype=int)
            for t, p in zip(y_true, y_pred):
                if 0 <= t < num_classes and 0 <= p < num_classes:
                    cm[t, p] += 1

            per_class_rec = []
            per_class_prec = []
            per_class_f1 = []

            for c in range(num_classes):
                tp = cm[c, c]
                fn = np.sum(cm[c, :]) - tp
                fp = np.sum(cm[:, c]) - tp

                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

                per_class_rec.append(float(rec))
                per_class_prec.append(float(prec))
                per_class_f1.append(float(f1))

            macro_f1 = float(np.mean(per_class_f1))
            macro_prec = float(np.mean(per_class_prec))
            macro_rec = float(np.mean(per_class_rec))
            balanced_acc = macro_rec

        # Class Accuracy (per-class diagonal accuracy)
        class_acc = {}
        for i, c_name in enumerate(class_names):
            row_sum = np.sum(cm[i, :])
            c_acc = float(cm[i, i] / row_sum) if row_sum > 0 else 0.0
            class_acc[c_name] = round(c_acc, 4)

        # Quadratic Weighted Kappa
        qwk = cls.compute_qwk(y_true, y_pred, num_classes=num_classes)

        metrics_result = {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "macro_precision": round(macro_prec, 4),
            "macro_recall": round(macro_rec, 4),
            "balanced_accuracy": round(balanced_acc, 4),
            "qwk": round(qwk, 4),
            "confusion_matrix": cm.tolist(),
            "class_accuracy": class_acc,
            "per_class_f1": [round(val, 4) for val in per_class_f1],
            "per_class_precision": [round(val, 4) for val in per_class_prec],
            "per_class_recall": [round(val, 4) for val in per_class_rec],
        }

        # ROC Curve and AUC for Binary Classification Mode
        if num_classes == 2 and y_prob is not None and len(y_prob) == len(y_true):
            try:
                probs_pos = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                if SKLEARN_AVAILABLE:
                    fpr, tpr, thresholds = roc_curve(y_true, probs_pos)
                    roc_auc = float(auc(fpr, tpr))
                    metrics_result["roc_auc"] = round(roc_auc, 4)
                    metrics_result["fpr"] = fpr.tolist()
                    metrics_result["tpr"] = tpr.tolist()
            except Exception as e:
                logger.debug(f"ROC AUC calculation skipped: {e}")

        return metrics_result
