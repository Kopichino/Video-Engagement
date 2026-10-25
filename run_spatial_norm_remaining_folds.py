#!/usr/bin/env python3
"""
Automated 5-Fold Pipeline for Experiment DG Phase 2:
Spatial Bounding Box Normalization + MixStyle + Domain Randomization

Executes Folds 1, 2, 3, and 4 on OUC-CGE, evaluates each fold zero-shot on COLER,
and aggregates full 5-fold cross-validation metrics across all 5 folds.
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import torch

from cross_dataset.training.checkpoint_manager import CheckpointManager
from cross_dataset.training.config_parser import load_experiment_config
from cross_dataset.training.dataset_factory import build_dataloaders
from cross_dataset.training.evaluator import Evaluator
from cross_dataset.training.logger import ExperimentLogger
from cross_dataset.training.metrics import MetricsCalculator
from cross_dataset.training.model_factory import build_model
from cross_dataset.training.trainer import Trainer
from cross_dataset.utils.logger import get_logger

logger = get_logger("SpatialNorm5FoldPipeline")


def train_fold(config_path: str, fold: int) -> dict:
    """Trains a single fold on OUC-CGE with spatial bounding box normalization."""
    config = load_experiment_config(config_path)
    config["dataset"]["fold"] = fold
    base_name = config["experiment"]["name"]
    config["experiment"]["name"] = f"{base_name}_fold{fold}"
    config["experiment"]["output_dir"] = f"outputs/{config['experiment']['name']}"

    logger.info(f"==================================================")
    logger.info(f" STARTING TRAINING: Fold {fold} on OUC-CGE")
    logger.info(f" Output Dir: {config['experiment']['output_dir']}")
    logger.info(f"==================================================")

    trainer = Trainer(config)
    metrics = trainer.fit()
    return metrics


def evaluate_on_coler(checkpoint_path: str, output_dir: str, fold: int) -> dict:
    """Evaluates the trained spatial-norm model zero-shot on COLER test set."""
    logger.info(f"==================================================")
    logger.info(f" EVALUATING FOLD {fold} ZERO-SHOT ON COLER")
    logger.info(f" Checkpoint: {checkpoint_path}")
    logger.info(f" Output Dir: {output_dir}")
    logger.info(f"==================================================")

    os.makedirs(output_dir, exist_ok=True)
    exp_logger = ExperimentLogger(output_dir=output_dir, experiment_name=f"eval_ouc_to_coler_fold{fold}")

    model_cfg = {
        "architecture": "r2plus1d_18",
        "pretrained": False,
        "num_classes": 3,
        "use_mixstyle": False,
    }
    model = build_model(model_cfg)

    ckpt_mgr = CheckpointManager(checkpoint_dir=exp_logger.checkpoints_dir)
    ckpt_mgr.load_checkpoint(checkpoint_path, model)

    ds_cfg = {
        "manifest_path": "data/manifests/unified_manifest.csv",
        "target_dataset": "COLER",
        "eval_target_dataset": "COLER",
        "label_type": "primary",
        "is_video_3d": True,
        "num_frames": 16,
        "img_size": 224,
        "fold": 0,
        "balance_classes": False,
        "augment": False,
        "spatial_norm": False,
        "batch_size": 8,
        "num_workers": 0,
        "pin_memory": False,
        "use_ram_cache": False,
    }
    dataloaders = build_dataloaders(ds_cfg)
    test_loader = dataloaders["test"]

    evaluator = Evaluator(model=model, use_amp=True, num_classes=3)
    pred_csv_path = os.path.join(exp_logger.predictions_dir, "transfer_test_predictions.csv")

    metrics, _ = evaluator.evaluate(test_loader, save_predictions_path=pred_csv_path)

    # Plot Confusion Matrix
    exp_logger.plot_confusion_matrix(
        cm=np.array(metrics["confusion_matrix"]),
        class_names=["Low", "Mid", "High"],
        filename="transfer_confusion_matrix.png",
    )

    eval_metrics_json = os.path.join(exp_logger.metrics_dir, "evaluation_metrics.json")
    with open(eval_metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Fold {fold} COLER Test Acc: {metrics.get('accuracy', 0.0):.4f} | Macro F1: {metrics.get('macro_f1', 0.0):.4f} | QWK: {metrics.get('qwk', 0.0):.4f}")
    exp_logger.close()
    return metrics


def aggregate_and_report():
    """Aggregates all 5 folds for both In-Domain (OUC) and Cross-Domain (COLER)."""
    logger.info("==================================================")
    logger.info(" AGGREGATING FULL 5-FOLD RESULTS")
    logger.info("==================================================")

    # 1. In-Domain OUC
    in_domain_results = []
    for f in range(5):
        if f == 0:
            pred_path = "outputs/exp_dg_train_ouc_spatial_norm/predictions/test_predictions.csv"
        else:
            pred_path = f"outputs/exp_dg_train_ouc_spatial_norm_fold{f}/predictions/test_predictions.csv"

        if os.path.exists(pred_path):
            df = pd.read_csv(pred_path)
            y_true = df["ground_truth"].values if "ground_truth" in df.columns else df["target"].values
            y_pred = df["predicted_label"].values if "predicted_label" in df.columns else df["prediction"].values
            m = MetricsCalculator.compute_metrics(y_true, y_pred, num_classes=3)
            in_domain_results.append({
                "fold": f,
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "qwk": m["qwk"],
                "macro_precision": m.get("macro_precision", 0.0),
                "macro_recall": m.get("macro_recall", 0.0),
                "confusion_matrix": m["confusion_matrix"],
            })

    # 2. Cross-Domain COLER
    cross_domain_results = []
    for f in range(5):
        if f == 0:
            pred_path = "outputs/exp_dg_eval_ouc_to_coler_spatial_norm/predictions/transfer_test_predictions.csv"
        else:
            pred_path = f"outputs/exp_dg_eval_ouc_to_coler_spatial_norm_fold{f}/predictions/transfer_test_predictions.csv"

        if os.path.exists(pred_path):
            df = pd.read_csv(pred_path)
            y_true = df["ground_truth"].values if "ground_truth" in df.columns else df["target"].values
            y_pred = df["predicted_label"].values if "predicted_label" in df.columns else df["prediction"].values
            m = MetricsCalculator.compute_metrics(y_true, y_pred, num_classes=3)
            cross_domain_results.append({
                "fold": f,
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "qwk": m["qwk"],
                "macro_precision": m.get("macro_precision", 0.0),
                "macro_recall": m.get("macro_recall", 0.0),
                "confusion_matrix": m["confusion_matrix"],
            })

    summary = {
        "in_domain_ouc": {
            "folds": in_domain_results,
            "mean_accuracy": float(np.mean([r["accuracy"] for r in in_domain_results])),
            "std_accuracy": float(np.std([r["accuracy"] for r in in_domain_results])),
            "mean_macro_f1": float(np.mean([r["macro_f1"] for r in in_domain_results])),
            "std_macro_f1": float(np.std([r["macro_f1"] for r in in_domain_results])),
            "mean_qwk": float(np.mean([r["qwk"] for r in in_domain_results])),
            "std_qwk": float(np.std([r["qwk"] for r in in_domain_results])),
        },
        "cross_domain_coler": {
            "folds": cross_domain_results,
            "mean_accuracy": float(np.mean([r["accuracy"] for r in cross_domain_results])),
            "std_accuracy": float(np.std([r["accuracy"] for r in cross_domain_results])),
            "mean_macro_f1": float(np.mean([r["macro_f1"] for r in cross_domain_results])),
            "std_macro_f1": float(np.std([r["macro_f1"] for r in cross_domain_results])),
            "mean_qwk": float(np.mean([r["qwk"] for r in cross_domain_results])),
            "std_qwk": float(np.std([r["qwk"] for r in cross_domain_results])),
        }
    }

    summary_path = "outputs/exp_dg_spatial_norm_5fold_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("==================================================")
    logger.info(" 5-FOLD CV SPATIAL NORMALIZATION FINAL SUMMARY")
    logger.info("==================================================")
    logger.info(f" In-Domain OUC Acc:      {summary['in_domain_ouc']['mean_accuracy']:.4f} +/- {summary['in_domain_ouc']['std_accuracy']:.4f}")
    logger.info(f" In-Domain OUC F1:       {summary['in_domain_ouc']['mean_macro_f1']:.4f} +/- {summary['in_domain_ouc']['std_macro_f1']:.4f}")
    logger.info(f" Cross-Domain COLER Acc: {summary['cross_domain_coler']['mean_accuracy']:.4f} +/- {summary['cross_domain_coler']['std_accuracy']:.4f}")
    logger.info(f" Cross-Domain COLER F1:  {summary['cross_domain_coler']['mean_macro_f1']:.4f} +/- {summary['cross_domain_coler']['std_macro_f1']:.4f}")
    logger.info(f" Cross-Domain COLER QWK: {summary['cross_domain_coler']['mean_qwk']:.4f} +/- {summary['cross_domain_coler']['std_qwk']:.4f}")
    logger.info(f" Summary JSON saved to:  {summary_path}")
    logger.info("==================================================")
    return summary


def main():
    config_path = "configs/experiment_DG/train_ouc_spatial_norm_5fold.yaml"
    
    for fold in [1, 2, 3, 4]:
        ckpt_path = f"outputs/exp_dg_train_ouc_spatial_norm_fold{fold}/checkpoints/best_model.pth"
        if not os.path.exists(ckpt_path):
            train_fold(config_path, fold)
        else:
            logger.info(f"Fold {fold} training already exists at {ckpt_path}. Skipping training.")

        eval_dir = f"outputs/exp_dg_eval_ouc_to_coler_spatial_norm_fold{fold}"
        eval_metrics = os.path.join(eval_dir, "metrics", "evaluation_metrics.json")
        if not os.path.exists(eval_metrics):
            evaluate_on_coler(ckpt_path, eval_dir, fold)
        else:
            logger.info(f"Fold {fold} COLER evaluation already exists at {eval_metrics}. Skipping evaluation.")

    aggregate_and_report()


if __name__ == "__main__":
    main()
