#!/usr/bin/env python3
"""
CLI Direct Evaluation / Transfer Testing Script for Experiment A.

Usage:
    python evaluate.py --config configs/experiment_A/eval_ouc_to_coler.yaml
    python evaluate.py --config configs/experiment_A/eval_coler_to_ouc.yaml
"""

import argparse
import json
import os
import sys

import numpy as np

from cross_dataset.training.checkpoint_manager import CheckpointManager
from cross_dataset.training.config_parser import load_experiment_config
from cross_dataset.training.dataset_factory import build_dataloaders
from cross_dataset.training.evaluator import Evaluator
from cross_dataset.training.logger import ExperimentLogger
from cross_dataset.training.model_factory import build_model
from cross_dataset.utils.logger import get_logger

logger = get_logger("EvalCLI")


def main():
    parser = argparse.ArgumentParser(description="Experiment A Evaluation / Direct Transfer CLI")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML evaluation configuration file (e.g. configs/experiment_A/eval_ouc_to_coler.yaml)",
    )
    args = parser.parse_args()

    logger.info(f"Loading evaluation config: {args.config}")
    config = load_experiment_config(args.config)

    exp_cfg = config["experiment"]
    ds_cfg = config["dataset"]
    model_cfg = config["model"]

    output_dir = os.path.abspath(exp_cfg["output_dir"])
    exp_logger = ExperimentLogger(output_dir=output_dir, experiment_name=exp_cfg["name"])

    # Save Config Snapshot
    snapshot_path = os.path.join(output_dir, "config_snapshot.yaml")
    config.save_snapshot(snapshot_path)

    # Build Model
    model = build_model(model_cfg)

    # Load Trained Checkpoint
    checkpoint_to_load = exp_cfg.get("checkpoint_to_load") or model_cfg.get("checkpoint_path")
    if not checkpoint_to_load or not os.path.exists(checkpoint_to_load):
        logger.warning(
            f"Checkpoint path '{checkpoint_to_load}' not found. "
            f"Attempting to search in default output directory..."
        )
        is_3d = bool(ds_cfg.get("is_video_3d", False) or ds_cfg.get("is_3d", False) or "3d" in exp_cfg.get("name", ""))
        if is_3d:
            if "eval_ouc" in exp_cfg["name"]:
                checkpoint_to_load = "outputs/exp_b_train_ouc_3d/checkpoints/best_model.pth"
            else:
                checkpoint_to_load = "outputs/exp_b_train_coler_3d/checkpoints/best_model.pth"
        else:
            if "eval_ouc" in exp_cfg["name"]:
                checkpoint_to_load = "outputs/exp_a_train_ouc/checkpoints/best_model.pth"
            else:
                checkpoint_to_load = "outputs/exp_a_train_coler/checkpoints/best_model.pth"

    if not os.path.exists(checkpoint_to_load):
        logger.error(
            f"Cannot perform evaluation! Checkpoint file '{checkpoint_to_load}' does not exist. "
            f"Train the model first using train.py"
        )
        sys.exit(1)

    ckpt_mgr = CheckpointManager(checkpoint_dir=exp_logger.checkpoints_dir)
    ckpt_mgr.load_checkpoint(checkpoint_to_load, model)

    # Build DataLoaders (Evaluates on test split of target dataset)
    dataloaders = build_dataloaders(ds_cfg)
    test_loader = dataloaders["test"]

    # Initialize Evaluator
    evaluator = Evaluator(
        model=model,
        use_amp=True,
        num_classes=int(model_cfg.get("num_classes", 3)),
    )

    pred_csv_path = os.path.join(exp_logger.predictions_dir, "transfer_test_predictions.csv")
    logger.info(f"Running direct transfer evaluation on dataset: {ds_cfg.get('target_dataset')}")

    metrics, _ = evaluator.evaluate(test_loader, save_predictions_path=pred_csv_path)

    # Plot Confusion Matrix & ROC Curve
    class_names = ["Low", "Mid", "High"] if evaluator.num_classes == 3 else ["Not Engaged", "Engaged"]
    exp_logger.plot_confusion_matrix(
        cm=np.array(metrics["confusion_matrix"]),
        class_names=class_names,
        filename="transfer_confusion_matrix.png",
    )

    if "fpr" in metrics and "tpr" in metrics and metrics["fpr"] is not None and metrics["tpr"] is not None:
        try:
            exp_logger.plot_roc_curve(
                fpr=metrics["fpr"],
                tpr=metrics["tpr"],
                roc_auc=metrics.get("roc_auc", 0.0),
                filename="transfer_roc_curve.png",
            )
        except Exception as e:
            logger.warning(f"Could not plot ROC curve: {e}")

    # Output JSON Metrics
    eval_metrics_json = os.path.join(exp_logger.metrics_dir, "evaluation_metrics.json")
    with open(eval_metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("==================================================")
    logger.info(" Direct Transfer Evaluation Execution PASSED!")
    logger.info(f" Target Dataset: {ds_cfg.get('target_dataset')}")
    logger.info(f" Accuracy:       {metrics.get('accuracy', 0.0):.4f}")
    logger.info(f" Macro F1:       {metrics.get('macro_f1', 0.0):.4f}")
    logger.info(f" QWK:            {metrics.get('qwk', 0.0):.4f}")
    logger.info(f" Metrics Saved:  {eval_metrics_json}")
    logger.info("==================================================")

    exp_logger.close()


if __name__ == "__main__":
    main()
