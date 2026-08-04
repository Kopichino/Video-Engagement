#!/usr/bin/env python3
"""
CLI Training Script for Experiment A (Frame-Level Transfer).

Supports single-fold training and full 5-fold cross-validation execution.

Usage:
    python train.py --config configs/experiment_A/train_ouc.yaml
    python train.py --config configs/experiment_A/train_ouc.yaml --fold 1
    python train.py --config configs/experiment_A/train_ouc.yaml --all-folds
"""

import argparse
import json
import os
import sys

import numpy as np

from cross_dataset.training.config_parser import load_experiment_config
from cross_dataset.training.trainer import Trainer
from cross_dataset.utils.logger import get_logger

logger = get_logger("TrainCLI")


def run_single_fold(config_path: str, fold: int) -> dict:
    """Loads configuration, overrides target fold, and executes training pipeline."""
    config = load_experiment_config(config_path)

    # Override fold and update experiment name/output directory if non-zero
    config["dataset"]["fold"] = fold
    if fold > 0:
        base_name = config["experiment"]["name"]
        config["experiment"]["name"] = f"{base_name}_fold{fold}"
        config["experiment"]["output_dir"] = f"outputs/{config['experiment']['name']}"

    logger.info(f"Initializing Trainer for experiment: {config['experiment']['name']} (Fold {fold})")
    trainer = Trainer(config)

    logger.info(f"Executing training pipeline for Fold {fold}...")
    metrics = trainer.fit()
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Experiment A Training CLI")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML experiment configuration file (e.g. configs/experiment_A/train_ouc.yaml)",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Target cross-validation fold (0 to 4). Overrides fold in config.",
    )
    parser.add_argument(
        "--all-folds",
        action="store_true",
        help="If set, runs 5-fold cross-validation sequentially across Folds 0, 1, 2, 3, and 4.",
    )
    args = parser.parse_args()

    if args.all_folds:
        logger.info("==================================================")
        logger.info(" STARTING FULL 5-FOLD CROSS-VALIDATION PIPELINE   ")
        logger.info("==================================================")

        fold_results = []
        for f in range(5):
            logger.info(f"\n>>>>>> EXECUTING FOLD {f} / 4 <<<<<<\n")
            res = run_single_fold(args.config, fold=f)
            fold_results.append(res)

        # Aggregate 5-Fold Metrics
        accs = [r.get("accuracy", 0.0) for r in fold_results]
        f1s = [r.get("macro_f1", 0.0) for r in fold_results]
        qwks = [r.get("qwk", 0.0) for r in fold_results]

        logger.info("==================================================")
        logger.info(" 5-FOLD CROSS-VALIDATION FINAL SUMMARY           ")
        logger.info("==================================================")
        logger.info(f" Mean Accuracy: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
        logger.info(f" Mean Macro F1: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
        logger.info(f" Mean QWK:      {np.mean(qwks):.4f} +/- {np.std(qwks):.4f}")
        logger.info("==================================================")

    else:
        target_fold = args.fold if args.fold is not None else 0
        final_metrics = run_single_fold(args.config, fold=target_fold)

        logger.info("==================================================")
        logger.info(f" Experiment Training Execution PASSED! (Fold {target_fold})")
        logger.info(f" Final Test Accuracy:  {final_metrics.get('accuracy', 0.0):.4f}")
        logger.info(f" Final Test Macro F1:  {final_metrics.get('macro_f1', 0.0):.4f}")
        logger.info(f" Final Test QWK:       {final_metrics.get('qwk', 0.0):.4f}")
        logger.info("==================================================")


if __name__ == "__main__":
    main()
