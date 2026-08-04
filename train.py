#!/usr/bin/env python3
"""
CLI Training Script for Experiment A (Frame-Level Transfer).

Usage:
    python train.py --config configs/experiment_A/train_ouc.yaml
    python train.py --config configs/experiment_A/train_coler.yaml
"""

import argparse
import sys

from cross_dataset.training.config_parser import load_experiment_config
from cross_dataset.training.trainer import Trainer
from cross_dataset.utils.logger import get_logger

logger = get_logger("TrainCLI")


def main():
    parser = argparse.ArgumentParser(description="Experiment A Training CLI")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML experiment configuration file (e.g. configs/experiment_A/train_ouc.yaml)",
    )
    args = parser.parse_args()

    logger.info(f"Loading experiment config: {args.config}")
    config = load_experiment_config(args.config)

    logger.info(f"Initializing Trainer for experiment: {config['experiment']['name']}")
    trainer = Trainer(config)

    logger.info("Executing training pipeline...")
    final_metrics = trainer.fit()

    logger.info("==================================================")
    logger.info(" Experiment Training Execution PASSED!")
    logger.info(f" Final Test Accuracy:  {final_metrics.get('accuracy', 0.0):.4f}")
    logger.info(f" Final Test Macro F1:  {final_metrics.get('macro_f1', 0.0):.4f}")
    logger.info(f" Final Test QWK:       {final_metrics.get('qwk', 0.0):.4f}")
    logger.info("==================================================")


if __name__ == "__main__":
    main()
