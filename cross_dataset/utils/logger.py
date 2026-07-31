"""
Reusable Logging Utility for Cross-Dataset Pipeline.
"""

import logging
import os
import sys
from typing import Optional


def get_logger(
    name: str = "cross_dataset",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Creates or retrieves a structured logger instance.

    Args:
        name: Name of the logger.
        log_file: Optional file path to write log messages to.
        level: Logging level (default: logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if logger is already configured
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    # File Handler (Optional)
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    return logger
