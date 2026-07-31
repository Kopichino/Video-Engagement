"""
Configurable Label Mapper Implementation.
Loads taxonomy mappings strictly from external JSON configuration files.
"""

import json
import os
from typing import Dict, Tuple, Optional, Any

from cross_dataset.utils.logger import get_logger

logger = get_logger("LabelMapper")


class ConfigurableLabelMapper:
    """
    Configurable Label Mapper that loads taxonomy definitions strictly from
    external config files without hardcoding labels in source code.

    Supports:
        - Primary Mapping (High, Mid, Low)
        - Binary Mapping (Engaged, Not Engaged)
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: Absolute or relative path to JSON label mappings file.
                         If None, resolves to default package config location.
        """
        if config_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base_dir, "config", "label_mappings.json")

        self.config_path = os.path.abspath(config_path)
        self.primary_mappings: Dict[str, int] = {}
        self.primary_display_names: Dict[int, str] = {}
        self.binary_mappings: Dict[str, int] = {}
        self.binary_display_names: Dict[int, str] = {}

        self.load_config(self.config_path)

    def load_config(self, config_path: str) -> None:
        """
        Parses JSON label configuration file into mapping tables.

        Args:
            config_path: Path to JSON configuration file.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Label mapping configuration file not found at: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        if "primary_3class" not in cfg or "binary_2class" not in cfg:
            raise KeyError(
                "Label configuration JSON must contain 'primary_3class' and 'binary_2class' sections."
            )

        # Parse Primary 3-Class Mapping (High, Mid, Low)
        primary_cfg = cfg["primary_3class"]
        self.primary_mappings = {k.lower(): int(v) for k, v in primary_cfg.get("mappings", {}).items()}
        self.primary_display_names = {int(k): str(v) for k, v in primary_cfg.get("display_names", {}).items()}

        # Parse Binary 2-Class Mapping (Engaged, Not Engaged)
        binary_cfg = cfg["binary_2class"]
        self.binary_mappings = {k.lower(): int(v) for k, v in binary_cfg.get("mappings", {}).items()}
        self.binary_display_names = {int(k): str(v) for k, v in binary_cfg.get("display_names", {}).items()}

        logger.debug(f"Loaded label mappings successfully from {config_path}")

    def map_primary(self, raw_label: str) -> int:
        """
        Maps a raw dataset label to Primary Target (High=2, Mid=1, Low=0).

        Args:
            raw_label: Raw string label (e.g. 'High', 'Disengagement', 'Very engagement').

        Returns:
            int: Primary integer label.
        """
        key = raw_label.strip().lower()
        if key not in self.primary_mappings:
            raise ValueError(
                f"Raw label '{raw_label}' (key: '{key}') is not defined in primary label mapping configuration ({self.config_path})."
            )
        return self.primary_mappings[key]

    def map_binary(self, raw_label: str) -> int:
        """
        Maps a raw dataset label to Binary Target (Engaged=1, Not Engaged=0).

        Args:
            raw_label: Raw string label (e.g. 'High', 'Disengagement', 'Normal').

        Returns:
            int: Binary integer label.
        """
        key = raw_label.strip().lower()
        if key not in self.binary_mappings:
            raise ValueError(
                f"Raw label '{raw_label}' (key: '{key}') is not defined in binary label mapping configuration ({self.config_path})."
            )
        return self.binary_mappings[key]

    def map_all(self, raw_label: str) -> Tuple[int, int]:
        """
        Maps a raw label to both Primary and Binary integer targets.

        Args:
            raw_label: Raw string label.

        Returns:
            Tuple[int, int]: (primary_label, binary_label)
        """
        return self.map_primary(raw_label), self.map_binary(raw_label)

    def get_primary_display_name(self, target_int: int) -> str:
        """
        Returns display string for primary integer label (e.g., 0 -> 'Low', 1 -> 'Mid', 2 -> 'High').
        """
        return self.primary_display_names.get(target_int, f"Unknown_Primary_{target_int}")

    def get_binary_display_name(self, target_int: int) -> str:
        """
        Returns display string for binary integer label (e.g., 0 -> 'Not Engaged', 1 -> 'Engaged').
        """
        return self.binary_display_names.get(target_int, f"Unknown_Binary_{target_int}")
