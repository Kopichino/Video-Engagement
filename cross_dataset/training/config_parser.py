"""
Configuration Parser for Experiment A Training Framework.
Loads, validates, and serializes YAML experiment configurations.
"""

import os
from typing import Dict, Any, Optional

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from cross_dataset.utils.logger import get_logger

logger = get_logger("ConfigParser")


class ExperimentConfig:
    """
    Wrapper for experiment configuration dictionary providing dot-notation or dict access.
    """

    def __init__(self, raw_dict: Dict[str, Any]):
        self._config = raw_dict
        self._validate_and_apply_defaults()

    def _validate_and_apply_defaults(self):
        """Validates mandatory configuration fields and sets sensible fallback defaults."""
        if "experiment" not in self._config:
            raise KeyError("YAML configuration missing mandatory 'experiment' section.")
        if "dataset" not in self._config:
            raise KeyError("YAML configuration missing mandatory 'dataset' section.")
        if "model" not in self._config:
            raise KeyError("YAML configuration missing mandatory 'model' section.")

        exp = self._config["experiment"]
        exp.setdefault("name", "unnamed_experiment")
        exp.setdefault("output_dir", os.path.join("outputs", exp["name"]))
        exp.setdefault("seed", 42)

        ds = self._config["dataset"]
        ds.setdefault("manifest_path", "data/manifests/unified_frame_dataset.csv")
        ds.setdefault("label_type", "primary")
        ds.setdefault("fold", 0)
        ds.setdefault("balance_classes", True)
        ds.setdefault("batch_size", 32)
        ds.setdefault("num_workers", 0)
        ds.setdefault("pin_memory", True)

        model = self._config["model"]
        model.setdefault("architecture", "resnet50")
        model.setdefault("pretrained", True)
        model.setdefault("num_classes", 3)
        model.setdefault("dropout_rate", 0.5)
        model.setdefault("freeze_backbone", False)
        model.setdefault("unfreeze_blocks", [])

        # Default Training Section
        train = self._config.setdefault("training", {})
        train.setdefault("epochs", 20)
        train.setdefault("use_amp", True)
        train.setdefault("gradient_clipping", 1.0)
        train.setdefault("gradient_accumulation_steps", 1)
        train.setdefault("auto_resume", True)
        
        es = train.setdefault("early_stopping", {})
        es.setdefault("enabled", True)
        es.setdefault("patience", 5)
        es.setdefault("min_delta", 0.001)
        es.setdefault("monitor", "val_macro_f1")
        es.setdefault("mode", "max")

        # Default Optimizer Section
        opt = self._config.setdefault("optimizer", {})
        opt.setdefault("type", "adamw")
        opt.setdefault("lr", 0.0001)
        opt.setdefault("weight_decay", 0.01)
        opt.setdefault("momentum", 0.9)

        # Default Scheduler Section
        sched = self._config.setdefault("scheduler", {})
        sched.setdefault("type", "cosine")
        sched.setdefault("t_max", train["epochs"])

        # Default Evaluation Section
        eval_sec = self._config.setdefault("evaluation", {})
        eval_sec.setdefault("eval_batch_size", ds["batch_size"])

    @property
    def raw_dict(self) -> Dict[str, Any]:
        return self._config

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def __getitem__(self, item: str) -> Any:
        return self._config[item]

    def save_snapshot(self, output_path: str):
        """Saves a snapshot of the current configuration to YAML file."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        if YAML_AVAILABLE:
            with open(output_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(self._config, f, default_flow_style=False, sort_keys=False)
        else:
            logger.warning("PyYAML not installed; dumping json snapshot fallback.")
            import json
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(self._config, f, indent=2)


def load_experiment_config(config_path: str) -> ExperimentConfig:
    """
    Loads YAML experiment configuration file and returns an ExperimentConfig object.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        ExperimentConfig: Validated experiment configuration object.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    if not YAML_AVAILABLE:
        raise ImportError("PyYAML package is required to parse YAML configs. Install with 'pip install pyyaml'.")

    with open(config_path, "r", encoding="utf-8") as f:
        raw_dict = yaml.safe_load(f)

    if not isinstance(raw_dict, dict):
        raise ValueError(f"Invalid YAML structure in {config_path}. Expected dictionary at root.")

    logger.info(f"Loaded experiment configuration from: {config_path}")
    return ExperimentConfig(raw_dict)
