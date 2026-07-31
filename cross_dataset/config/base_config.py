"""
Base Configuration Loader for Cross-Dataset Pipeline.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class DatasetPathsConfig:
    """Paths configuration for OUC-CGE and COLER datasets."""
    root_dir: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ouc_cge_root: str = field(init=False)
    coler_root: str = field(init=False)
    manifest_dir: str = field(init=False)
    manifest_path: str = field(init=False)
    label_mappings_config: str = field(init=False)

    def __post_init__(self):
        self.ouc_cge_root = os.path.join(self.root_dir, "OUC-CGE-dataset")
        self.coler_root = os.path.join(self.root_dir, "Ver4-20260630T020728Z-3-005", "Ver4")
        self.manifest_dir = os.path.join(self.root_dir, "data", "manifests")
        self.manifest_path = os.path.join(self.manifest_dir, "unified_manifest.csv")
        self.label_mappings_config = os.path.join(self.root_dir, "data", "configs", "label_mappings.json")


@dataclass
class PreprocessingConfig:
    """Settings for dataset parsing, validation, and splitting."""
    num_folds: int = 5
    random_seed: int = 37247
    validate_videos: bool = True
    min_video_frames: int = 1
    allowed_video_extensions: tuple = (".mp4", ".avi", ".mov")
    allowed_image_extensions: tuple = (".png", ".jpg", ".jpeg")


@dataclass
class PipelineConfig:
    """Master Pipeline Configuration."""
    paths: DatasetPathsConfig = field(default_factory=DatasetPathsConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)

    def save_to_json(self, json_path: str):
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
        data = {
            "paths": {
                "root_dir": self.paths.root_dir,
                "ouc_cge_root": self.paths.ouc_cge_root,
                "coler_root": self.paths.coler_root,
                "manifest_path": self.paths.manifest_path,
                "label_mappings_config": self.paths.label_mappings_config,
            },
            "preprocessing": {
                "num_folds": self.preprocessing.num_folds,
                "random_seed": self.preprocessing.random_seed,
                "validate_videos": self.preprocessing.validate_videos,
            }
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_json(cls, json_path: str) -> "PipelineConfig":
        """Load configuration from JSON file."""
        config = cls()
        if not os.path.exists(json_path):
            return config

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "preprocessing" in data:
            prep = data["preprocessing"]
            config.preprocessing.num_folds = prep.get("num_folds", config.preprocessing.num_folds)
            config.preprocessing.random_seed = prep.get("random_seed", config.preprocessing.random_seed)
            config.preprocessing.validate_videos = prep.get("validate_videos", config.preprocessing.validate_videos)
        
        return config


def load_config(config_path: Optional[str] = None) -> PipelineConfig:
    """
    Factory function to load PipelineConfig.
    """
    if config_path and os.path.exists(config_path):
        return PipelineConfig.load_from_json(config_path)
    return PipelineConfig()
