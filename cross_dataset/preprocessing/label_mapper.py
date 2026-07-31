"""
Label Taxonomy Mapper Wrapper for Preprocessing Pipeline.
Delegates to ConfigurableLabelMapper using external configuration files.
"""

from typing import Tuple, Optional
from cross_dataset.label_mapping.mapper import ConfigurableLabelMapper

# Module-level singleton instance loading mappings from config
_global_mapper: Optional[ConfigurableLabelMapper] = None


def _get_mapper(config_path: Optional[str] = None) -> ConfigurableLabelMapper:
    global _global_mapper
    if _global_mapper is None or config_path is not None:
        _global_mapper = ConfigurableLabelMapper(config_path=config_path)
    return _global_mapper


class LabelMapper:
    """
    Preprocessing wrapper for ConfigurableLabelMapper.
    """

    @classmethod
    def get_labels(cls, raw_label: str, config_path: Optional[str] = None) -> Tuple[int, int]:
        """
        Maps raw string label to primary (High/Mid/Low) and binary (Engaged/Not Engaged) targets.

        Args:
            raw_label: Raw label string.
            config_path: Optional path to JSON label mappings file.

        Returns:
            Tuple[int, int]: (primary_label_3class, binary_label)
        """
        mapper = _get_mapper(config_path)
        return mapper.map_all(raw_label)

    @classmethod
    def get_primary_display_name(cls, label_int: int, config_path: Optional[str] = None) -> str:
        """Returns primary display name (High/Mid/Low) for target integer."""
        mapper = _get_mapper(config_path)
        return mapper.get_primary_display_name(label_int)

    @classmethod
    def get_binary_display_name(cls, label_int: int, config_path: Optional[str] = None) -> str:
        """Returns binary display name (Engaged/Not Engaged) for target integer."""
        mapper = _get_mapper(config_path)
        return mapper.get_binary_display_name(label_int)
