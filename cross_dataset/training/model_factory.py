"""
Model Factory for ResNet-50 Frame-Level Classifier.

Supports:
    - torchvision ResNet-50 backbone
    - ImageNet pretrained vs. scratch initialization
    - Configurable dropout & classification head
    - Fine-grained backbone layer freezing & unfreezing
"""

from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn

try:
    import torchvision.models as models
    from torchvision.models import ResNet50_Weights
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

from cross_dataset.utils.logger import get_logger

logger = get_logger("ModelFactory")


class ResNet50EngagementClassifier(nn.Module):
    """
    ResNet-50 Backbone with configurable dropout and classification head
    for engagement recognition.
    """

    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
        hidden_dim: Optional[int] = 512,
    ):
        super().__init__()
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision must be installed to instantiate ResNet50EngagementClassifier.")

        self.num_classes = num_classes
        self.pretrained = pretrained
        self.dropout_rate = dropout_rate

        # Load ResNet-50 backbone
        if pretrained:
            weights = ResNet50_Weights.DEFAULT
            logger.info("Loading torchvision ResNet-50 with ImageNet pretrained weights...")
        else:
            weights = None
            logger.info("Initializing torchvision ResNet-50 from scratch...")

        self.backbone = models.resnet50(weights=weights)

        # Retrieve backbone feature extraction dimension (2048 for ResNet-50)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove default fc layer

        # Build Configurable Classifier Head
        if hidden_dim and hidden_dim > 0:
            self.classifier = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def freeze_entire_backbone(self):
        """Freezes all parameters in the ResNet-50 feature extraction backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("ResNet-50 backbone frozen.")

    def unfreeze_entire_backbone(self):
        """Unfreezes all parameters in the ResNet-50 feature extraction backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("ResNet-50 backbone unfrozen.")

    def unfreeze_specific_blocks(self, block_names: List[str]):
        """
        Unfreezes specific sub-modules/blocks (e.g. ['layer4', 'layer3']).

        Args:
            block_names: List of module attribute names to unfreeze.
        """
        for name, child in self.backbone.named_children():
            if name in block_names:
                for param in child.parameters():
                    param.requires_grad = True
                logger.info(f"Unfroze backbone layer block: '{name}'")


def build_model(model_config: Dict[str, Any]) -> ResNet50EngagementClassifier:
    """
    Model Factory builder function.

    Args:
        model_config: Model configuration dictionary from experiment config.

    Returns:
        ResNet50EngagementClassifier: Instantiated PyTorch model.
    """
    arch = model_config.get("architecture", "resnet50").lower()
    if arch != "resnet50":
        logger.warning(f"Architecture '{arch}' requested. Defaulting to torchvision resnet50.")

    num_classes = model_config.get("num_classes", 3)
    pretrained = model_config.get("pretrained", True)
    dropout_rate = model_config.get("dropout_rate", 0.5)
    hidden_dim = model_config.get("hidden_dim", 512)

    model = ResNet50EngagementClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        hidden_dim=hidden_dim,
    )

    # Layer Freezing Logic
    freeze_backbone = model_config.get("freeze_backbone", False)
    unfreeze_blocks = model_config.get("unfreeze_blocks", [])

    if freeze_backbone:
        model.freeze_entire_backbone()
        if unfreeze_blocks:
            model.unfreeze_specific_blocks(unfreeze_blocks)

    return model
