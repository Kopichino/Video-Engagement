"""
Model Factory for 2D Spatial (ResNet-50) and 3D Spatiotemporal (R(2+1)D-18) Classifiers.

Supports:
    - 2D torchvision ResNet-50 backbone (Experiment A)
    - 3D torchvision R(2+1)D-18 video backbone (Experiment B)
    - ImageNet / Kinetics-400 pretrained vs scratch initialization
    - Configurable dropout & classification head
    - Fine-grained backbone layer freezing & unfreezing
"""

from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn

try:
    import torchvision.models as models
    from torchvision.models import ResNet50_Weights
    try:
        from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights, r3d_18, R3D_18_Weights
        TORCHVISION_VIDEO_AVAILABLE = True
    except ImportError:
        TORCHVISION_VIDEO_AVAILABLE = False
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    TORCHVISION_VIDEO_AVAILABLE = False

from cross_dataset.utils.logger import get_logger

logger = get_logger("ModelFactory")


class ResNet50EngagementClassifier(nn.Module):
    """
    ResNet-50 2D Spatial Backbone for Frame-Level Engagement Recognition (Experiment A).
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

        if pretrained:
            weights = ResNet50_Weights.DEFAULT
            logger.info("Loading torchvision ResNet-50 with ImageNet pretrained weights...")
        else:
            weights = None
            logger.info("Initializing torchvision ResNet-50 from scratch...")

        self.backbone = models.resnet50(weights=weights)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

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
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("ResNet-50 backbone frozen.")

    def unfreeze_entire_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("ResNet-50 backbone unfrozen.")

    def unfreeze_specific_blocks(self, block_names: List[str]):
        for name, child in self.backbone.named_children():
            if name in block_names:
                for param in child.parameters():
                    param.requires_grad = True
                logger.info(f"Unfroze backbone layer block: '{name}'")


class R2Plus1DEngagementClassifier(nn.Module):
    """
    R(2+1)D-18 3D Spatiotemporal Video Backbone for 16-Frame Sequence Engagement Recognition (Experiment B).
    """

    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
        hidden_dim: Optional[int] = 512,
    ):
        super().__init__()
        if not TORCHVISION_VIDEO_AVAILABLE:
            raise ImportError("torchvision.models.video must be available to instantiate R2Plus1DEngagementClassifier.")

        self.num_classes = num_classes
        self.pretrained = pretrained
        self.dropout_rate = dropout_rate

        if pretrained:
            weights = R2Plus1D_18_Weights.DEFAULT
            logger.info("Loading torchvision R(2+1)D-18 with Kinetics-400 pretrained weights...")
        else:
            weights = None
            logger.info("Initializing torchvision R(2+1)D-18 from scratch...")

        self.backbone = r2plus1d_18(weights=weights)

        # Retrieve feature dimension (512 for R(2+1)D-18)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

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
        # Input shape: [B, C, T, H, W]
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def freeze_entire_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("R(2+1)D-18 3D backbone frozen.")

    def unfreeze_entire_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("R(2+1)D-18 3D backbone unfrozen.")


def build_model(model_config: Dict[str, Any]) -> nn.Module:
    """
    Model Factory builder function. Supports 2D ResNet-50 and 3D R(2+1)D-18.

    Args:
        model_config: Model configuration dictionary from experiment config.

    Returns:
        nn.Module: Instantiated PyTorch model.
    """
    arch = model_config.get("architecture", "resnet50").lower()
    num_classes = model_config.get("num_classes", 3)
    pretrained = model_config.get("pretrained", True)
    dropout_rate = model_config.get("dropout_rate", 0.5)
    hidden_dim = model_config.get("hidden_dim", 512)

    if arch in ["r2plus1d_18", "r2plus1d", "resnet3d_18", "r3d_18"]:
        logger.info(f"Building 3D Video Backbone: '{arch}'")
        model = R2Plus1DEngagementClassifier(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            hidden_dim=hidden_dim,
        )
    else:
        logger.info(f"Building 2D Spatial Backbone: '{arch}'")
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
        if hasattr(model, "freeze_entire_backbone"):
            model.freeze_entire_backbone()
        if unfreeze_blocks and hasattr(model, "unfreeze_specific_blocks"):
            model.unfreeze_specific_blocks(unfreeze_blocks)

    return model
