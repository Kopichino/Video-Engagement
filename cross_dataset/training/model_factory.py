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


class MixStyle(nn.Module):
    """
    MixStyle module for Domain Generalization (Zhou et al., ICLR 2021).
    Mixes feature statistics (channel-wise mean and variance) of randomly paired samples
    within a mini-batch to synthesize novel domain styles during training.
    Supports both 5D spatiotemporal tensors [B, C, T, H, W] and 4D spatial tensors [B, C, H, W].
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.3, eps: float = 1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x

        if torch.rand(1).item() > self.p:
            return x

        B = x.size(0)
        if B <= 1:
            return x

        reduction_dims = list(range(2, x.dim()))
        mu = x.mean(dim=reduction_dims, keepdim=True)
        var = x.var(dim=reduction_dims, keepdim=True, unbiased=False)
        sig = (var + self.eps).sqrt()

        x_norm = (x - mu) / sig

        perm = torch.randperm(B)
        lmda = torch.distributions.Beta(self.alpha, self.alpha).sample().to(x.device)

        mu_mix = mu * lmda + mu[perm] * (1.0 - lmda)
        sig_mix = sig * lmda + sig[perm] * (1.0 - lmda)

        return x_norm * sig_mix + mu_mix


class R2Plus1DEngagementClassifier(nn.Module):
    """
    R(2+1)D-18 3D Spatiotemporal Video Backbone for 16-Frame Sequence Engagement Recognition (Experiment B).
    Optionally supports MixStyle domain generalization.
    """

    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
        hidden_dim: Optional[int] = 512,
        use_mixstyle: bool = False,
        mixstyle_p: float = 0.5,
        mixstyle_alpha: float = 0.3,
    ):
        super().__init__()
        if not TORCHVISION_VIDEO_AVAILABLE:
            raise ImportError("torchvision.models.video must be available to instantiate R2Plus1DEngagementClassifier.")

        self.num_classes = num_classes
        self.pretrained = pretrained
        self.dropout_rate = dropout_rate
        self.use_mixstyle = use_mixstyle

        if self.use_mixstyle:
            self.mixstyle = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha)
            logger.info(f"MixStyle enabled in R(2+1)D-18 (p={mixstyle_p}, alpha={mixstyle_alpha})")

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
        if self.use_mixstyle:
            x = self.backbone.stem(x)
            x = self.backbone.layer1(x)
            x = self.mixstyle(x)
            x = self.backbone.layer2(x)
            x = self.mixstyle(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            x = self.backbone.avgpool(x)
            features = torch.flatten(x, 1)
        else:
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
            use_mixstyle=bool(model_config.get("use_mixstyle", False)),
            mixstyle_p=float(model_config.get("mixstyle_p", 0.5)),
            mixstyle_alpha=float(model_config.get("mixstyle_alpha", 0.3)),
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
