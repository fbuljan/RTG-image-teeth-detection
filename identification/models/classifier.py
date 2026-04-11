"""
Tooth classification model.

ResNet-18 backbone with configurable classification head.
Backbone/head separation enables reuse in Phase 3 (swap head for embedding projection).
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ToothClassifier(nn.Module):
    """
    ResNet-18 based classifier for tooth crops.

    Args:
        num_classes: Number of output classes.
        pretrained: Use ImageNet pretrained weights.
        dropout: Dropout rate before the classification head.
    """

    def __init__(self, num_classes: int, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        self.feature_dim = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract backbone features without classification head."""
        return self.backbone(x)
