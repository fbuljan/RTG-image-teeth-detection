"""
Tooth embedding model for metric learning.

ResNet-18 backbone with projection head producing L2-normalized embeddings.
Reuses backbone pattern from classifier.py. Backbone/head separation
enables Phase 4 metadata fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ToothEmbeddingModel(nn.Module):
    """
    ResNet-18 based embedding model for tooth crops.

    Args:
        embedding_dim: Output embedding dimension.
        pretrained: Use ImageNet pretrained weights.
        dropout: Dropout rate before projection head.
        num_fdi_classes: If set, adds a parallel FDI classification head fed by
            the same post-dropout 512-d features (Phase 8.4 multi-task aux loss).
            When set, forward() returns (embeddings, fdi_logits). Backwards
            compatible: default None preserves the single-output forward.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        pretrained: bool = True,
        dropout: float = 0.2,
        num_fdi_classes: int | None = None,
    ):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        self.feature_dim = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Identity()

        self.dropout = nn.Dropout(dropout)
        self.projection_head = nn.Linear(self.feature_dim, embedding_dim)
        # Phase 8.4: optional FDI auxiliary classification head. Fed POST-dropout
        # to share stochastic feature subspace with the projection head (forces
        # joint regularisation; design-review recommendation).
        self.fdi_head: nn.Linear | None = (
            nn.Linear(self.feature_dim, num_fdi_classes)
            if num_fdi_classes is not None
            else None
        )

    def forward(self, x: torch.Tensor):
        """Returns L2-normalized embeddings of shape (B, embedding_dim).

        If fdi_head is enabled, returns (embeddings, fdi_logits).
        """
        features = self.backbone(x)
        f_drop = self.dropout(features)
        embeddings = self.projection_head(f_drop)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        if self.fdi_head is not None:
            fdi_logits = self.fdi_head(f_drop)
            return embeddings, fdi_logits
        return embeddings

    def get_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract raw backbone features (512-dim) without projection."""
        return self.backbone(x)


class ToothEmbeddingModelWithMetadata(nn.Module):
    """
    ResNet-18 backbone + FDI metadata embedding, fused at feature level.

    Architecture:
        image -> ResNet-18 -> 512-dim features
        fdi_idx -> Embedding(num_fdi, fdi_embedding_dim) -> 16-dim
        concat -> Dropout -> Linear(512+16, embedding_dim) -> L2 normalize
    """

    def __init__(
        self,
        num_fdi: int,
        fdi_embedding_dim: int = 16,
        embedding_dim: int = 128,
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        self.feature_dim = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Identity()

        self.fdi_embedding = nn.Embedding(num_fdi, fdi_embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.projection_head = nn.Linear(self.feature_dim + fdi_embedding_dim, embedding_dim)

    def forward(self, image: torch.Tensor, fdi_idx: torch.Tensor) -> torch.Tensor:
        visual = self.backbone(image)          # (B, 512)
        meta = self.fdi_embedding(fdi_idx)     # (B, fdi_dim)
        fused = torch.cat([visual, meta], dim=1)
        fused = self.dropout(fused)
        embeddings = self.projection_head(fused)
        return F.normalize(embeddings, p=2, dim=1)
