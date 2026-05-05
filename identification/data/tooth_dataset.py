"""
PyTorch Dataset for tooth crop images.

Loads crops from the manifest CSV, supports configurable label columns
(tooth_fdi, erupted, root_complete), crop modes (raw/masked), and
augmentation transforms.
"""

import random
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class GaussianNoise:
    """Add Gaussian noise to a tensor (applied after ToTensor, before Normalize)."""

    def __init__(self, sigma: float = 0.01):
        self.sigma = sigma

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.sigma > 0:
            tensor = tensor + torch.randn_like(tensor) * self.sigma
        return tensor


def get_train_transforms(aug_cfg: Optional[dict] = None) -> transforms.Compose:
    """Build training transforms with medical-image-safe augmentations."""
    cfg = aug_cfg or {}
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(
            degrees=cfg.get("rotation_degrees", 10),
            translate=(cfg.get("translate", 0.05), cfg.get("translate", 0.05)),
        ),
        transforms.ColorJitter(
            brightness=cfg.get("brightness", 0.1),
            contrast=cfg.get("contrast", 0.1),
        ),
        transforms.GaussianBlur(
            kernel_size=cfg.get("blur_kernel", 3),
            sigma=(0.1, cfg.get("blur_sigma_max", 0.5)),
        ),
        transforms.ToTensor(),
        GaussianNoise(sigma=cfg.get("noise_sigma", 0.01)),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms() -> transforms.Compose:
    """Build validation/test transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class ToothDataset(Dataset):
    """
    Dataset for tooth crop classification tasks.

    Args:
        manifest_path: Path to manifest CSV.
        split: One of "train", "val", "test".
        root_dir: Project root for resolving relative crop paths.
        crop_mode: "raw" or "masked".
        target_col: Column to use as label (e.g., "tooth_fdi", "erupted").
        filter_fn: Optional callable to filter the DataFrame before use.
        transform: Torchvision transform pipeline.
        label_map: Pre-built label mapping dict. If None, must call build_label_map().
    """

    def __init__(
        self,
        manifest_path: str,
        split: str,
        root_dir: Optional[str] = None,
        crop_mode: str = "raw",
        target_col: str = "tooth_fdi",
        filter_fn: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        label_map: Optional[Dict[str, int]] = None,
        return_metadata: bool = False,
        fdi_label_map: Optional[Dict[str, int]] = None,
    ):
        self.root_dir = Path(root_dir) if root_dir else PROJECT_ROOT
        self.crop_mode = crop_mode
        self.target_col = target_col
        self.transform = transform
        self.return_metadata = return_metadata
        self.fdi_label_map = fdi_label_map

        # Load and filter manifest
        df = pd.read_csv(manifest_path, dtype=str)
        if filter_fn is not None:
            df = filter_fn(df)
        self.df = df[df["split"] == split].reset_index(drop=True)

        # Path column
        self.path_col = "crop_path" if crop_mode == "raw" else "masked_crop_path"

        # Label mapping
        if label_map is None:
            raise ValueError("label_map is required. Use ToothDataset.build_label_map().")
        self.label_map = label_map
        self.label_to_name = {v: k for k, v in label_map.items()}

        if return_metadata and fdi_label_map is None:
            raise ValueError("fdi_label_map is required when return_metadata=True.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.root_dir / row[self.path_col]

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            # Return black image on load failure
            img = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform is not None:
            img = self.transform(img)

        label = self.label_map[row[self.target_col]]

        if self.return_metadata:
            fdi_idx = self.fdi_label_map[row["tooth_fdi"]]
            return img, label, fdi_idx

        return img, label

    @staticmethod
    def build_label_map(
        manifest_path: str,
        target_col: str = "tooth_fdi",
        filter_fn: Optional[Callable] = None,
    ) -> Dict[str, int]:
        """
        Build a deterministic label mapping from all splits.

        Returns dict mapping unique target values to contiguous integers.
        For tooth_fdi, sorts numerically. For boolean columns, sorts alphabetically.
        """
        df = pd.read_csv(manifest_path, dtype=str)
        if filter_fn is not None:
            df = filter_fn(df)

        values = df[target_col].dropna().unique().tolist()

        # Sort numerically if all values are digits, else alphabetically
        if all(v.isdigit() for v in values):
            values = sorted(values, key=int)
        else:
            values = sorted(values)

        return {v: i for i, v in enumerate(values)}

    @property
    def num_classes(self) -> int:
        return len(self.label_map)

    def get_labels(self) -> list:
        """Return integer labels for all samples (used by PK sampler)."""
        return [self.label_map[row[self.target_col]] for _, row in self.df.iterrows()]

    def get_class_counts(self) -> Dict[str, int]:
        """Return per-class sample counts for this split."""
        counts = self.df[self.target_col].value_counts().to_dict()
        return counts
