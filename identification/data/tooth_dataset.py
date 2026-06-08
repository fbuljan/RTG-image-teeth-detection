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
    """Build training transforms with medical-image-safe augmentations.

    The optional ``scale`` key enables bbox-jitter (Phase 8.2): RandomAffine
    samples a per-image isotropic scale in the given (min, max) range, which
    simulates the YOLO-vs-GT crop-framing distribution shift that Phase 7.1
    surfaced (median 4.8 deg polygon disagreement, ~30% of crops differ
    measurably in framing).
    """
    cfg = aug_cfg or {}
    translate = cfg.get("translate", 0.05)
    scale_cfg = cfg.get("scale")  # None disables; (lo, hi) tuple enables jitter
    affine_kwargs = dict(
        degrees=cfg.get("rotation_degrees", 10),
        translate=(translate, translate),
        fill=0,
    )
    if scale_cfg is not None:
        affine_kwargs["scale"] = tuple(scale_cfg)
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(**affine_kwargs),
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
        self._split = split

        # Load and filter manifest
        df = pd.read_csv(manifest_path, dtype=str)
        if filter_fn is not None:
            df = filter_fn(df)
        self.df = df[df["split"] == split].reset_index(drop=True)

        # Path column
        self.path_col = "crop_path" if crop_mode == "raw" else "masked_crop_path"

        # Phase 8.3 — optional GT->YOLO crop blend. Configured by enable_yolo_blend().
        self._blend_map: Optional[Dict[tuple, str]] = None
        self._blend_prob: float = 0.0
        self._blend_log: list = []  # captures first-epoch substitution trace for diagnostics

        # Label mapping
        if label_map is None:
            raise ValueError("label_map is required. Use ToothDataset.build_label_map().")
        self.label_map = label_map
        self.label_to_name = {v: k for k, v in label_map.items()}

        if return_metadata and fdi_label_map is None:
            raise ValueError("fdi_label_map is required when return_metadata=True.")

    def __len__(self) -> int:
        return len(self.df)

    def enable_yolo_blend(self, pair_table_path: str, prob: float,
                          log_trace: bool = True) -> int:
        """
        Phase 8.3 — at train time, substitute the GT crop with a verified-aligned
        YOLO crop with probability `prob`. Only accept=True rows in the pair table
        are eligible (IoU>=0.5 AND fdi_confidence>=0.5).

        Hard guard: only enable on training split.

        Returns: number of (image_id, fdi) pairs available in the blend map.
        """
        if self._split != "train":
            raise RuntimeError(
                f"enable_yolo_blend called on split={self._split!r}; "
                "blending must never touch val/test."
            )
        if not (0.0 <= prob <= 1.0):
            raise ValueError(f"blend prob must be in [0,1], got {prob}")

        pt = pd.read_csv(pair_table_path)
        eligible = pt[pt["accept"].astype(str).str.lower().isin(["true", "1"])]
        # Sanity: all eligible image_ids must be train-split panoramics in self.df
        train_ids = set(self.df["image_id"].unique())
        leaked = set(eligible["image_id"].unique()) - train_ids
        if leaked:
            raise RuntimeError(
                f"pair table contains {len(leaked)} image_ids not in this dataset's "
                f"train split; would leak val/test data. Examples: "
                f"{sorted(leaked)[:3]}"
            )
        self._blend_map = {
            (str(row.image_id), str(row.tooth_fdi)): str(row.yolo_crop_path)
            for row in eligible.itertuples(index=False)
        }
        self._blend_prob = float(prob)
        if log_trace:
            self._blend_log = []
        return len(self._blend_map)

    def get_blend_trace_summary(self) -> dict:
        """Return the per-epoch substitution stats captured since enable_yolo_blend."""
        if not self._blend_log:
            return {"calls": 0, "substituted": 0, "rate": 0.0}
        n_subs = sum(1 for entry in self._blend_log if entry["substituted"])
        return {
            "calls": len(self._blend_log),
            "substituted": n_subs,
            "rate": n_subs / len(self._blend_log) if self._blend_log else 0.0,
            "first10": self._blend_log[:10],
        }

    def reset_blend_trace(self) -> None:
        self._blend_log = []

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = row[self.path_col]

        # Phase 8.3 GT->YOLO blend (training only, gated by enable_yolo_blend)
        if self._blend_map is not None and self._blend_prob > 0.0:
            key = (str(row["image_id"]), str(row["tooth_fdi"]))
            substituted = False
            if key in self._blend_map and random.random() < self._blend_prob:
                path = self._blend_map[key]
                substituted = True
            if len(self._blend_log) < 5000:  # keep trace bounded
                self._blend_log.append({
                    "image_id": key[0], "tooth_fdi": key[1],
                    "substituted": substituted,
                })

        img_path = self.root_dir / path

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
