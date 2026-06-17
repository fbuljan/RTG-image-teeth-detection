"""
Confidence-weighted aggregation.

Post-hoc weighted aggregation of per-tooth embeddings using a learned softmax
over per-tooth feature scores. The deployed embedder is unchanged; only the
aggregation function and the registry are re-computed.

Features per tooth (all derivable from cached Stage A/C output):
  - fdi_idx          -> indexes a learned per-FDI scalar prior `beta_fdi[fdi_i]`
  - yolo_logit       = logit(yolo_conf)
  - log_norm_area    = log(polygon_area / image_area)
  - low_conf_flag    = 1[fdi_conf < 0.5]
Image-level scalar:
  - log_n_teeth      = log(number of detected teeth in image)

Score:
  s_i = beta_fdi[fdi_i] + alpha * yolo_logit + gamma * log_norm_area
        + eta * low_conf_flag + mu * log_n_teeth
Weights:
  w_i = softmax(s_i / T)
Person embedding:
  e_person = L2(sum_i w_i * emb_i)

When all betas/alpha/gamma/eta/mu are zero, softmax reduces to uniform and the
output equals the deployed mean-pool baseline.

This module exposes:
  - PerToothFeatures: dataclass + extractor from a list of StageACOutput
  - WeightConfig: dataclass holding the fitted hyperparameters
  - weighted_person_embedding(): aggregation function
  - All helpers reusable by fit_aggregator.py (val) and eval_aggregator.py (test).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Per-tooth feature extraction                                                #
# --------------------------------------------------------------------------- #

@dataclass
class PerToothFeatures:
    """Per-tooth features extracted from one Stage A/C cached payload."""

    image_id: str
    person_id: str
    fdi: List[str]              # length T
    fdi_idx: np.ndarray         # int (T,), mapped via FDI_LABEL_MAP
    yolo_logit: np.ndarray      # float (T,)
    log_norm_area: np.ndarray   # float (T,)
    low_conf_flag: np.ndarray   # float (T,), 0/1
    log_n_teeth: float          # scalar, image-level
    n_teeth: int                # T


def _logit(p: float, eps: float = 1e-6) -> float:
    p = float(np.clip(p, eps, 1.0 - eps))
    return float(np.log(p / (1.0 - p)))


def _polygon_area(poly: list) -> float:
    """OpenCV contour area in pixels (poly is list of [x,y])."""
    arr = np.asarray(poly, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 3:
        return 0.0
    return float(cv2.contourArea(arr))


def build_fdi_label_map(manifest_path: Path) -> Dict[str, int]:
    """Build a deterministic FDI -> int label map from the manifest.

    Uses the same protocol as ToothDataset.build_label_map(target_col='tooth_fdi').
    """
    import pandas as pd
    df = pd.read_csv(manifest_path, dtype=str)
    values = df["tooth_fdi"].dropna().unique().tolist()
    if all(v.isdigit() for v in values):
        values = sorted(values, key=int)
    else:
        values = sorted(values)
    return {v: i for i, v in enumerate(values)}


def extract_features(stage_ac_payload: dict, fdi_label_map: Dict[str, int]) -> PerToothFeatures:
    """Extract per-tooth feature arrays from one Stage A/C JSON payload."""
    image_id = stage_ac_payload["image_id"]
    person_id = stage_ac_payload["person_id"]
    fdi_list = stage_ac_payload["fdi_labels"]
    yolo_confs = stage_ac_payload["yolo_confidences"]
    fdi_confs = stage_ac_payload["fdi_confidences"]
    polygons = stage_ac_payload["polygons"]
    pano_w, pano_h = stage_ac_payload["pano_size"]
    image_area = float(pano_w) * float(pano_h)

    n = len(fdi_list)
    # FDI -> idx; unknown FDIs map to a special bucket (e.g., 0 with downweight)
    # Use last index + 1 as "unknown" if any FDI not in map
    max_idx = max(fdi_label_map.values()) if fdi_label_map else -1
    unknown_idx = max_idx + 1

    fdi_idx = np.array(
        [fdi_label_map.get(fdi, unknown_idx) for fdi in fdi_list], dtype=np.int64,
    )
    yolo_logit = np.array([_logit(c) for c in yolo_confs], dtype=np.float32)
    areas = np.array([_polygon_area(p) for p in polygons], dtype=np.float32)
    # Replace zero areas with the smallest non-zero to avoid -inf
    nonzero_min = float(areas[areas > 0].min()) if (areas > 0).any() else 1.0
    areas = np.where(areas > 0, areas, nonzero_min)
    log_norm_area = np.log(areas / image_area).astype(np.float32)
    low_conf_flag = np.array(
        [1.0 if c < 0.5 else 0.0 for c in fdi_confs], dtype=np.float32,
    )
    log_n_teeth = float(np.log(max(n, 1)))

    return PerToothFeatures(
        image_id=image_id,
        person_id=person_id,
        fdi=list(fdi_list),
        fdi_idx=fdi_idx,
        yolo_logit=yolo_logit,
        log_norm_area=log_norm_area,
        low_conf_flag=low_conf_flag,
        log_n_teeth=log_n_teeth,
        n_teeth=n,
    )


# --------------------------------------------------------------------------- #
# Weight configuration + softmax aggregation                                  #
# --------------------------------------------------------------------------- #

@dataclass
class WeightConfig:
    """Hyperparameters for confidence-weighted aggregation.

    `beta_fdi` is a (num_fdi_classes + 1,) vector; index num_fdi_classes is the
    "unknown FDI" bucket. The mean-pool baseline corresponds to all scalars=0,
    any T (softmax reduces to uniform).
    """
    alpha: float = 0.0          # yolo_logit coefficient
    gamma: float = 0.0          # log_norm_area coefficient
    eta: float = 0.0            # low_conf_flag coefficient (negative => downweight)
    mu: float = 0.0             # log_n_teeth coefficient (image-level)
    T: float = 1.0              # softmax temperature (high T -> uniform)
    beta_fdi: np.ndarray = field(default_factory=lambda: np.zeros(53, dtype=np.float32))

    def is_mean_pool(self) -> bool:
        return (
            abs(self.alpha) < 1e-9 and abs(self.gamma) < 1e-9
            and abs(self.eta) < 1e-9 and abs(self.mu) < 1e-9
            and float(np.abs(self.beta_fdi).max()) < 1e-9
        )

    def to_payload(self) -> dict:
        return {
            "alpha": float(self.alpha),
            "gamma": float(self.gamma),
            "eta": float(self.eta),
            "mu": float(self.mu),
            "T": float(self.T),
            "beta_fdi": self.beta_fdi.tolist(),
        }

    @classmethod
    def from_payload(cls, p: dict) -> "WeightConfig":
        return cls(
            alpha=float(p["alpha"]),
            gamma=float(p["gamma"]),
            eta=float(p["eta"]),
            mu=float(p["mu"]),
            T=float(p["T"]),
            beta_fdi=np.asarray(p["beta_fdi"], dtype=np.float32),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_payload(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "WeightConfig":
        with open(path) as f:
            return cls.from_payload(json.load(f))


def compute_weights(features: PerToothFeatures, cfg: WeightConfig) -> np.ndarray:
    """Compute per-tooth softmax weights for one image."""
    if features.n_teeth == 0:
        return np.zeros(0, dtype=np.float32)
    beta_per_tooth = cfg.beta_fdi[features.fdi_idx]
    s = (
        beta_per_tooth
        + cfg.alpha * features.yolo_logit
        + cfg.gamma * features.log_norm_area
        + cfg.eta * features.low_conf_flag
        + cfg.mu * features.log_n_teeth  # broadcast constant
    )
    # Numerically stable softmax with temperature
    T = max(cfg.T, 1e-3)
    s_scaled = s / T
    s_scaled -= s_scaled.max()
    w = np.exp(s_scaled)
    Z = w.sum()
    if Z <= 0 or not np.isfinite(Z):
        return np.full(features.n_teeth, 1.0 / features.n_teeth, dtype=np.float32)
    return (w / Z).astype(np.float32)


def weighted_person_embedding(
    per_tooth_embs: np.ndarray,        # (T, D)
    features: PerToothFeatures,
    cfg: WeightConfig,
) -> np.ndarray:
    """Return L2-normalised person embedding under cfg's softmax weights."""
    w = compute_weights(features, cfg)
    if per_tooth_embs.shape[0] == 0:
        return np.zeros(per_tooth_embs.shape[1], dtype=np.float32)
    if w.shape[0] != per_tooth_embs.shape[0]:
        # Safety: drop the mismatch and fall back to uniform
        w = np.full(per_tooth_embs.shape[0], 1.0 / per_tooth_embs.shape[0], dtype=np.float32)
    pooled = (w[:, None] * per_tooth_embs).sum(axis=0)
    nrm = np.linalg.norm(pooled)
    if nrm < 1e-12:
        return pooled.astype(np.float32)
    return (pooled / nrm).astype(np.float32)


def effective_n_teeth(features: PerToothFeatures, cfg: WeightConfig) -> float:
    """Effective number of teeth contributing to the pool, per Shannon-style guard.

    For uniform weights w_i = 1/T, returns T. For one-hot weights, returns 1.
    Used as a sparsity guard (pre-registered weighted-aggregation criterion #5).
    """
    if features.n_teeth == 0:
        return 0.0
    w = compute_weights(features, cfg)
    denom = float((w ** 2).sum())
    if denom <= 0:
        return 0.0
    return 1.0 / denom
