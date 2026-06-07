"""End-to-end YOLO-pipeline evaluation (Phase 8.0 baseline).

Runs the deployed demo pipeline (YOLO segmentation → crop → FDI dedup → embed)
on the held-out test split and reports R1/R5/R10/mAP at n_query ∈ {1, 2, 4, 8, 16}
with bootstrap 95% CIs over persons. (Closed-set verification AUROC is not
computed here; the open-set AUROC for Phase 8.6 is derived downstream from the
`records` flat list in `heldout_enrol.json`.) Adds two adversarial slices:

* rotation-stress: every test panoramic is rotated by a uniformly drawn angle in
  ±rotation_deg before YOLO segmentation. PAIRED to the baseline by using the
  same per-person random tooth subsets, so paired-bootstrap difference CIs are
  meaningful at N=178.
* held-out enrolment: a sample of test persons is removed from the full
  1,178-person production registry; queries from the held-out persons are
  recorded against the remaining 1,148. Per-query records (similarity, gap,
  pid, label) are saved flat so Phase 8.6 can compute open-set AUROC downstream.

Two-layer cache so Phase 8.1–8.10 can swap the embedder without re-running YOLO:

  Stage A/C cache (YOLO + crops + FDI dedup) is keyed on
    (image_id, rotation_deg, yolo_hash, fdi_hash, crop_size, yolo_conf, yolo_iou, yolo_imgsz)
  Embedding cache is keyed on
    (stage_ac_key, embedder_hash)

Also computes YOLO mask-mAP per rotation bucket (vs GT red-mask polygons from
the per-FDI mask PNGs in dataset_raw), so a Phase 8.1 rotation-stress drop can
be attributed to YOLO or the embedder.

Usage:
    python -m identification.evaluation.evaluate_pipeline \\
        --output-dir identification/runs/phase8_baseline \\
        --rotation-deg 30 \\
        --heldout-count 30
"""

from __future__ import annotations

import argparse
import base64
import functools
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Apple Silicon BLAS sometimes flags spurious FMA edge cases as divide-by-zero
# / overflow / invalid during matmul on L2-normalised float32 vectors. The
# resulting matrix is mathematically fine (we assert finiteness explicitly on
# every result), so we suppress these noise warnings.
warnings.filterwarnings("ignore", message=".*encountered in matmul.*", category=RuntimeWarning)

# Make `print()` flush immediately so background-run progress is visible.
print = functools.partial(print, flush=True)  # type: ignore[assignment]
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from backend.pipeline import (
    PipelineConfig,
    PipelineModels,
    _expand_bbox,
    _to_tensor,
)
from identification.models.embedding_model import ToothEmbeddingModelWithMetadata
from identification.models.retrieval_index import RetrievalIndex

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Hashing + cache keys
# ---------------------------------------------------------------------------

def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _stage_ac_key(
    image_id: str,
    rotation_deg: float,
    yolo_hash: str,
    fdi_hash: str,
    crop_size: int,
    yolo_conf: float,
    yolo_iou: float,
    yolo_imgsz: int,
) -> str:
    deg = int(round(rotation_deg * 100))
    return (
        f"{image_id}__rot{deg:+06d}__yolo{yolo_hash}__fdi{fdi_hash}"
        f"__c{crop_size}__yc{int(yolo_conf*1000)}__yi{int(yolo_iou*1000)}__ys{yolo_imgsz}"
    )


def _emb_key(stage_ac_key: str, embedder_hash: str) -> str:
    return f"{stage_ac_key}__emb{embedder_hash}"


# ---------------------------------------------------------------------------
# Stage A/C cache payload — YOLO + crops + FDI dedup
# ---------------------------------------------------------------------------

@dataclass
class StageACOutput:
    """Output of YOLO + crop + FDI dedup, before embedding."""

    person_id: str
    image_id: str
    rotation_deg: float
    polygons: list[list[list[float]]]   # raw YOLO polygons in panoramic (rotated) coords
    bboxes: list[list[float]]           # tight polygon bboxes, expanded by 10%
    crops_b64: list[str]                # PNG bytes, base64 — the exact tensors fed to embedder
    fdi_labels: list[str]
    fdi_confidences: list[float]
    yolo_confidences: list[float]
    yolo_mask_iou_vs_gt: float | None   # mean IoU of YOLO masks vs GT polygons; None if GT unavailable
    yolo_mask_recall_vs_gt: float | None  # frac of GT teeth recovered (IoU >= 0.5)
    yolo_mask_precision_vs_gt: float | None  # frac of YOLO masks matched
    n_dropped_dedup: int
    pano_size: list[int]                # (w, h) AFTER rotation

    def to_payload(self) -> dict:
        return {
            "person_id": self.person_id,
            "image_id": self.image_id,
            "rotation_deg": self.rotation_deg,
            "polygons": self.polygons,
            "bboxes": self.bboxes,
            "crops_b64": self.crops_b64,
            "fdi_labels": self.fdi_labels,
            "fdi_confidences": self.fdi_confidences,
            "yolo_confidences": self.yolo_confidences,
            "yolo_mask_iou_vs_gt": self.yolo_mask_iou_vs_gt,
            "yolo_mask_recall_vs_gt": self.yolo_mask_recall_vs_gt,
            "yolo_mask_precision_vs_gt": self.yolo_mask_precision_vs_gt,
            "n_dropped_dedup": self.n_dropped_dedup,
            "pano_size": self.pano_size,
        }

    @classmethod
    def from_payload(cls, p: dict) -> "StageACOutput":
        return cls(
            person_id=p["person_id"],
            image_id=p["image_id"],
            rotation_deg=float(p["rotation_deg"]),
            polygons=[[[float(x), float(y)] for x, y in poly] for poly in p["polygons"]],
            bboxes=[list(map(float, b)) for b in p["bboxes"]],
            crops_b64=list(p["crops_b64"]),
            fdi_labels=list(p["fdi_labels"]),
            fdi_confidences=list(p["fdi_confidences"]),
            yolo_confidences=list(p["yolo_confidences"]),
            yolo_mask_iou_vs_gt=p.get("yolo_mask_iou_vs_gt"),
            yolo_mask_recall_vs_gt=p.get("yolo_mask_recall_vs_gt"),
            yolo_mask_precision_vs_gt=p.get("yolo_mask_precision_vs_gt"),
            n_dropped_dedup=int(p["n_dropped_dedup"]),
            pano_size=list(map(int, p["pano_size"])),
        )


def _is_failure_sentinel(payload: dict) -> bool:
    return payload.get("_failed") is True


def _crop_to_b64(crop: Image.Image) -> str:
    buf = io.BytesIO()
    crop.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _b64_to_crop(s: str) -> Image.Image:
    raw = base64.b64decode(s.encode("ascii"))
    return Image.open(io.BytesIO(raw)).convert("RGB")


# ---------------------------------------------------------------------------
# GT polygon loading (for YOLO mask-mAP)
# ---------------------------------------------------------------------------

def load_gt_polygons(panoramic_id: str) -> dict[str, np.ndarray]:
    """Return {fdi: polygon} for the GT red-mask polygons in dataset_raw.

    Mirrors utils/yolo/masks_to_yolo_dataset.py:54-65 (HSV red-mask detection).
    Returns the largest-contour polygon per FDI, in panoramic (un-rotated) coords.
    """
    folder = PROJECT_ROOT / "dataset_raw" / panoramic_id
    if not folder.exists():
        return {}
    out: dict[str, np.ndarray] = {}
    prefix = f"{panoramic_id}+"
    for entry in folder.iterdir():
        if not entry.name.startswith(prefix) or not entry.name.endswith(".png"):
            continue
        m = re.match(re.escape(prefix) + r"(\d+)\.png$", entry.name)
        if not m:
            continue
        fdi = m.group(1)
        mask_bgr = cv2.imread(str(entry))
        if mask_bgr is None:
            continue
        hsv = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
        red = cv2.bitwise_or(m1, m2)
        contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        biggest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(biggest) < 30:
            continue
        poly = biggest.squeeze()
        if poly.ndim != 2 or poly.shape[0] < 3:
            continue
        out[fdi] = poly.astype(np.float32)
    return out


def _polygon_to_mask(poly: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Render a polygon (N,2) into a binary mask (H,W) uint8 of the given (W,H) canvas."""
    w, h = size
    mask = np.zeros((h, w), dtype=np.uint8)
    if poly is None or len(poly) < 3:
        return mask
    cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
    return mask


def _rotate_polygon(poly: np.ndarray, deg: float, src_size: tuple[int, int],
                     dst_size: tuple[int, int]) -> np.ndarray:
    """Rotate a polygon by `deg` around the source-image center, mapping into dst canvas.

    Source center = (src_w/2, src_h/2); dst center = (dst_w/2, dst_h/2).
    Used to align GT polygons (un-rotated) with the YOLO output coordinate frame
    (rotated). Matches PIL.Image.rotate(deg, expand=False) center-rotation
    convention. PIL rotates COUNTER-clockwise for positive deg, so the affine
    matches the standard 2D rotation matrix with deg→radians.
    """
    if abs(deg) < 1e-6 and src_size == dst_size:
        return poly.copy()
    sw, sh = src_size
    dw, dh = dst_size
    theta = np.deg2rad(deg)
    c, s = np.cos(theta), np.sin(theta)
    cx_src, cy_src = sw / 2.0, sh / 2.0
    cx_dst, cy_dst = dw / 2.0, dh / 2.0
    # Translate to origin, rotate (counter-clockwise to match PIL), translate to dst center.
    # PIL's positive angle = counter-clockwise; in image coords (y down) that's R(-theta).
    # Empirically: PIL rotates counter-clockwise as seen on screen → in image-coord matrix
    # this means y is flipped relative to math convention. Use R(theta) with image coords:
    rotated = np.empty_like(poly)
    rotated[:, 0] = c * (poly[:, 0] - cx_src) + s * (poly[:, 1] - cy_src) + cx_dst
    rotated[:, 1] = -s * (poly[:, 0] - cx_src) + c * (poly[:, 1] - cy_src) + cy_dst
    return rotated


def compute_yolo_mask_vs_gt(
    yolo_polygons: list[np.ndarray],
    gt_polygons: dict[str, np.ndarray],
    src_size: tuple[int, int],
    dst_size: tuple[int, int],
    rotation_deg: float,
    iou_threshold: float = 0.5,
) -> tuple[float, float, float]:
    """Match YOLO masks to GT masks under one-to-one greedy IoU, return (mean_iou, recall, precision).

    GT polygons are rotated into the YOLO output coordinate frame first.
    """
    if not gt_polygons or not yolo_polygons:
        return (0.0, 0.0, 0.0)
    # Rasterize all masks at the YOLO output canvas size.
    yolo_masks = [_polygon_to_mask(p, dst_size) for p in yolo_polygons]
    gt_rotated = {
        fdi: _rotate_polygon(poly, rotation_deg, src_size, dst_size)
        for fdi, poly in gt_polygons.items()
    }
    gt_masks = {fdi: _polygon_to_mask(p, dst_size) for fdi, p in gt_rotated.items()}

    # All-pairs IoU
    n_y = len(yolo_masks)
    fdis = list(gt_masks.keys())
    n_g = len(fdis)
    iou = np.zeros((n_y, n_g), dtype=np.float32)
    for i, ym in enumerate(yolo_masks):
        if ym.sum() == 0:
            continue
        for j, fdi in enumerate(fdis):
            gm = gt_masks[fdi]
            if gm.sum() == 0:
                continue
            inter = int(np.logical_and(ym, gm).sum())
            if inter == 0:
                continue
            union = int(np.logical_or(ym, gm).sum())
            iou[i, j] = inter / max(1, union)

    # Greedy 1-to-1 matching
    matched_ious: list[float] = []
    used_y = set()
    used_g = set()
    flat = [(iou[i, j], i, j) for i in range(n_y) for j in range(n_g) if iou[i, j] > 0]
    flat.sort(reverse=True)
    for v, i, j in flat:
        if i in used_y or j in used_g:
            continue
        used_y.add(i)
        used_g.add(j)
        matched_ious.append(v)

    if not matched_ious:
        return (0.0, 0.0, 0.0)
    mean_iou = float(np.mean(matched_ious))
    n_tp = sum(1 for v in matched_ious if v >= iou_threshold)
    recall = n_tp / max(1, n_g)
    precision = n_tp / max(1, n_y)
    return (mean_iou, recall, precision)


# ---------------------------------------------------------------------------
# Stage A/C extraction (YOLO + crop + FDI dedup)
# ---------------------------------------------------------------------------

def _rotated_pano_temp_path(pano: Image.Image, deg: float, scratch: Path) -> tuple[Path, Image.Image]:
    """Rotate the PIL image and write it to a temp file.

    We write a temp file rather than passing a numpy array to ultralytics so the
    backend's `source=str(path)` path (which uses cv2 BGR loading) is matched
    bit-for-bit. expand=False keeps the canvas size identical so YOLO sees the
    same aspect ratio as upright.
    """
    if abs(deg) < 1e-6:
        rotated = pano
    else:
        rotated = pano.rotate(deg, resample=Image.BILINEAR, expand=False, fillcolor=(0, 0, 0))
    scratch.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="pano_", suffix=".png", dir=str(scratch))
    os.close(fd)
    rotated.save(name, format="PNG")
    return Path(name), rotated


def extract_stage_ac(
    panoramic_path: Path,
    person_id: str,
    image_id: str,
    models: PipelineModels,
    rotation_deg: float = 0.0,
    scratch_dir: Path | None = None,
    gt_polygons: dict[str, np.ndarray] | None = None,
) -> StageACOutput | None:
    """Synchronous Stage A + Stage B + Stage C of the deployed pipeline.

    Mirrors backend.pipeline.run_pipeline lines covering segmentation, polygon
    bbox derivation, 10% padding, FDI classification, and duplicate-FDI dedup.
    Embedding is NOT done here so the result can be cached and reused across
    Phase 8.1+ embedder swaps.
    """
    cfg = models.config
    device = models.device

    pano = Image.open(panoramic_path).convert("RGB")
    src_w, src_h = pano.size
    scratch_dir = scratch_dir or (cfg.temp_dir / "phase8_eval_scratch")
    rotated_path, rotated_pil = _rotated_pano_temp_path(pano, rotation_deg, scratch_dir)
    dst_w, dst_h = rotated_pil.size

    try:
        results = models.yolo_seg.predict(
            source=str(rotated_path),
            conf=cfg.yolo_conf,
            iou=cfg.yolo_iou,
            imgsz=cfg.yolo_imgsz,
            verbose=False,
            device=device,
        )
    finally:
        try:
            rotated_path.unlink()
        except FileNotFoundError:
            pass

    if not results:
        return None
    res0 = results[0]
    boxes = res0.boxes
    if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
        return None
    masks_obj = getattr(res0, "masks", None)
    if masks_obj is None or masks_obj.xy is None or len(masks_obj.xy) == 0:
        return None

    polygons = [np.asarray(p, dtype=np.float32) for p in masks_obj.xy]
    yolo_confs_all = (
        boxes.conf.cpu().numpy().tolist() if boxes.conf is not None else [1.0] * len(polygons)
    )

    # Compute YOLO mask-vs-GT IoU now (BEFORE bbox derivation) so it reflects YOLO output directly.
    mask_iou = mask_recall = mask_prec = None
    if gt_polygons:
        mask_iou, mask_recall, mask_prec = compute_yolo_mask_vs_gt(
            polygons, gt_polygons, (src_w, src_h), (dst_w, dst_h), rotation_deg,
        )

    # Polygon-tight bbox + 10% expansion (matches backend pipeline.py Stage A→B)
    bboxes_raw = []
    for poly in polygons:
        bboxes_raw.append([
            float(poly[:, 0].min()), float(poly[:, 1].min()),
            float(poly[:, 0].max()), float(poly[:, 1].max()),
        ])
    bboxes_raw_arr = np.asarray(bboxes_raw)

    # Stage B: crops (expanded bbox)
    crops: list[Image.Image] = []
    bboxes_expanded: list[list[float]] = []
    for x1, y1, x2, y2 in bboxes_raw_arr:
        ex1, ey1, ex2, ey2 = _expand_bbox(
            (float(x1), float(y1), float(x2), float(y2)),
            rotated_pil.size, padding_ratio=0.1,
        )
        crops.append(rotated_pil.crop((ex1, ey1, ex2, ey2)))
        bboxes_expanded.append([float(ex1), float(ey1), float(ex2), float(ey2)])

    # Stage C: FDI classification
    fdi_logits = []
    with torch.no_grad():
        for crop in crops:
            t = _to_tensor(crop, cfg.crop_size, device)
            fdi_logits.append(models.fdi_classifier(t))
    probs = F.softmax(torch.cat(fdi_logits, dim=0), dim=1).cpu().numpy()
    fdi_idx = probs.argmax(axis=1)
    fdi_conf_all = probs.max(axis=1)
    fdi_labels_all = [models.fdi_label_inv[int(k)] for k in fdi_idx]

    # FDI dedup — mirror backend/pipeline.py lines 420-428 exactly.
    # When the second crop wins (>conf), the previously-kept index is removed.
    # n_dropped = number of crops dropped (either the current loser or the one being replaced).
    seen: dict[str, int] = {}
    keep_indices: list[int] = []
    n_dropped = 0
    for i, fdi in enumerate(fdi_labels_all):
        if fdi in seen and fdi_conf_all[i] <= fdi_conf_all[seen[fdi]]:
            n_dropped += 1
            continue
        if fdi in seen:
            n_dropped += 1
            keep_indices.remove(seen[fdi])
        seen[fdi] = i
        keep_indices.append(i)
    keep_indices.sort()
    if not keep_indices:
        return None

    fdi_kept = [fdi_labels_all[i] for i in keep_indices]
    fdi_conf_kept = [float(fdi_conf_all[i]) for i in keep_indices]
    yolo_conf_kept = [float(yolo_confs_all[i]) for i in keep_indices]
    crops_kept = [crops[i] for i in keep_indices]
    polygons_kept = [polygons[i].tolist() for i in keep_indices]
    bboxes_kept = [bboxes_expanded[i] for i in keep_indices]

    return StageACOutput(
        person_id=person_id,
        image_id=image_id,
        rotation_deg=float(rotation_deg),
        polygons=polygons_kept,
        bboxes=bboxes_kept,
        crops_b64=[_crop_to_b64(c) for c in crops_kept],
        fdi_labels=fdi_kept,
        fdi_confidences=fdi_conf_kept,
        yolo_confidences=yolo_conf_kept,
        yolo_mask_iou_vs_gt=mask_iou,
        yolo_mask_recall_vs_gt=mask_recall,
        yolo_mask_precision_vs_gt=mask_prec,
        n_dropped_dedup=n_dropped,
        pano_size=[dst_w, dst_h],
    )


# ---------------------------------------------------------------------------
# Two-layer cache: stage A/C + embeddings
# ---------------------------------------------------------------------------

class PipelineCache:
    """Two-layer disk cache. YOLO+FDI work is reusable across embedder swaps."""

    def __init__(
        self,
        output_dir: Path,
        yolo_hash: str,
        fdi_hash: str,
        embedder_hash: str,
        crop_size: int,
        yolo_conf: float,
        yolo_iou: float,
        yolo_imgsz: int,
        scratch_dir: Path,
    ):
        self.stage_ac_dir = output_dir / "cache" / "stage_ac"
        self.emb_dir = output_dir / "cache" / "embeddings"
        self.stage_ac_dir.mkdir(parents=True, exist_ok=True)
        self.emb_dir.mkdir(parents=True, exist_ok=True)
        self.yolo_hash = yolo_hash
        self.fdi_hash = fdi_hash
        self.embedder_hash = embedder_hash
        self.crop_size = crop_size
        self.yolo_conf = yolo_conf
        self.yolo_iou = yolo_iou
        self.yolo_imgsz = yolo_imgsz
        self.scratch_dir = scratch_dir

    def get_stage_ac(
        self,
        models: PipelineModels,
        panoramic_path: Path,
        person_id: str,
        image_id: str,
        rotation_deg: float,
        gt_polygons: dict[str, np.ndarray] | None,
    ) -> StageACOutput | None:
        key = _stage_ac_key(
            image_id, rotation_deg, self.yolo_hash, self.fdi_hash,
            self.crop_size, self.yolo_conf, self.yolo_iou, self.yolo_imgsz,
        )
        p = self.stage_ac_dir / f"{key}.json"
        if p.exists():
            with open(p) as f:
                payload = json.load(f)
            if _is_failure_sentinel(payload):
                return None
            return StageACOutput.from_payload(payload)
        out = extract_stage_ac(
            panoramic_path, person_id, image_id, models,
            rotation_deg=rotation_deg, scratch_dir=self.scratch_dir,
            gt_polygons=gt_polygons,
        )
        if out is None:
            with open(p, "w") as f:
                json.dump({"_failed": True}, f)
            return None
        with open(p, "w") as f:
            json.dump(out.to_payload(), f)
        return out

    def get_embeddings(
        self,
        models: PipelineModels,
        stage_ac: StageACOutput,
    ) -> np.ndarray | None:
        stage_key = _stage_ac_key(
            stage_ac.image_id, stage_ac.rotation_deg, self.yolo_hash, self.fdi_hash,
            self.crop_size, self.yolo_conf, self.yolo_iou, self.yolo_imgsz,
        )
        key = _emb_key(stage_key, self.embedder_hash)
        p = self.emb_dir / f"{key}.npy"
        if p.exists():
            return np.load(p)
        # Compute
        crops = [_b64_to_crop(b) for b in stage_ac.crops_b64]
        if not crops:
            return None
        embs: list[np.ndarray] = []
        with torch.no_grad():
            for crop, fdi in zip(crops, stage_ac.fdi_labels):
                t = _to_tensor(crop, self.crop_size, models.device)
                if models.embedder_uses_metadata:
                    fdi_idx = models.embedder_fdi_label_map.get(fdi, 0)
                    ft = torch.tensor([fdi_idx], dtype=torch.long, device=models.device)
                    emb = models.embedder(t, ft)
                else:
                    emb = models.embedder(t)
                embs.append(emb.cpu().numpy()[0])
        arr = np.stack(embs).astype(np.float32)
        np.save(p, arr)
        return arr


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return (v / n if n > 1e-12 else v).astype(np.float32)


def _mean_pool(arr: np.ndarray) -> np.ndarray:
    return _l2_normalize(arr.mean(axis=0))


# ---------------------------------------------------------------------------
# Multi-tooth sweep (symmetric, Phase-5-comparable) with PAIRED trial sampling
# ---------------------------------------------------------------------------

def _draw_paired_subsets(
    per_person_embs: dict[str, np.ndarray],
    eligible: list[str],
    n_query: int,
    n_trials: int,
    rng: np.random.Generator,
) -> list[dict[str, np.ndarray]]:
    """For each trial, draw a permutation of each person's teeth.

    Returns a list of trials; each trial maps pid → permutation (indices).
    The permutation is what ROTATION-stress reuses to remain paired with baseline.
    """
    trials: list[dict[str, np.ndarray]] = []
    for _ in range(n_trials):
        trial = {pid: rng.permutation(len(per_person_embs[pid])) for pid in eligible}
        trials.append(trial)
    return trials


def _evaluate_sweep_symmetric_paired(
    per_person_embs: dict[str, np.ndarray],
    n_query_list: list[int],
    n_trials: int,
    rng: np.random.Generator,
    bootstrap_rng: np.random.Generator,
) -> tuple[list[dict], dict[int, dict[str, list[np.ndarray]]]]:
    """Symmetric sweep + return per-trial permutations for rotation-stress pairing.

    For each n_query, returns:
      - results: a list of {n_query, rank1_mean, rank1_ci95, ...} dicts
      - permutations[n_query] = list of trials, each {pid: indices}

    The permutations are returned so rotation-stress can reuse the SAME tooth indices,
    making a paired-difference bootstrap valid.
    """
    results: list[dict] = []
    permutations: dict[int, list[dict[str, np.ndarray]]] = {}
    for n_q in n_query_list:
        # Need at least n_q + 1 teeth (so gallery is non-empty)
        eligible = [pid for pid, e in per_person_embs.items() if len(e) >= n_q + 1]
        if len(eligible) < 5:
            results.append({"n_query": n_q, "n_persons": len(eligible), "skipped": True})
            permutations[n_q] = []
            continue

        trial_perms = _draw_paired_subsets(per_person_embs, eligible, n_q, n_trials, rng)
        permutations[n_q] = trial_perms

        # Per (trial, person) R1/R5/R10/AP indicator matrix
        n_eligible = len(eligible)
        match_r1 = np.zeros((n_trials, n_eligible), dtype=bool)
        match_r5 = np.zeros_like(match_r1)
        match_r10 = np.zeros_like(match_r1)
        ap_arr = np.zeros((n_trials, n_eligible), dtype=np.float64)

        for t, perm in enumerate(trial_perms):
            queries, galleries = [], []
            for pid in eligible:
                arr = per_person_embs[pid]
                idx = perm[pid]
                q_idx = idx[:n_q]
                g_idx = idx[n_q:]
                queries.append(_mean_pool(arr[q_idx]))
                galleries.append(_mean_pool(arr[g_idx]))
            Q = np.stack(queries)
            G = np.stack(galleries)
            if not (np.isfinite(Q).all() and np.isfinite(G).all()):
                raise RuntimeError(
                    f"sym sweep n_q={n_q} trial={t}: non-finite query/gallery embeddings"
                )
            sim = Q @ G.T
            if not np.isfinite(sim).all():
                raise RuntimeError(
                    f"sym sweep n_q={n_q} trial={t}: non-finite similarity matrix"
                )
            ranked = np.argsort(-sim, axis=1)
            pids_arr = np.array(eligible)
            ranked_labels = pids_arr[ranked]
            mat = ranked_labels == pids_arr[:, None]

            match_r1[t] = mat[:, 0]
            match_r5[t] = mat[:, :5].any(axis=1)
            match_r10[t] = mat[:, :10].any(axis=1)
            first_pos = np.argmax(mat, axis=1)
            valid = mat.any(axis=1)
            ap_arr[t] = np.where(valid, 1.0 / (first_pos + 1), 0.0)

        # Per-person point estimate = mean across trials
        per_person_r1 = match_r1.mean(axis=0)
        per_person_r5 = match_r5.mean(axis=0)
        per_person_ap = ap_arr.mean(axis=0)

        # Point estimate = mean across persons of per-person mean across trials
        rank1 = float(per_person_r1.mean())
        rank5 = float(per_person_r5.mean())
        rank10 = float(match_r10.mean(axis=0).mean())
        mAP = float(per_person_ap.mean())

        # Bootstrap over persons (averaging across trials inside each bootstrap).
        # Uses a dedicated bootstrap_rng so CIs are independent of slice ordering.
        n_boot = 1000
        boot_r1 = np.empty(n_boot)
        for b in range(n_boot):
            sel = bootstrap_rng.integers(0, n_eligible, size=n_eligible)
            boot_r1[b] = per_person_r1[sel].mean()
        ci_low, ci_high = np.percentile(boot_r1, [2.5, 97.5])

        results.append({
            "n_query": n_q,
            "method": "mean",
            "n_persons": n_eligible,
            "n_trials": n_trials,
            "rank1_mean": rank1,
            "rank1_ci95_low": float(ci_low),
            "rank1_ci95_high": float(ci_high),
            "rank5_mean": rank5,
            "rank10_mean": rank10,
            "mAP_mean": mAP,
            "per_person_r1": per_person_r1.tolist(),
            "pids": eligible,
        })
    return results, permutations


def _paired_diff_bootstrap(
    baseline_per_person: dict[int, dict[str, float]],
    stress_per_person: dict[int, dict[str, float]],
    n_query_list: list[int],
    rng: np.random.Generator,
    bootstrap_rng: np.random.Generator | None = None,
) -> list[dict]:
    """For each n_query, compute the paired rotation-vs-baseline R1 difference + bootstrap CI.

    baseline_per_person[n_q] = {pid: r1_mean_across_trials_for_this_person}
    """
    out: list[dict] = []
    for n_q in n_query_list:
        if n_q not in baseline_per_person or n_q not in stress_per_person:
            continue
        bp = baseline_per_person[n_q]
        sp = stress_per_person[n_q]
        common = sorted(set(bp.keys()) & set(sp.keys()))
        if not common:
            out.append({"n_query": n_q, "skipped": True})
            continue
        delta = np.array([sp[p] - bp[p] for p in common], dtype=np.float64)
        n_boot = 1000
        n = len(common)
        boot = np.empty(n_boot)
        boot_gen = bootstrap_rng if bootstrap_rng is not None else rng
        for b in range(n_boot):
            sel = boot_gen.integers(0, n, size=n)
            boot[b] = delta[sel].mean()
        ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
        out.append({
            "n_query": n_q,
            "n_persons_paired": n,
            "delta_r1_mean": float(delta.mean()),
            "delta_r1_ci95_low": float(ci_low),
            "delta_r1_ci95_high": float(ci_high),
        })
    return out


# ---------------------------------------------------------------------------
# Full-registry sweep (deployment scenario) — also paired across slices
# ---------------------------------------------------------------------------

def _evaluate_against_full_registry_paired(
    per_person_embs: dict[str, np.ndarray],
    registry_index: RetrievalIndex,
    n_query_list: list[int],
    permutations: dict[int, list[dict[str, np.ndarray]]],
    rng: np.random.Generator,
    bootstrap_rng: np.random.Generator | None = None,
) -> list[dict]:
    """Query the full deployed registry. Trial permutations are SHARED with the symmetric sweep."""
    n_reg = len(registry_index)
    results: list[dict] = []
    for n_q in n_query_list:
        eligible = [pid for pid, e in per_person_embs.items() if len(e) >= n_q]
        if len(eligible) < 5:
            results.append({"n_query": n_q, "n_persons": len(eligible), "skipped": True})
            continue
        trial_perms = permutations.get(n_q, [])
        n_trials = max(1, len(trial_perms))

        n_eligible = len(eligible)
        match_r1 = np.zeros((n_trials, n_eligible), dtype=bool)
        match_r5 = np.zeros_like(match_r1)
        match_r10 = np.zeros_like(match_r1)
        sim_top1_arr = np.zeros((n_trials, n_eligible), dtype=np.float64)
        gap_top12_arr = np.zeros((n_trials, n_eligible), dtype=np.float64)

        for t in range(n_trials):
            perm = trial_perms[t] if trial_perms else None
            for j, pid in enumerate(eligible):
                arr = per_person_embs[pid]
                # Use the baseline permutation only if its tooth count matches
                # this slice's count for the same pid; otherwise draw fresh.
                # This is the same pairing-collapse fix as in the rotation
                # symmetric sweep — modulo would silently duplicate indices.
                if perm is not None and pid in perm and len(perm[pid]) == len(arr):
                    idx = perm[pid][:n_q]
                else:
                    idx = rng.permutation(len(arr))[:n_q]
                q = _mean_pool(arr[idx])
                sims, ids = registry_index.search(q, k=10)
                match_r1[t, j] = ids[0] == pid
                match_r5[t, j] = pid in ids[:5]
                match_r10[t, j] = pid in ids[:10]
                sim_top1_arr[t, j] = float(sims[0])
                gap_top12_arr[t, j] = float(sims[0] - sims[1]) if len(sims) > 1 else 1.0

        per_person_r1 = match_r1.mean(axis=0)
        rank1 = float(per_person_r1.mean())
        rank5 = float(match_r5.mean(axis=0).mean())
        rank10 = float(match_r10.mean(axis=0).mean())

        n_boot = 1000
        boot_r1 = np.empty(n_boot)
        boot_gen = bootstrap_rng if bootstrap_rng is not None else rng
        for b in range(n_boot):
            sel = boot_gen.integers(0, n_eligible, size=n_eligible)
            boot_r1[b] = per_person_r1[sel].mean()
        ci_low, ci_high = np.percentile(boot_r1, [2.5, 97.5])

        results.append({
            "n_query": n_q,
            "n_persons": n_eligible,
            "n_registry": n_reg,
            "n_trials": n_trials,
            "rank1_mean": rank1,
            "rank1_ci95_low": float(ci_low),
            "rank1_ci95_high": float(ci_high),
            "rank5_mean": rank5,
            "rank10_mean": rank10,
            "sim_top1_median": float(np.median(sim_top1_arr)),
            "gap_top12_median": float(np.median(gap_top12_arr)),
            "per_person_r1": per_person_r1.tolist(),
            "pids": eligible,
        })
    return results


# ---------------------------------------------------------------------------
# Per-FDI breakdown — Phase 8.0 plan deliverable
# ---------------------------------------------------------------------------

def _per_fdi_breakdown(
    stage_outputs: dict[str, StageACOutput],
    per_person_embs: dict[str, np.ndarray],
    registry_index: RetrievalIndex,
) -> list[dict]:
    """For each FDI, query the single tooth alone against the full registry.

    The query is the single L2-normalized tooth embedding (no aggregation).
    Reports per-FDI R1/R5 with N support.
    """
    by_fdi: dict[str, list[tuple[str, np.ndarray]]] = {}
    for pid, embs in per_person_embs.items():
        st = stage_outputs.get(pid)
        if st is None:
            continue
        for i, fdi in enumerate(st.fdi_labels):
            by_fdi.setdefault(fdi, []).append((pid, embs[i]))

    out: list[dict] = []
    for fdi in sorted(by_fdi.keys()):
        items = by_fdi[fdi]
        if len(items) < 15:  # plan rule: suppress FDIs with n<15
            continue
        n_correct_r1 = 0
        n_correct_r5 = 0
        for pid, emb in items:
            q = _l2_normalize(emb)
            _, ids = registry_index.search(q, k=5)
            n_correct_r1 += int(ids[0] == pid)
            n_correct_r5 += int(pid in ids[:5])
        out.append({
            "fdi": fdi,
            "n": len(items),
            "rank1": n_correct_r1 / len(items),
            "rank5": n_correct_r5 / len(items),
        })
    return out


# ---------------------------------------------------------------------------
# Held-out enrolment slice
# ---------------------------------------------------------------------------

def _rebuild_index_without(full_index: RetrievalIndex, drop_ids: set[str]) -> RetrievalIndex:
    new_index = RetrievalIndex(dim=full_index.dim)
    vecs, ids = [], []
    for i, pid in enumerate(full_index.person_ids):
        if pid in drop_ids:
            continue
        vecs.append(full_index.index.reconstruct(i))
        ids.append(pid)
    if not vecs:
        return new_index
    arr = np.stack(vecs).astype(np.float32)
    new_index.add(arr, ids)
    return new_index


def evaluate_heldout_enrolment(
    per_person_embs: dict[str, np.ndarray],
    full_registry: RetrievalIndex,
    n_holdout: int,
    n_trials: int,
    rng: np.random.Generator,
    bootstrap_rng: np.random.Generator | None = None,
    n_query: int = 16,
) -> dict:
    """Drop n_holdout test persons from the full registry per trial, record per-query signals.

    Returns a flat records list (`records`) with one row per (pid, trial), labelled
    "oos" or "in_registry". Phase 8.6 will compute AUROC + person-stratified bootstrap
    from these records directly.
    """
    test_pids = list(per_person_embs.keys())
    if len(test_pids) < n_holdout + 5:
        reason = f"not enough usable test persons ({len(test_pids)}) for holdout={n_holdout} (need >={n_holdout + 5})"
        print(f"  [heldout_enrol] SKIPPED: {reason}")
        return {"skipped": True, "reason": reason, "n_usable_persons": len(test_pids)}

    records: list[dict] = []
    in_r1_per_trial: list[float] = []
    for trial in range(n_trials):
        held = set(rng.choice(test_pids, size=n_holdout, replace=False))
        sub_index = _rebuild_index_without(full_registry, held)

        trial_in_r1: list[bool] = []
        for pid in test_pids:
            arr = per_person_embs[pid]
            if len(arr) < 1:
                continue
            n_eff = min(n_query, len(arr))
            idx = rng.permutation(len(arr))[:n_eff]
            q = _mean_pool(arr[idx])
            sims, ids = sub_index.search(q, k=2)
            label = "oos" if pid in held else "in_registry"
            sim_top1 = float(sims[0])
            gap = float(sims[0] - sims[1]) if len(sims) > 1 else 1.0
            records.append({
                "pid": pid,
                "trial": trial,
                "label": label,
                "n_query": n_eff,
                "sim_top1": sim_top1,
                "gap_top12": gap,
                "top1_pid": ids[0],
                "registry_size": len(sub_index),
            })
            if label == "in_registry":
                trial_in_r1.append(ids[0] == pid)
        if trial_in_r1:
            in_r1_per_trial.append(float(np.mean(trial_in_r1)))

    # Bootstrap CI for in-registry R1
    in_r1_arr = np.array(in_r1_per_trial)
    if len(in_r1_arr):
        n_boot = 1000
        boot = np.empty(n_boot)
        n = len(in_r1_arr)
        boot_gen = bootstrap_rng if bootstrap_rng is not None else rng
        for b in range(n_boot):
            sel = boot_gen.integers(0, n, size=n)
            boot[b] = in_r1_arr[sel].mean()
        in_r1_ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
        in_r1_mean = float(in_r1_arr.mean())
    else:
        in_r1_ci = (None, None)
        in_r1_mean = None

    # Distribution summaries
    oos_sim = np.array([r["sim_top1"] for r in records if r["label"] == "oos"])
    in_sim = np.array([r["sim_top1"] for r in records if r["label"] == "in_registry"])
    oos_gap = np.array([r["gap_top12"] for r in records if r["label"] == "oos"])
    in_gap = np.array([r["gap_top12"] for r in records if r["label"] == "in_registry"])

    def _summ(a: np.ndarray) -> dict:
        if not len(a):
            return {}
        return {
            "mean": float(a.mean()), "median": float(np.median(a)),
            "std": float(a.std()),
            "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90)),
        }

    return {
        "n_holdout": n_holdout,
        "n_trials": n_trials,
        "n_query": n_query,
        "in_registry_r1_mean": in_r1_mean,
        "in_registry_r1_ci95_low": in_r1_ci[0],
        "in_registry_r1_ci95_high": in_r1_ci[1],
        "oos_sim_top1": _summ(oos_sim),
        "in_sim_top1": _summ(in_sim),
        "oos_gap_top12": _summ(oos_gap),
        "in_gap_top12": _summ(in_gap),
        "records": records,
    }


# ---------------------------------------------------------------------------
# Sanity checks (catch silent failures before the full run)
# ---------------------------------------------------------------------------

def _sanity_check_polygon_rotation() -> None:
    """Assert that _rotate_polygon matches PIL's actual rotation of a dot."""
    src_w, src_h = 600, 400
    img = Image.new("L", (src_w, src_h), 0)
    # A 3x3 white dot at (450, 250)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            img.putpixel((450 + dx, 250 + dy), 255)
    deg = 30.0
    rot = img.rotate(deg, resample=Image.BILINEAR, expand=False, fillcolor=0)
    arr = np.asarray(rot)
    ys, xs = np.where(arr > 100)
    if len(xs) == 0:
        raise RuntimeError("polygon-rotation sanity: PIL rotated the dot off-canvas")
    empirical = np.array([xs.mean(), ys.mean()])
    predicted = _rotate_polygon(
        np.array([[450.0, 250.0]], dtype=np.float32),
        deg, (src_w, src_h), (src_w, src_h),
    )[0]
    err = np.linalg.norm(empirical - predicted)
    if err > 2.0:
        raise RuntimeError(
            f"polygon-rotation sanity FAILED: empirical={empirical}, "
            f"predicted={predicted}, err={err:.2f}px (expected <2px)"
        )
    print(f"  [sanity] polygon-rotation OK: err={err:.2f}px at {deg}°")


def _sanity_check_registry_overlap(
    test_pids: list[str], registry_index: RetrievalIndex,
) -> None:
    """Assert that test pids actually appear in the registry. R1=0 silently is the bug."""
    overlap = set(test_pids) & set(registry_index.person_ids)
    print(f"  [sanity] test/registry pid overlap: {len(overlap)} / {len(test_pids)} test pids enrolled")
    if not overlap:
        raise RuntimeError(
            f"sanity: NO test pids overlap with registry — "
            f"test_pids sample={test_pids[:2]}, registry sample={registry_index.person_ids[:2]}"
        )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _build_test_panoramic_list(manifest_path: Path) -> list[tuple[str, str, Path]]:
    df = pd.read_csv(manifest_path, dtype=str)
    test = df[df["split"] == "test"].drop_duplicates("person_id")
    out: list[tuple[str, str, Path]] = []
    for _, row in test.iterrows():
        image_id = row["image_id"]
        pano_path = PROJECT_ROOT / "dataset_raw" / image_id / f"{image_id}.png"
        if pano_path.exists():
            out.append((row["person_id"], image_id, pano_path))
    return out


def _aggregate_yolo_mask_metrics(
    stage_outputs: dict[str, StageACOutput],
    angles: dict[str, float] | None,
    buckets: list[tuple[float, float]],
) -> list[dict]:
    """Group YOLO mask-mAP by |rotation_deg| bucket."""
    out = []
    for lo, hi in buckets:
        ious = []
        recs = []
        precs = []
        n = 0
        for pid, st in stage_outputs.items():
            if st.yolo_mask_iou_vs_gt is None:
                continue
            a = abs(angles[pid]) if angles and pid in angles else abs(st.rotation_deg)
            if lo <= a < hi or (hi == buckets[-1][1] and a == hi):
                ious.append(st.yolo_mask_iou_vs_gt)
                recs.append(st.yolo_mask_recall_vs_gt or 0.0)
                precs.append(st.yolo_mask_precision_vs_gt or 0.0)
                n += 1
        if n == 0:
            continue
        out.append({
            "abs_angle_bucket": [lo, hi],
            "n": n,
            "mean_iou": float(np.mean(ious)),
            "mean_recall_iou50": float(np.mean(recs)),
            "mean_precision_iou50": float(np.mean(precs)),
        })
    return out


def _extract_for_split(
    label: str,
    test_persons: list[tuple[str, str, Path, float]],
    cache: PipelineCache,
    models: PipelineModels,
    load_gt: bool,
) -> tuple[dict[str, np.ndarray], dict[str, StageACOutput], int]:
    """Extract stage A/C + embeddings for each panoramic with its per-person angle."""
    per_person: dict[str, np.ndarray] = {}
    stage_outputs: dict[str, StageACOutput] = {}
    n_failed = 0
    t0 = time.perf_counter()
    for i, (pid, image_id, pano_path, angle) in enumerate(test_persons):
        gt_polys = load_gt_polygons(image_id) if load_gt else {}
        st = cache.get_stage_ac(models, pano_path, pid, image_id, angle, gt_polys)
        if st is None:
            n_failed += 1
            continue
        embs = cache.get_embeddings(models, st)
        if embs is None or len(embs) == 0:
            n_failed += 1
            continue
        per_person[pid] = embs
        stage_outputs[pid] = st
        if (i + 1) % 25 == 0 or i == len(test_persons) - 1:
            print(f"  [{label}] {i + 1}/{len(test_persons)} done "
                  f"(failed={n_failed}, {time.perf_counter() - t0:.1f}s)")
    return per_person, stage_outputs, n_failed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="identification/runs/phase8_baseline")
    parser.add_argument("--manifest", default="identification/data/manifest_clean.csv")
    parser.add_argument("--n-query-list", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--rotation-deg", type=float, default=30.0)
    parser.add_argument("--heldout-count", type=int, default=30)
    parser.add_argument("--heldout-trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-rotation", action="store_true")
    parser.add_argument("--skip-heldout", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional: cap test persons for a smoke test.")
    args = parser.parse_args()

    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = PipelineConfig()
    models = PipelineModels(config=config)
    print("Loading pipeline...")
    models.load_all()

    yolo_hash = _file_hash(config.yolo_seg_weights)
    fdi_hash = _file_hash(config.fdi_classifier)
    embedder_hash = _file_hash(config.embedder)
    print(f"YOLO seg hash: {yolo_hash}, FDI hash: {fdi_hash}, embedder hash: {embedder_hash}")

    test_persons_all = _build_test_panoramic_list(PROJECT_ROOT / args.manifest)
    if args.limit:
        test_persons_all = test_persons_all[: args.limit]
    print(f"Test panoramics available on disk: {len(test_persons_all)}")

    print("Running sanity checks...")
    _sanity_check_polygon_rotation()
    _sanity_check_registry_overlap(
        [p for p, _, _ in test_persons_all], models.registry_index,
    )

    cache = PipelineCache(
        output_dir=output_dir,
        yolo_hash=yolo_hash,
        fdi_hash=fdi_hash,
        embedder_hash=embedder_hash,
        crop_size=config.crop_size,
        yolo_conf=config.yolo_conf,
        yolo_iou=config.yolo_iou,
        yolo_imgsz=config.yolo_imgsz,
        scratch_dir=config.temp_dir / "phase8_eval_scratch",
    )

    # Independent RNGs per slice so --skip-* is reproducible. We deliberately
    # spawn a single bootstrap_rng that's used wherever we resample with
    # replacement, so the bootstrap CIs are independent of the order in which
    # the slice RNGs get consumed.
    seed_root = np.random.SeedSequence(args.seed)
    baseline_rng, rotation_rng, heldout_rng, angle_rng, bootstrap_rng = (
        np.random.default_rng(s) for s in seed_root.spawn(5)
    )

    payloads: dict[str, dict] = {}

    # --- Baseline (upright) ---
    baseline_per_person_r1: dict[int, dict[str, float]] = {}
    baseline_permutations: dict[int, list[dict[str, np.ndarray]]] = {}
    if not args.skip_baseline:
        print("\n[baseline] extracting Stage A/C + embeddings (upright)...")
        upright_persons = [(p, i, path, 0.0) for p, i, path in test_persons_all]
        per_person, stage_outputs, n_failed = _extract_for_split(
            "baseline", upright_persons, cache, models, load_gt=True,
        )
        print(f"[baseline] usable: {len(per_person)}, failed: {n_failed}")

        print("[baseline] symmetric sweep...")
        sweep_sym, baseline_permutations = _evaluate_sweep_symmetric_paired(
            per_person, args.n_query_list, args.n_trials, baseline_rng, bootstrap_rng,
        )
        for s in sweep_sym:
            if s.get("skipped"):
                continue
            print(f"  sym n={s['n_query']:>2}: R1={s['rank1_mean']:.4f} "
                  f"[{s['rank1_ci95_low']:.3f}, {s['rank1_ci95_high']:.3f}], "
                  f"R5={s['rank5_mean']:.4f}, mAP={s['mAP_mean']:.4f}")

        print("[baseline] full-registry sweep (deployment scenario)...")
        sweep_reg = _evaluate_against_full_registry_paired(
            per_person, models.registry_index, args.n_query_list,
            baseline_permutations, baseline_rng, bootstrap_rng,
        )
        for s in sweep_reg:
            if s.get("skipped"):
                continue
            print(f"  reg n={s['n_query']:>2}: R1={s['rank1_mean']:.4f} "
                  f"[{s['rank1_ci95_low']:.3f}, {s['rank1_ci95_high']:.3f}], "
                  f"sim_med={s['sim_top1_median']:.3f}")

        print("[baseline] per-FDI breakdown (single-tooth, full registry)...")
        per_fdi = _per_fdi_breakdown(stage_outputs, per_person, models.registry_index)
        for r in per_fdi:
            print(f"  FDI {r['fdi']:>2}: n={r['n']:>3}  R1={r['rank1']:.3f}  R5={r['rank5']:.3f}")

        # Cache per-person R1 for paired-difference with rotation slice
        for s in sweep_sym:
            if s.get("skipped"):
                continue
            baseline_per_person_r1[s["n_query"]] = {
                pid: r1 for pid, r1 in zip(s["pids"], s["per_person_r1"])
            }

        yolo_mask = _aggregate_yolo_mask_metrics(
            stage_outputs, angles=None,
            buckets=[(0.0, 0.001)],  # single bucket for upright
        )
        baseline_payload = {
            "label": "yolo_eval",
            "rotation_deg": 0.0,
            "n_persons_attempted": len(upright_persons),
            "n_persons_usable": len(per_person),
            "n_persons_failed": n_failed,
            "mean_teeth_per_person": float(np.mean([len(e) for e in per_person.values()])) if per_person else 0.0,
            "yolo_checkpoint_hash": yolo_hash,
            "fdi_classifier_hash": fdi_hash,
            "embedder_hash": embedder_hash,
            "embedder_checkpoint": str(config.embedder.relative_to(PROJECT_ROOT)),
            "registry_dir": str(config.registry_dir.relative_to(PROJECT_ROOT)),
            "registry_size": len(models.registry_index),
            "n_query_list": args.n_query_list,
            "n_trials": args.n_trials,
            "sweep_symmetric": sweep_sym,
            "sweep_full_registry": sweep_reg,
            "per_fdi": per_fdi,
            "yolo_mask_vs_gt": yolo_mask,
        }
        with open(output_dir / "yolo_eval.json", "w") as f:
            json.dump(baseline_payload, f, indent=2)
        payloads["baseline"] = baseline_payload
        print(f"[baseline] saved → {output_dir/'yolo_eval.json'}")

    # --- Rotation-stress slice (paired to baseline by permutations) ---
    if not args.skip_rotation:
        print(f"\n[rotation_stress] drawing per-person angles in ±{args.rotation_deg:.0f}°...")
        rotated_persons: list[tuple[str, str, Path, float]] = []
        angle_by_pid: dict[str, float] = {}
        for pid, image_id, pano_path in test_persons_all:
            angle = float(angle_rng.uniform(-args.rotation_deg, args.rotation_deg))
            rotated_persons.append((pid, image_id, pano_path, angle))
            angle_by_pid[pid] = angle

        print("[rotation_stress] extracting Stage A/C + embeddings (rotated)...")
        per_person, stage_outputs, n_failed = _extract_for_split(
            "rotation_stress", rotated_persons, cache, models, load_gt=True,
        )

        print("[rotation_stress] symmetric sweep (REUSING baseline permutations for pairing)...")
        sweep_sym_rot: list[dict] = []
        per_person_r1_rot: dict[int, dict[str, float]] = {}
        for n_q in args.n_query_list:
            base_perms = baseline_permutations.get(n_q, []) if not args.skip_baseline else []
            # Persons that survive in BOTH baseline and rotation, with enough teeth in rotation
            eligible_pids = [
                pid for pid in per_person.keys()
                if len(per_person[pid]) >= n_q + 1
            ]
            if not base_perms:
                # No baseline this run — draw fresh
                base_perms = _draw_paired_subsets(per_person, eligible_pids, n_q, args.n_trials, rotation_rng)
            if len(eligible_pids) < 5:
                sweep_sym_rot.append({"n_query": n_q, "n_persons": len(eligible_pids), "skipped": True})
                continue

            n_trials_eff = len(base_perms)
            match_r1 = np.zeros((n_trials_eff, len(eligible_pids)), dtype=bool)
            match_r5 = np.zeros_like(match_r1)

            # Track which pids are genuinely paired (same tooth count as baseline) vs
            # had to be re-drawn freshly because the rotation slice produced a different
            # number of teeth. Paired-diff bootstrap will use only the truly-paired subset.
            truly_paired: set[str] = set()
            for t, perm in enumerate(base_perms):
                queries, galleries, used = [], [], []
                for pid in eligible_pids:
                    arr = per_person[pid]
                    if pid in perm and len(perm[pid]) == len(arr):
                        # Tooth count matches baseline → reuse the same permutation index-for-index
                        idx = perm[pid]
                        q_idx = idx[:n_q]
                        g_idx = idx[n_q:]
                        truly_paired.add(pid)
                    else:
                        # YOLO under rotation gave a different tooth count for this pid.
                        # Fall back to a fresh draw; this pid is NOT paired with baseline.
                        sh = rotation_rng.permutation(len(arr))
                        q_idx = sh[:n_q]
                        g_idx = sh[n_q:]
                    if len(g_idx) == 0:
                        continue
                    queries.append(_mean_pool(arr[q_idx]))
                    galleries.append(_mean_pool(arr[g_idx]))
                    used.append(pid)
                if not used:
                    continue
                Q = np.stack(queries)
                G = np.stack(galleries)
                if not (np.isfinite(Q).all() and np.isfinite(G).all()):
                    raise RuntimeError(
                        f"rot sym sweep n_q={n_q} trial={t}: non-finite embeddings"
                    )
                sim = Q @ G.T
                if not np.isfinite(sim).all():
                    raise RuntimeError(
                        f"rot sym sweep n_q={n_q} trial={t}: non-finite sim matrix"
                    )
                ranked = np.argsort(-sim, axis=1)
                pids_arr = np.array(used)
                ranked_labels = pids_arr[ranked]
                mat = ranked_labels == pids_arr[:, None]
                # Map back to eligible_pids index
                idx_map = {pid: k for k, pid in enumerate(eligible_pids)}
                for u_i, pid in enumerate(used):
                    k = idx_map[pid]
                    match_r1[t, k] = mat[u_i, 0]
                    match_r5[t, k] = mat[u_i, :5].any()

            per_person_r1 = match_r1.mean(axis=0)
            per_person_r5 = match_r5.mean(axis=0)
            rank1 = float(per_person_r1.mean())
            rank5 = float(per_person_r5.mean())

            n_boot = 1000
            n_e = len(eligible_pids)
            boot_r1 = np.empty(n_boot)
            for b in range(n_boot):
                sel = bootstrap_rng.integers(0, n_e, size=n_e)
                boot_r1[b] = per_person_r1[sel].mean()
            ci_low, ci_high = np.percentile(boot_r1, [2.5, 97.5])

            sweep_sym_rot.append({
                "n_query": n_q,
                "method": "mean",
                "n_persons": n_e,
                "n_persons_truly_paired": len(truly_paired),
                "n_trials": n_trials_eff,
                "rank1_mean": rank1,
                "rank1_ci95_low": float(ci_low),
                "rank1_ci95_high": float(ci_high),
                "rank5_mean": rank5,
                "per_person_r1": per_person_r1.tolist(),
                "pids": eligible_pids,
                "truly_paired_pids": sorted(truly_paired),
            })
            # Only include truly-paired pids in the per-person R1 map for paired-diff bootstrap.
            per_person_r1_rot[n_q] = {
                pid: r for pid, r in zip(eligible_pids, per_person_r1) if pid in truly_paired
            }
            print(f"  rot sym n={n_q:>2}: R1={rank1:.4f} [{ci_low:.3f}, {ci_high:.3f}]")

        # Full-registry sweep under rotation (deployment scenario)
        print("[rotation_stress] full-registry sweep under rotation...")
        sweep_reg_rot = _evaluate_against_full_registry_paired(
            per_person, models.registry_index, args.n_query_list,
            baseline_permutations, rotation_rng, bootstrap_rng,
        )
        for s in sweep_reg_rot:
            if s.get("skipped"):
                continue
            print(f"  rot reg n={s['n_query']:>2}: R1={s['rank1_mean']:.4f} "
                  f"[{s['rank1_ci95_low']:.3f}, {s['rank1_ci95_high']:.3f}]")

        # Paired difference vs baseline (truly-paired pids only)
        paired_diff = []
        if baseline_per_person_r1:
            paired_diff = _paired_diff_bootstrap(
                baseline_per_person_r1, per_person_r1_rot, args.n_query_list,
                rotation_rng, bootstrap_rng,
            )
            for d in paired_diff:
                if d.get("skipped"):
                    continue
                print(f"  PAIRED Δ n={d['n_query']:>2}: Δ R1 = {d['delta_r1_mean']:+.4f} "
                      f"[{d['delta_r1_ci95_low']:+.3f}, {d['delta_r1_ci95_high']:+.3f}]  "
                      f"(n_paired={d['n_persons_paired']})")

        # YOLO mask-mAP per absolute-angle bucket
        yolo_buckets = _aggregate_yolo_mask_metrics(
            stage_outputs, angle_by_pid,
            buckets=[(0.0, 5.0), (5.0, 15.0), (15.0, 25.0), (25.0, args.rotation_deg + 0.01)],
        )

        rot_payload = {
            "label": "rotation_stress",
            "rotation_deg_max": args.rotation_deg,
            "per_person_angle": angle_by_pid,
            "n_persons_attempted": len(rotated_persons),
            "n_persons_usable": len(per_person),
            "n_persons_failed": n_failed,
            "yolo_checkpoint_hash": yolo_hash,
            "n_query_list": args.n_query_list,
            "n_trials": args.n_trials,
            "sweep_symmetric_rotated": sweep_sym_rot,
            "sweep_full_registry_rotated": sweep_reg_rot,
            "paired_diff_vs_baseline": paired_diff,
            "yolo_mask_vs_gt_by_bucket": yolo_buckets,
        }
        with open(output_dir / "rotation_stress.json", "w") as f:
            json.dump(rot_payload, f, indent=2)
        payloads["rotation_stress"] = rot_payload
        print(f"[rotation_stress] saved → {output_dir/'rotation_stress.json'}")

    # --- Held-out enrolment slice ---
    if not args.skip_heldout:
        print("\n[heldout_enrol] reusing upright cached embeddings...")
        upright_persons = [(p, i, path, 0.0) for p, i, path in test_persons_all]
        per_person, _, _ = _extract_for_split(
            "heldout_enrol", upright_persons, cache, models, load_gt=False,
        )
        heldout = evaluate_heldout_enrolment(
            per_person, models.registry_index,
            n_holdout=args.heldout_count, n_trials=args.heldout_trials,
            rng=heldout_rng, bootstrap_rng=bootstrap_rng,
            n_query=max(args.n_query_list),
        )
        with open(output_dir / "heldout_enrol.json", "w") as f:
            json.dump(heldout, f, indent=2)
        if not heldout.get("skipped"):
            print(f"  in_registry R1: {heldout['in_registry_r1_mean']:.4f} "
                  f"[{heldout['in_registry_r1_ci95_low']:.3f}, {heldout['in_registry_r1_ci95_high']:.3f}]")
            print(f"  OOS  sim_top1 median: {heldout['oos_sim_top1']['median']:.3f}  "
                  f"p10-p90: [{heldout['oos_sim_top1']['p10']:.3f}, {heldout['oos_sim_top1']['p90']:.3f}]")
            print(f"  IN   sim_top1 median: {heldout['in_sim_top1']['median']:.3f}  "
                  f"p10-p90: [{heldout['in_sim_top1']['p10']:.3f}, {heldout['in_sim_top1']['p90']:.3f}]")
        print(f"[heldout_enrol] saved → {output_dir/'heldout_enrol.json'}")

    print("\nPhase 8.0 baseline complete.")


if __name__ == "__main__":
    main()
