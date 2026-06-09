"""End-to-end identification pipeline used by the demo backend.

Stages:
    A. YOLO detection on the uploaded panoramic           -> bboxes
    B. Crop each detected tooth, resize to 224x224         -> torch tensors
    C. FDI classifier predicts the tooth number            -> per-crop FDI
    D. Embedding model produces a 128-d embedding per tooth
    E. Mean-pool embeddings + L2 normalize                 -> 128-d query vector
    F. FAISS search against the registry                   -> top-K persons
    G. Assemble results with fake names                    -> JSON payload

The pipeline is implemented as an async generator that yields stage events
suitable for streaming to the frontend over SSE.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from backend.visualization import (
    render_fdi_overlay,
    render_segmentation_overlay,
    render_yolo_overlay,
)
from identification.data.tooth_dataset import IMAGENET_MEAN, IMAGENET_STD
from identification.evaluation.evaluate_embedding import load_checkpoint as load_embedder_checkpoint
from identification.models.classifier import ToothClassifier
from identification.models.embedding_model import ToothEmbeddingModelWithMetadata
from identification.models.retrieval_index import RetrievalIndex

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _default_registry_dir() -> Path:
    """Phase 9.1 — default to the YOLO-built registry to match the Phase 8.0
    canonical baseline (R1 = 82.6% [79.8, 86.0]). Honour DEMO_USE_YOLO_REGISTRY=0
    for one-line rollback to the legacy GT-built registry (R1 ≈ 57.3%).
    """
    if os.environ.get("DEMO_USE_YOLO_REGISTRY", "1") == "0":
        return PROJECT_ROOT / "identification/registry"
    return PROJECT_ROOT / "identification/registry_ensemble_yolo/embedding_fdi_init_v1"


@dataclass
class PipelineConfig:
    """Per-server configuration that drives the pipeline."""

    yolo_weights: Path = PROJECT_ROOT / "runs-detection/train3/weights/best.pt"
    yolo_seg_weights: Path = PROJECT_ROOT / "runs-segmentation/default-seg/weights/best.pt"
    fdi_classifier: Path = PROJECT_ROOT / "identification/runs/tooth_fdi_raw/best.pt"
    embedder: Path = PROJECT_ROOT / "identification/runs/embedding_fdi_init_v1/best.pt"
    registry_dir: Path = field(default_factory=_default_registry_dir)
    # Ensemble: per-model checkpoints + per-model registry directories. The
    # backend loads each one and ensembles cosine similarities at search time.
    ensemble_checkpoints: dict[str, Path] = field(default_factory=lambda: {
        "baseline": PROJECT_ROOT / "identification/runs/embedding_triplet_v1/best.pt",
        "masked":   PROJECT_ROOT / "identification/runs/embedding_triplet_masked_v1/best.pt",
        "metadata": PROJECT_ROOT / "identification/runs/embedding_metadata_v1/best.pt",
        "fdi_init": PROJECT_ROOT / "identification/runs/embedding_fdi_init_v1/best.pt",
    })
    # Ensemble registries are built from YOLO-extracted crops so the query
    # distribution matches what the demo feeds in at inference. See Phase 7.1
    # "Deployment caveat" in thesis_notes.md for the full story.
    ensemble_registry_dir: Path = PROJECT_ROOT / "identification/registry_ensemble_yolo"
    temp_dir: Path = PROJECT_ROOT / "backend/temp"
    yolo_conf: float = 0.25
    yolo_iou: float = 0.45
    yolo_imgsz: int = 640
    top_k: int = 5
    crop_size: int = 224
    min_teeth_warning: int = 4
    default_mode: str = "segmentation"  # or "detection"
    # On MPS the whole pipeline runs in ~1 second, which is too fast for the
    # user to see the stage overlays flash by. Enforce a minimum dwell time
    # per stage so each visualization is actually visible during the demo.
    min_stage_dwell_ms: float = 900.0

    def device(self) -> str:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"


@dataclass
class PipelineModels:
    """Lazy-loaded models, kept in memory across requests."""

    config: PipelineConfig
    yolo: Any = None
    yolo_seg: Any = None
    fdi_classifier: ToothClassifier | None = None
    fdi_label_inv: dict[int, str] = field(default_factory=dict)
    embedder: torch.nn.Module | None = None
    embedder_uses_metadata: bool = False
    embedder_fdi_label_map: dict[str, int] = field(default_factory=dict)
    registry_index: RetrievalIndex | None = None
    registry_meta: dict[str, dict] = field(default_factory=dict)
    # Phase 9.2 — Phase 8.6 open-set calibration (locked) + provenance hash index.
    open_set_calibration: dict | None = None
    panoramic_sha256_to_pid: dict[str, str] = field(default_factory=dict)
    # Ensemble: parallel arrays of
    # (name, model, uses_metadata, fdi_label_map, crop_mode)
    ensemble_models: list[tuple[str, torch.nn.Module, bool, dict[str, int], str]] = field(
        default_factory=list
    )
    ensemble_indexes: list[RetrievalIndex] = field(default_factory=list)
    device: str = "cpu"

    def load_all(self) -> None:
        self.device = self.config.device()
        print(f"[pipeline] device={self.device}")

        # 1. YOLO detector and segmenter — both loaded so the user can switch
        # modes per query without paying a reload.
        from ultralytics import YOLO  # local import to avoid loading until needed
        print(f"[pipeline] loading YOLO detector from {self.config.yolo_weights}")
        self.yolo = YOLO(str(self.config.yolo_weights))
        print(f"[pipeline] loading YOLO segmenter from {self.config.yolo_seg_weights}")
        self.yolo_seg = YOLO(str(self.config.yolo_seg_weights))

        # 2. FDI classifier
        print(f"[pipeline] loading FDI classifier from {self.config.fdi_classifier}")
        ckpt = torch.load(self.config.fdi_classifier, map_location=self.device, weights_only=False)
        label_map = ckpt["label_map"]
        cfg = ckpt["config"]
        self.fdi_classifier = ToothClassifier(
            num_classes=len(label_map),
            pretrained=False,
            dropout=cfg["model"].get("dropout", 0.2),
        )
        self.fdi_classifier.load_state_dict(ckpt["model_state_dict"])
        self.fdi_classifier.to(self.device).eval()
        self.fdi_label_inv = {v: k for k, v in label_map.items()}

        # 3. Embedder
        print(f"[pipeline] loading embedder from {self.config.embedder}")
        embedder, embedder_cfg, _, embedder_ckpt = load_embedder_checkpoint(
            str(self.config.embedder), self.device
        )
        self.embedder = embedder
        self.embedder_uses_metadata = isinstance(embedder, ToothEmbeddingModelWithMetadata)
        if self.embedder_uses_metadata:
            self.embedder_fdi_label_map = embedder_ckpt["fdi_label_map"]

        # 4. Registry
        print(f"[pipeline] loading registry from {self.config.registry_dir}")
        self.registry_index = RetrievalIndex.load(
            str(self.config.registry_dir / "index"),
            dim=embedder.projection_head.out_features,
        )
        with open(self.config.registry_dir / "registry_meta.json") as f:
            payload = json.load(f)
        self.registry_meta = {p["person_id"]: p for p in payload["persons"]}
        print(f"[pipeline] registry size: {len(self.registry_index)} persons")

        # 4b. Phase 8.6 open-set calibration — load the locked threshold + z-score
        # stats so the pipeline can emit a calibrated open-set decision per query.
        calib_path = PROJECT_ROOT / "identification/runs/phase8_open_set/phase8_open_set_calibration.json"
        if calib_path.exists():
            with open(calib_path) as f:
                self.open_set_calibration = json.load(f)
            thr = self.open_set_calibration["operating_point"]["threshold"]
            print(f"[pipeline] open-set calibration loaded (threshold z = {thr:.4f}, mode = {self.open_set_calibration['mode']})")
        else:
            print(f"[pipeline] WARN: open-set calibration not found at {calib_path}; open_set_decision will be 'unknown'")

        # 4c. Phase 9.2 provenance — precompute SHA-256 of every registry
        # panoramic so we can flag self-match queries (re-uploads of an enrolled
        # image) vs novel uploads in the API response.
        self.panoramic_sha256_to_pid = {}
        n_hashed = 0
        for pid, meta in self.registry_meta.items():
            rel_path = meta.get("panoramic_path")
            if not rel_path:
                continue
            full = PROJECT_ROOT / rel_path
            if not full.exists():
                continue
            try:
                h = hashlib.sha256(full.read_bytes()).hexdigest()
                self.panoramic_sha256_to_pid[h] = pid
                n_hashed += 1
            except OSError:
                continue
        print(f"[pipeline] provenance hash index: {n_hashed}/{len(self.registry_meta)} panoramics hashed")

        # 5. Ensemble — load every other embedder + its registry so the user
        # can flip between single-model and ensemble per query. If the YOLO-
        # aligned registries aren't built yet, we skip the ensemble (single
        # mode still works), so the demo degrades gracefully.
        first_registry_path = next(iter(self.config.ensemble_checkpoints.values()))
        first_subdir = self.config.ensemble_registry_dir / first_registry_path.parent.name
        if not (first_subdir / "index.faiss").exists():
            print(f"[pipeline] ensemble registries not found at {self.config.ensemble_registry_dir}; "
                  "ensemble mode disabled (single mode still works)")
            return

        for name, ckpt_path in self.config.ensemble_checkpoints.items():
            print(f"[pipeline] loading ensemble member '{name}' from {ckpt_path}")
            model, mcfg, _, ckpt = load_embedder_checkpoint(str(ckpt_path), self.device)
            uses_meta = isinstance(model, ToothEmbeddingModelWithMetadata)
            fdi_map = ckpt["fdi_label_map"] if uses_meta else {}
            crop_mode = mcfg.get("data", {}).get("crop_mode", "raw")
            self.ensemble_models.append((name, model, uses_meta, fdi_map, crop_mode))

            # Map checkpoint name → registry sub-dir built by build_registry.
            registry_subdir = self.config.ensemble_registry_dir / ckpt_path.parent.name
            index = RetrievalIndex.load(
                str(registry_subdir / "index"),
                dim=model.projection_head.out_features,
            )
            self.ensemble_indexes.append(index)
        print(f"[pipeline] ensemble members: {[m[0] for m in self.ensemble_models]}")


def _resize_with_padding(crop: Image.Image, size: int) -> Image.Image:
    """Match training-time crop pipeline: resize keeping aspect, pad to size×size with black.

    Mirrors `identification/scripts/extract_crops_gt.py::resize_with_padding`.
    """
    rgb = crop.convert("RGB")
    w, h = rgb.size
    if w <= 0 or h <= 0:
        return Image.new("RGB", (size, size), (0, 0, 0))
    scale = size / max(w, h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = rgb.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas


def _expand_bbox(bbox: tuple[float, float, float, float], image_size: tuple[int, int],
                  padding_ratio: float = 0.1) -> tuple[int, int, int, int]:
    """Expand a bbox by `padding_ratio` on each side, clamping to image bounds.

    Matches the GT crop extractor (`extract_crops_gt.py`, padding_ratio=0.1).
    """
    x1, y1, x2, y2 = bbox
    img_w, img_h = image_size
    w, h = x2 - x1, y2 - y1
    pad_x = w * padding_ratio
    pad_y = h * padding_ratio
    x1 = max(0, int(round(x1 - pad_x)))
    y1 = max(0, int(round(y1 - pad_y)))
    x2 = min(img_w, int(round(x2 + pad_x)))
    y2 = min(img_h, int(round(y2 + pad_y)))
    return x1, y1, x2, y2


def _to_tensor(crop: Image.Image, size: int, device: str) -> torch.Tensor:
    """Match the same crop+normalize pipeline used during training."""
    canvas = _resize_with_padding(crop, size)
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    arr = (arr - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().to(device)
    return tensor


async def _dwell(elapsed_ms: float, min_ms: float) -> None:
    """If a stage ran faster than `min_ms`, sleep so the user can see the overlay."""
    if min_ms <= 0:
        return
    remaining = (min_ms - elapsed_ms) / 1000.0
    if remaining > 0:
        await asyncio.sleep(remaining)


def _confidence_label(top1: float, top2: float) -> str:
    """Categorize result confidence based on top-1 similarity and the top-1/top-2 gap.

    Thresholds calibrated against the observed gap distribution on the full
    registry (20 test queries):
        min 0.0001 · p25 0.0008 · median 0.0016 · p75 0.0029 · max 0.0062
    With these tiers, ~25% of queries land in each of high / medium / uncertain.
    """
    gap = top1 - top2
    if top1 < 0.7:
        return "low"
    if gap >= 0.003:   # top quartile of observed gaps
        return "high"
    if gap >= 0.001:   # middle 50%
        return "medium"
    return "uncertain"  # bottom quartile — truly indistinguishable


def _open_set_score(top1_sim: float, calibration: dict | None) -> tuple[float | None, str]:
    """Phase 8.6 locked open-set scoring.

    Returns (z_scored_sim_top1, decision) where decision is one of:
        "in_registry"  — score >= threshold (probably enrolled)
        "rejected"     — score <  threshold (probably not enrolled)
        "unknown"      — calibration not loaded (fail-open: never rejects)

    The locked calibration's `mode` is `sim_top1_only` with weights `[1, 0, 0, 0, 0]`
    over z-scored features. The fallback rule fired during Phase 8.6 because
    the 5-feature LR did not lift AUROC by >= 0.030 over sim-only on val.
    """
    if not calibration:
        return None, "unknown"
    mu = calibration["zscore_mu"][0]
    sd = calibration["zscore_sd"][0]
    z = (top1_sim - mu) / sd if sd > 0 else 0.0
    weight = calibration["weights"][0]
    bias = calibration.get("bias", 0.0)
    score = weight * z + bias
    threshold = calibration["operating_point"]["threshold"]
    decision = "in_registry" if score >= threshold else "rejected"
    return float(score), decision


def _query_provenance(upload_path: Path, sha256_to_pid: dict[str, str]) -> tuple[str, str | None]:
    """Phase 9.2 — classify whether the uploaded query is a known registry image.

    Returns (provenance, expected_person_id) where provenance is one of:
        "self_match" — bytes match an enrolled panoramic; expected_person_id is set
        "novel"      — bytes do not match any enrolled panoramic
        "unknown"    — could not read the upload (filesystem error)

    Note that "self_match" means the *image* matches an enrolled image — the
    deployed dataset only has one panoramic per person, so this is also the
    "self" person. The "heldout" category (different image of an enrolled
    person) is structurally impossible on this dataset but reserved for the
    Phase 9.8 curated-OOS picks.
    """
    try:
        h = hashlib.sha256(upload_path.read_bytes()).hexdigest()
    except OSError:
        return "unknown", None
    pid = sha256_to_pid.get(h)
    if pid is None:
        return "novel", None
    return "self_match", pid


async def run_pipeline(
    panoramic_path: Path,
    query_id: str,
    models: PipelineModels,
    mode: str = "segmentation",
    ensemble: bool = False,
) -> AsyncGenerator[dict, None]:
    """Stream pipeline events as the query is processed.

    Each yielded dict is `{"event": <name>, "data": <json-serializable>}` and is
    converted to an SSE message by the caller.

    `mode` selects between two YOLO backends:
      * "detection" — runs the bbox-only detector; crops are taken from xyxy boxes.
      * "segmentation" — runs the seg model; crops use the tight bbox of each
        predicted mask (which more closely matches the GT-mask-based crops used
        during embedder training).

    `ensemble` switches between a single embedder (default, FDI-init) and a
    score-level ensemble of all four trained embedders (Phase 7.1).
    """
    cfg = models.config
    device = models.device
    query_dir = cfg.temp_dir / query_id
    query_dir.mkdir(parents=True, exist_ok=True)

    if mode not in ("detection", "segmentation"):
        mode = cfg.default_mode

    timings: dict[str, float] = {}
    polygons: list[np.ndarray] = []  # only populated in segmentation mode

    # --- Stage A: YOLO ---
    stage_name = "detect" if mode == "detection" else "segment"
    stage_message = (
        "Detecting teeth..." if mode == "detection" else "Segmenting teeth..."
    )
    t0 = time.perf_counter()
    yield {"event": "stage_start", "data": {"stage": stage_name, "message": stage_message, "mode": mode}}
    await asyncio.sleep(0)

    yolo_model = models.yolo if mode == "detection" else models.yolo_seg
    results = yolo_model.predict(
        source=str(panoramic_path),
        conf=cfg.yolo_conf,
        iou=cfg.yolo_iou,
        imgsz=cfg.yolo_imgsz,
        verbose=False,
        device=device if device != "mps" else "mps",
    )
    if not results:
        yield {"event": "error", "data": {"message": "YOLO returned no results."}}
        return

    res0 = results[0]
    boxes = res0.boxes
    if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
        yield {"event": "error", "data": {"message": "No teeth found in this image. Is it a panoramic X-ray?"}}
        return

    bboxes = boxes.xyxy.cpu().numpy()  # (N, 4): x1, y1, x2, y2
    confidences = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(bboxes))

    if mode == "segmentation":
        masks_obj = getattr(res0, "masks", None)
        if masks_obj is None or masks_obj.xy is None or len(masks_obj.xy) == 0:
            yield {"event": "error", "data": {"message": "Segmenter found no tooth masks."}}
            return
        polygons = [np.asarray(p) for p in masks_obj.xy]
        # Replace YOLO's bbox with the tight bbox of each mask polygon — gives
        # crops closer to the GT-mask-bbox + 10% padding used during embedder
        # training.
        derived = []
        for poly in polygons:
            xs, ys = poly[:, 0], poly[:, 1]
            derived.append([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())])
        bboxes = np.asarray(derived)

    n_teeth = len(bboxes)

    overlay_filename = f"{stage_name}_overlay.png"
    overlay_path = query_dir / overlay_filename
    if mode == "segmentation":
        render_segmentation_overlay(panoramic_path, polygons, overlay_path)
    else:
        render_yolo_overlay(panoramic_path, bboxes.tolist(), overlay_path)
    timings[stage_name] = (time.perf_counter() - t0) * 1000

    yield {
        "event": "stage_complete",
        "data": {
            "stage": stage_name,
            "mode": mode,
            "n_teeth": int(n_teeth),
            "annotated_image_url": f"/api/intermediate/{query_id}/{overlay_filename}",
            "elapsed_ms": round(timings[stage_name], 1),
        },
    }
    await _dwell(timings[stage_name], cfg.min_stage_dwell_ms)

    if n_teeth < cfg.min_teeth_warning:
        yield {
            "event": "warning",
            "data": {
                "code": "few_teeth",
                "message": f"Found only {n_teeth} teeth — identification likely unreliable. Try another image.",
            },
        }

    # --- Stage B: Crops ---
    pano = Image.open(panoramic_path).convert("RGB")
    pano_arr = np.asarray(pano)  # (H, W, 3) uint8 — used for masked variants
    crops: list[Image.Image] = []
    masked_crops: list[Image.Image | None] = []
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        ex1, ey1, ex2, ey2 = _expand_bbox((float(x1), float(y1), float(x2), float(y2)),
                                            pano.size, padding_ratio=0.1)
        crops.append(pano.crop((ex1, ey1, ex2, ey2)))

        # Build masked variant when we have polygons (segmentation mode +
        # ensemble use it). In detection mode polygons is empty and the masked
        # crop falls back to the raw crop — fine because the masked ensemble
        # member then sees a degraded but reasonable input.
        if ensemble and polygons and i < len(polygons):
            mask_full = np.zeros(pano_arr.shape[:2], dtype=np.uint8)
            poly_int = polygons[i].astype(np.int32)
            try:
                import cv2
                cv2.fillPoly(mask_full, [poly_int], color=1)
            except Exception:
                # Fall back to PIL polygon fill if cv2 isn't available
                from PIL import ImageDraw as _ID
                mask_im = Image.new("L", pano.size, 0)
                _ID.Draw(mask_im).polygon(
                    [(float(p[0]), float(p[1])) for p in polygons[i]], fill=1,
                )
                mask_full = np.asarray(mask_im)
            region = pano_arr[ey1:ey2, ex1:ex2].copy()
            region_mask = mask_full[ey1:ey2, ex1:ex2]
            region[region_mask == 0] = 0
            masked_crops.append(Image.fromarray(region))
        else:
            masked_crops.append(None)

    # --- Stage C: FDI classification ---
    t0 = time.perf_counter()
    yield {"event": "stage_start", "data": {"stage": "fdi", "message": "Numbering teeth..."}}
    await asyncio.sleep(0)

    fdi_logits_all = []
    with torch.no_grad():
        for crop in crops:
            tensor = _to_tensor(crop, cfg.crop_size, device)
            logits = models.fdi_classifier(tensor)
            fdi_logits_all.append(logits)
    logits_tensor = torch.cat(fdi_logits_all, dim=0)
    probs = F.softmax(logits_tensor, dim=1).cpu().numpy()
    fdi_class_idx = probs.argmax(axis=1)
    fdi_confidences = probs.max(axis=1)
    fdi_labels = [models.fdi_label_inv[idx] for idx in fdi_class_idx]

    # Resolve FDI duplicates: two crops claiming the same tooth → keep higher-confidence one.
    seen: dict[str, int] = {}
    keep_indices: list[int] = []
    dropped: list[dict] = []
    for i, fdi in enumerate(fdi_labels):
        if fdi in seen and fdi_confidences[i] <= fdi_confidences[seen[fdi]]:
            dropped.append({"index": i, "fdi": fdi, "reason": "duplicate"})
            continue
        if fdi in seen:
            dropped.append({"index": seen[fdi], "fdi": fdi, "reason": "duplicate"})
            keep_indices.remove(seen[fdi])
        seen[fdi] = i
        keep_indices.append(i)

    keep_indices.sort()
    bboxes_kept = bboxes[keep_indices]
    fdi_kept = [fdi_labels[i] for i in keep_indices]
    fdi_conf_kept = [float(fdi_confidences[i]) for i in keep_indices]
    crops_kept = [crops[i] for i in keep_indices]
    masked_crops_kept = [masked_crops[i] for i in keep_indices]
    polygons_kept = (
        [polygons[i] for i in keep_indices] if polygons else None
    )

    n_uncertain = int(sum(1 for c in fdi_conf_kept if c < 0.5))

    fdi_overlay = query_dir / "fdi_overlay.png"
    render_fdi_overlay(
        panoramic_path,
        bboxes_kept.tolist(),
        fdi_kept,
        fdi_overlay,
        polygons=polygons_kept,
    )
    timings["fdi"] = (time.perf_counter() - t0) * 1000

    yield {
        "event": "stage_complete",
        "data": {
            "stage": "fdi",
            "n_teeth": len(crops_kept),
            "n_uncertain": n_uncertain,
            "n_dropped": len(dropped),
            "annotated_image_url": f"/api/intermediate/{query_id}/fdi_overlay.png",
            "elapsed_ms": round(timings["fdi"], 1),
        },
    }
    await _dwell(timings["fdi"], cfg.min_stage_dwell_ms)

    if not crops_kept:
        yield {"event": "error", "data": {"message": "No teeth survived FDI assignment."}}
        return

    if n_uncertain > 0 and n_uncertain / max(1, len(crops_kept)) >= 0.4:
        yield {
            "event": "warning",
            "data": {
                "code": "fdi_uncertain",
                "message": "Tooth numbering was uncertain; results may be less accurate.",
            },
        }

    # --- Stage D: Embeddings ---
    # In single mode we run one embedder per crop. In ensemble mode we run
    # every loaded embedder per crop and keep the per-model embedding arrays
    # separate; aggregation + search happen per-model and are combined at the
    # similarity-matrix level (Phase 7.1 score-level ensemble).
    t0 = time.perf_counter()
    yield {
        "event": "stage_start",
        "data": {
            "stage": "embed",
            "message": (
                "Generating ensemble embeddings..."
                if ensemble
                else "Generating embeddings..."
            ),
            "total": len(crops_kept),
        },
    }
    await asyncio.sleep(0)

    if ensemble:
        # Per-model embedding lists, parallel to crops_kept.
        per_model_embeddings: list[list[np.ndarray]] = [
            [] for _ in models.ensemble_models
        ]
        # Pre-compute raw and masked tensors once per tooth.
        with torch.no_grad():
            for i, (crop, fdi, masked_crop) in enumerate(
                zip(crops_kept, fdi_kept, masked_crops_kept)
            ):
                raw_tensor = _to_tensor(crop, cfg.crop_size, device)
                masked_tensor = (
                    _to_tensor(masked_crop, cfg.crop_size, device)
                    if masked_crop is not None
                    else raw_tensor  # detection mode: no polygons → use raw
                )
                for j, (name, model, uses_meta, fdi_map, crop_mode) in enumerate(
                    models.ensemble_models
                ):
                    tensor = masked_tensor if crop_mode == "masked" else raw_tensor
                    if uses_meta:
                        fdi_idx = fdi_map.get(fdi, 0)
                        fdi_t = torch.tensor([fdi_idx], dtype=torch.long, device=device)
                        emb = model(tensor, fdi_t)
                    else:
                        emb = model(tensor)
                    per_model_embeddings[j].append(emb.cpu().numpy()[0])
                if (i + 1) % 4 == 0 or i == len(crops_kept) - 1:
                    yield {
                        "event": "progress",
                        "data": {"stage": "embed", "current": i + 1, "total": len(crops_kept)},
                    }
                    await asyncio.sleep(0)
        ensemble_emb_arrays = [np.stack(lst) for lst in per_model_embeddings]
        # Use the FDI-init array (first non-baseline non-masked-non-metadata) as
        # the canonical per-tooth array for downstream things that expect one.
        # Find the fdi_init index if present; fall back to last.
        canonical_idx = next(
            (j for j, m in enumerate(models.ensemble_models) if m[0] == "fdi_init"),
            len(ensemble_emb_arrays) - 1,
        )
        embeddings_arr = ensemble_emb_arrays[canonical_idx]
    else:
        embeddings = []
        with torch.no_grad():
            for i, (crop, fdi) in enumerate(zip(crops_kept, fdi_kept)):
                tensor = _to_tensor(crop, cfg.crop_size, device)
                if models.embedder_uses_metadata:
                    fdi_idx = models.embedder_fdi_label_map.get(fdi)
                    if fdi_idx is None:
                        fdi_idx = 0  # fallback if FDI is unseen by metadata model
                    fdi_idx_tensor = torch.tensor([fdi_idx], dtype=torch.long, device=device)
                    emb = models.embedder(tensor, fdi_idx_tensor)
                else:
                    emb = models.embedder(tensor)
                embeddings.append(emb.cpu().numpy()[0])
                if (i + 1) % 4 == 0 or i == len(crops_kept) - 1:
                    yield {
                        "event": "progress",
                        "data": {"stage": "embed", "current": i + 1, "total": len(crops_kept)},
                    }
                    await asyncio.sleep(0)
        embeddings_arr = np.stack(embeddings)
        ensemble_emb_arrays = None  # type: ignore[assignment]

    timings["embed"] = (time.perf_counter() - t0) * 1000

    yield {
        "event": "stage_complete",
        "data": {
            "stage": "embed",
            "n_embeddings": int(len(embeddings_arr)),
            "elapsed_ms": round(timings["embed"], 1),
            "ensemble": ensemble,
            "ensemble_members": [m[0] for m in models.ensemble_models] if ensemble else None,
        },
    }
    await _dwell(timings["embed"], cfg.min_stage_dwell_ms)

    # --- Stage E: Aggregation (mean + L2) ---
    def _pool(arr: np.ndarray) -> np.ndarray:
        pooled = arr.mean(axis=0)
        n = np.linalg.norm(pooled)
        return (pooled / n if n > 1e-12 else pooled).astype(np.float32)

    if ensemble:
        per_model_query_vecs = [_pool(a) for a in ensemble_emb_arrays]  # type: ignore[arg-type]
    query_vec = _pool(embeddings_arr)

    # --- Stage F: FAISS search ---
    t0 = time.perf_counter()
    n_candidates = len(models.registry_index)
    yield {
        "event": "stage_start",
        "data": {
            "stage": "search",
            "message": (
                f"Searching {n_candidates} candidates with {len(models.ensemble_models)} models..."
                if ensemble
                else f"Searching {n_candidates} candidates..."
            ),
        },
    }
    # Give the user a brief moment to read the search status before results pop.
    await asyncio.sleep(0.4)

    if ensemble:
        # Mean of per-model similarity vectors over the full registry, then top-k.
        # Each per-model index is searched with k=n_candidates so we can average.
        sim_stack = []
        for j, index in enumerate(models.ensemble_indexes):
            sims_full, ids_full = index.search(per_model_query_vecs[j], k=n_candidates)
            # The ID order is the same across all 4 indexes (we verified during
            # build), so we can stack scores directly by reordering by person_id.
            # Build a position lookup from this model's index order.
            order = {pid: i for i, pid in enumerate(ids_full)}
            # Reorder to a canonical sequence (the first index's order)
            if j == 0:
                canonical_order = list(ids_full)
                sim_stack.append(sims_full)
            else:
                reordered = np.array(
                    [sims_full[order[pid]] for pid in canonical_order],
                    dtype=np.float64,
                )
                sim_stack.append(reordered)
        avg_sims = np.mean(np.stack(sim_stack, axis=0), axis=0)
        top_idx = np.argsort(-avg_sims)[: cfg.top_k]
        sims = avg_sims[top_idx]
        neighbor_ids = [canonical_order[i] for i in top_idx]
    else:
        sims, neighbor_ids = models.registry_index.search(query_vec, k=cfg.top_k)
    timings["search"] = (time.perf_counter() - t0) * 1000

    # --- Stage G: Assemble result ---
    results_list = []
    for rank, (sim, person_id) in enumerate(zip(sims, neighbor_ids)):
        meta = models.registry_meta.get(person_id, {})
        results_list.append({
            "rank": rank + 1,
            "person_id": person_id,
            "fake_name": meta.get("fake_name", person_id),
            "n_teeth": meta.get("n_teeth"),
            "similarity": float(sim),
        })

    top1_top2_gap = float(sims[0] - sims[1]) if len(sims) > 1 else 1.0
    confidence = _confidence_label(float(sims[0]), float(sims[1]) if len(sims) > 1 else 0.0)

    # Phase 9.2 — Phase 8.6 calibrated open-set decision + provenance.
    open_set_score, open_set_decision = _open_set_score(
        float(sims[0]), models.open_set_calibration,
    )
    query_provenance, expected_pid = _query_provenance(
        panoramic_path, models.panoramic_sha256_to_pid,
    )

    # Per-tooth contribution: dot each tooth's embedding against the top-1
    # gallery profile. In ensemble mode we average the per-model contributions.
    tooth_contributions: list[dict] = []
    try:
        top1_person = results_list[0]["person_id"]
        if ensemble:
            per_tooth_sims_acc = np.zeros(len(embeddings_arr), dtype=np.float64)
            for j, index in enumerate(models.ensemble_indexes):
                top1_idx = index.person_ids.index(top1_person)
                top1_vec = index.index.reconstruct(top1_idx)
                per_tooth_sims_acc += ensemble_emb_arrays[j] @ top1_vec  # type: ignore[index]
            per_tooth_sims = per_tooth_sims_acc / len(models.ensemble_indexes)
        else:
            top1_faiss_idx = models.registry_index.person_ids.index(top1_person)
            top1_vec = models.registry_index.index.reconstruct(top1_faiss_idx)
            per_tooth_sims = embeddings_arr @ top1_vec  # (n_teeth,)
        for i, fdi in enumerate(fdi_kept):
            tooth_contributions.append({
                "fdi": fdi,
                "fdi_confidence": fdi_conf_kept[i],
                "similarity_to_top1": float(per_tooth_sims[i]),
            })
        tooth_contributions.sort(key=lambda c: c["similarity_to_top1"], reverse=True)
    except Exception:
        tooth_contributions = []

    timings["total"] = sum(timings.values())

    yield {
        "event": "stage_complete",
        "data": {
            "stage": "search",
            "results": results_list,
            "confidence": confidence,
            "top1_top2_gap": top1_top2_gap,
            "timings_ms": {k: round(v, 1) for k, v in timings.items()},
            "n_query_teeth": int(len(embeddings_arr)),
            "n_dropped": len(dropped),
            "tooth_contributions": tooth_contributions,
            "ensemble": ensemble,
            "ensemble_members": [m[0] for m in models.ensemble_models] if ensemble else None,
            # Phase 9.2 — calibrated open-set + provenance (consumed by Phase 9.3 UI).
            "open_set_score": open_set_score,
            "open_set_decision": open_set_decision,
            "open_set_threshold": (
                models.open_set_calibration["operating_point"]["threshold"]
                if models.open_set_calibration else None
            ),
            "query_provenance": query_provenance,
            "expected_person_id": expected_pid,
        },
    }

    yield {"event": "done", "data": {}}


def cleanup_temp_dir(temp_dir: Path, max_age_seconds: int = 3600) -> int:
    """Delete query subdirectories older than `max_age_seconds`. Returns count."""
    if not temp_dir.exists():
        return 0
    now = time.time()
    removed = 0
    for child in temp_dir.iterdir():
        if not child.is_dir():
            continue
        if now - child.stat().st_mtime > max_age_seconds:
            for f in child.iterdir():
                try:
                    f.unlink()
                except OSError:
                    pass
            try:
                child.rmdir()
                removed += 1
            except OSError:
                pass
    return removed
