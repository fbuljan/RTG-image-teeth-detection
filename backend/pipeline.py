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
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from backend.visualization import render_fdi_overlay, render_yolo_overlay
from identification.data.tooth_dataset import IMAGENET_MEAN, IMAGENET_STD
from identification.evaluation.evaluate_embedding import load_checkpoint as load_embedder_checkpoint
from identification.models.classifier import ToothClassifier
from identification.models.embedding_model import ToothEmbeddingModelWithMetadata
from identification.models.retrieval_index import RetrievalIndex

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class PipelineConfig:
    """Per-server configuration that drives the pipeline."""

    yolo_weights: Path = PROJECT_ROOT / "runs-detection/train3/weights/best.pt"
    fdi_classifier: Path = PROJECT_ROOT / "identification/runs/tooth_fdi_raw/best.pt"
    embedder: Path = PROJECT_ROOT / "identification/runs/embedding_fdi_init_v1/best.pt"
    registry_dir: Path = PROJECT_ROOT / "identification/registry"
    temp_dir: Path = PROJECT_ROOT / "backend/temp"
    yolo_conf: float = 0.25
    yolo_iou: float = 0.45
    yolo_imgsz: int = 640
    top_k: int = 5
    crop_size: int = 224
    min_teeth_warning: int = 4
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
    fdi_classifier: ToothClassifier | None = None
    fdi_label_inv: dict[int, str] = field(default_factory=dict)
    embedder: torch.nn.Module | None = None
    embedder_uses_metadata: bool = False
    embedder_fdi_label_map: dict[str, int] = field(default_factory=dict)
    registry_index: RetrievalIndex | None = None
    registry_meta: dict[str, dict] = field(default_factory=dict)
    device: str = "cpu"

    def load_all(self) -> None:
        self.device = self.config.device()
        print(f"[pipeline] device={self.device}")

        # 1. YOLO detector
        from ultralytics import YOLO  # local import to avoid loading until needed
        print(f"[pipeline] loading YOLO from {self.config.yolo_weights}")
        self.yolo = YOLO(str(self.config.yolo_weights))

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


async def run_pipeline(
    panoramic_path: Path,
    query_id: str,
    models: PipelineModels,
) -> AsyncGenerator[dict, None]:
    """Stream pipeline events as the query is processed.

    Each yielded dict is `{"event": <name>, "data": <json-serializable>}` and is
    converted to an SSE message by the caller.
    """
    cfg = models.config
    device = models.device
    query_dir = cfg.temp_dir / query_id
    query_dir.mkdir(parents=True, exist_ok=True)

    timings: dict[str, float] = {}

    # --- Stage A: YOLO ---
    t0 = time.perf_counter()
    yield {"event": "stage_start", "data": {"stage": "yolo", "message": "Detecting teeth..."}}
    await asyncio.sleep(0)

    results = models.yolo.predict(
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

    boxes = results[0].boxes
    if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
        yield {"event": "error", "data": {"message": "No teeth found in this image. Is it a panoramic X-ray?"}}
        return

    bboxes = boxes.xyxy.cpu().numpy()  # (N, 4): x1, y1, x2, y2
    confidences = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(bboxes))
    n_teeth = len(bboxes)

    yolo_overlay = query_dir / "yolo_overlay.png"
    render_yolo_overlay(panoramic_path, bboxes.tolist(), yolo_overlay)
    timings["yolo"] = (time.perf_counter() - t0) * 1000

    yield {
        "event": "stage_complete",
        "data": {
            "stage": "yolo",
            "n_teeth": int(n_teeth),
            "annotated_image_url": f"/api/intermediate/{query_id}/yolo_overlay.png",
            "elapsed_ms": round(timings["yolo"], 1),
        },
    }
    await _dwell(timings["yolo"], cfg.min_stage_dwell_ms)

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
    crops: list[Image.Image] = []
    for x1, y1, x2, y2 in bboxes:
        ex1, ey1, ex2, ey2 = _expand_bbox((float(x1), float(y1), float(x2), float(y2)),
                                            pano.size, padding_ratio=0.1)
        crops.append(pano.crop((ex1, ey1, ex2, ey2)))

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

    n_uncertain = int(sum(1 for c in fdi_conf_kept if c < 0.5))

    fdi_overlay = query_dir / "fdi_overlay.png"
    render_fdi_overlay(panoramic_path, bboxes_kept.tolist(), fdi_kept, fdi_overlay)
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
    t0 = time.perf_counter()
    yield {
        "event": "stage_start",
        "data": {"stage": "embed", "message": "Generating embeddings...", "total": len(crops_kept)},
    }
    await asyncio.sleep(0)

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
    timings["embed"] = (time.perf_counter() - t0) * 1000

    yield {
        "event": "stage_complete",
        "data": {
            "stage": "embed",
            "n_embeddings": int(len(embeddings_arr)),
            "elapsed_ms": round(timings["embed"], 1),
        },
    }
    await _dwell(timings["embed"], cfg.min_stage_dwell_ms)

    # --- Stage E: Aggregation (mean + L2) ---
    pooled = embeddings_arr.mean(axis=0)
    norm = np.linalg.norm(pooled)
    if norm > 1e-12:
        pooled = pooled / norm
    query_vec = pooled.astype(np.float32)

    # --- Stage F: FAISS search ---
    t0 = time.perf_counter()
    yield {
        "event": "stage_start",
        "data": {
            "stage": "search",
            "message": f"Searching {len(models.registry_index)} candidates...",
        },
    }
    # Give the user a brief moment to read the search status before results pop.
    await asyncio.sleep(0.4)

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
