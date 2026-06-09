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
from identification.training.train_demographic_classifier import MLPHead as DemographicHead

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
    sessions_dir: Path = PROJECT_ROOT / "backend/sessions"
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
    # Phase 9.3 — sorted in_registry / oos sim_top1 arrays for percentile lookup.
    sim_top1_in_registry_sorted: np.ndarray | None = None
    sim_top1_oos_sorted: np.ndarray | None = None
    # Phase 9.4 — Phase 8.10 age head (sex head NOT wired; failed Pass).
    age_head: torch.nn.Module | None = None
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

        # 4b-bis. Phase 9.3 — load held-out enrolment records to build percentile
        # lookup tables. Lets us show "your sim_top1 is at the 73rd percentile
        # of correct identifications" instead of bare cosine.
        heldout_path = PROJECT_ROOT / "identification/runs/phase8_deployed_yolo_reg/heldout_enrol.json"
        if heldout_path.exists():
            with open(heldout_path) as f:
                heldout = json.load(f)
            in_reg_sims = sorted(
                r["sim_top1"] for r in heldout.get("records", [])
                if r.get("label") == "in_registry"
            )
            oos_sims = sorted(
                r["sim_top1"] for r in heldout.get("records", [])
                if r.get("label") == "oos"
            )
            self.sim_top1_in_registry_sorted = np.array(in_reg_sims, dtype=np.float64) if in_reg_sims else None
            self.sim_top1_oos_sorted = np.array(oos_sims, dtype=np.float64) if oos_sims else None
            print(f"[pipeline] percentile tables: in_registry={len(in_reg_sims)} oos={len(oos_sims)} sim_top1 values")
        else:
            print(f"[pipeline] WARN: heldout_enrol.json not found at {heldout_path}; percentile field will be null")

        # 4b-ter. Phase 9.4 — Phase 8.10 age regression head on the frozen
        # embedder. Sex head intentionally NOT loaded (Phase 8.10 pre-registered
        # rule: sex acc 0.556 ≈ chance baseline 0.539, fails the 0.65 marginal
        # floor; wiring would mislead users).
        age_head_path = PROJECT_ROOT / "identification/runs/demographic_v2/age_head.pt"
        if age_head_path.exists():
            try:
                head = DemographicHead(in_dim=128, hidden=64, dropout=0.3, out_dim=1)
                state = torch.load(age_head_path, map_location=self.device, weights_only=True)
                head.load_state_dict(state)
                head.to(self.device).eval()
                self.age_head = head
                print(f"[pipeline] Phase 8.10 age head loaded from {age_head_path}")
            except Exception as exc:  # noqa: BLE001 — degrade gracefully
                print(f"[pipeline] WARN: failed to load age head: {exc}")
        else:
            print(f"[pipeline] WARN: age head not found at {age_head_path}; age estimate will be null")

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


def _estimate_age(
    age_head,
    device,
    query_vec: np.ndarray,
    pool_size: int,
    open_set_decision: str,
) -> dict | None:
    """Phase 9.5.1 — honest age estimate.

    Gate emission on the open-set verdict (rejected queries get no age — the
    embedder is out of distribution, the head's output is meaningless). Drop
    the prediction-driven "in_dense_bucket" CI tightening from Phase 9.4 — the
    dense-bucket MAE in Phase 8.10 was conditioned on *ground-truth* age, and
    at inference we don't have the ground truth. A 17-year-old whose embedding
    saturates the head to 10.5y would otherwise inherit the tight CI and
    confident styling. Instead: always use a single conservative CI, clamp the
    *displayed* value to the training range [6, 18] with a saturation flag,
    and widen further on small pools (the head was trained on per-person mean
    embeddings — 1-tooth and 2-tooth fragments are an unvalidated pool-size
    shift on top of the GT→YOLO shift already disclosed).
    """
    if age_head is None or open_set_decision == "rejected":
        return None
    try:
        with torch.no_grad():
            q = torch.from_numpy(query_vec).unsqueeze(0).to(device)
            pred_raw = float(age_head(q).squeeze().item())
    except Exception:
        return None

    train_lo, train_hi = 6.0, 18.0
    # Saturation: either the head hit a training-range boundary, or the
    # ground-truth-conditional Phase 8.10 dense-MAE never applied here. We
    # treat anything outside [6, 13) as elevated risk since per-bucket MAE
    # climbed to 2.09y in 16-18y.
    saturated = pred_raw <= train_lo + 0.05 or pred_raw >= train_hi - 0.05
    elevated_risk = saturated or pred_raw < 6.0 or pred_raw >= 13.0
    small_pool = pool_size < 8

    # CI: 2.5y covers worst-bucket MAE (16-18y: 2.09y) on GT-mean embeddings
    # plus a GT→YOLO buffer. Small pools widen to 4.0y (no validated MAE).
    if small_pool:
        ci_half = 4.0
    elif elevated_risk:
        ci_half = 3.5
    else:
        ci_half = 2.5

    value_display = max(train_lo, min(train_hi, pred_raw))
    return {
        "value": pred_raw,
        "value_display": value_display,
        "ci_low": max(train_lo, value_display - ci_half),
        "ci_high": min(train_hi, value_display + ci_half),
        "ci_half": ci_half,
        "saturation_risk": elevated_risk,
        "pool_size": pool_size,
        "small_pool": small_pool,
        "training_range": [train_lo, train_hi],
    }


def _sim_top1_percentile(
    sim: float, sorted_arr: np.ndarray | None,
) -> float | None:
    """Phase 9.3 — what fraction of reference sim_top1 values are below this one?

    With `sorted_arr` = sorted in-registry sim_top1s (740 values from Phase 8.6
    held-out enrolment), the returned percentile answers "of all *correct*
    identifications in the test eval, what fraction had a lower sim than this
    query?" — a far more legible signal than raw cosine.
    """
    if sorted_arr is None or len(sorted_arr) == 0:
        return None
    # Position in sorted array (right-side gives strict-less-than count).
    idx = int(np.searchsorted(sorted_arr, sim, side="right"))
    return float(idx) / float(len(sorted_arr))


def _build_search_payload(
    query_vec: np.ndarray,
    fdi_used: list[str],
    fdi_conf_used: list[float],
    models: PipelineModels,
    config: PipelineConfig,
    panoramic_path: Path | None,
    sub_embs: np.ndarray | None = None,
) -> dict:
    """Phase 9.5 — shared search-stage payload builder for /api/identify and
    /api/search-fragment. Takes a pooled+L2-normalised query vector and returns
    the same JSON shape the SSE search event emits.

    `sub_embs` is the (n, D) array of L2-normed per-tooth embeddings that were
    pooled into `query_vec`. When supplied, per-tooth contributions are computed
    against the *new* top-1 — without it we leave contributions empty so the
    frontend cannot render the stale numbers from the parent /identify run.
    """
    sims, neighbor_ids = models.registry_index.search(query_vec, k=config.top_k)
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

    top1 = float(sims[0])
    top2 = float(sims[1]) if len(sims) > 1 else 0.0
    top1_top2_gap = top1 - top2 if len(sims) > 1 else 1.0

    open_set_score, open_set_decision = _open_set_score(top1, models.open_set_calibration)
    sim_top1_pct = _sim_top1_percentile(top1, models.sim_top1_in_registry_sorted)
    # Percentile is only well-defined for rank-1 — the reference distribution is
    # held-out *correct identifications* (rank-1 hits). Applying it to ranks
    # 2-5 silently relabels runners-up as "correct identifications observed at
    # similarity X," which the audit flagged as a category error. Emit null for
    # the non-top-1 ranks so the UI renders an em-dash, not a misleading number.
    for r in results_list:
        r["similarity_percentile"] = (
            _sim_top1_percentile(r["similarity"], models.sim_top1_in_registry_sorted)
            if r["rank"] == 1
            else None
        )

    # Provenance only makes sense when we have a panoramic_path (full /identify run).
    if panoramic_path is not None:
        query_provenance, expected_pid = _query_provenance(panoramic_path, models.panoramic_sha256_to_pid)
    else:
        query_provenance, expected_pid = "unknown", None

    # Per-tooth contribution to the *new* top-1, computed only when the caller
    # supplied the underlying per-tooth embeddings (fragment-search path). The
    # parent /identify run uses its own block further down with the full
    # embeddings_arr; passing sub_embs here lets fragment search emit a
    # contributions panel that actually matches the new top-1 it just selected
    # (audit fix — without this, the frontend retains the original /identify
    # contributions, which were dot products against the OLD top-1).
    tooth_contributions: list[dict] = []
    if sub_embs is not None and len(sub_embs) == len(fdi_used) and len(results_list) > 0:
        try:
            top1_person = results_list[0]["person_id"]
            top1_idx = models.registry_index.person_ids.index(top1_person)
            top1_vec = models.registry_index.index.reconstruct(top1_idx)
            per_tooth_sims = sub_embs @ top1_vec  # (n_teeth,)
            for i, fdi in enumerate(fdi_used):
                tooth_contributions.append({
                    "fdi": fdi,
                    "fdi_confidence": fdi_conf_used[i],
                    "similarity_to_top1": float(per_tooth_sims[i]),
                })
            tooth_contributions.sort(key=lambda c: c["similarity_to_top1"], reverse=True)
        except Exception:
            tooth_contributions = []

    # Age estimate — gated on open_set_decision and pool size (Phase 9.5.1).
    age_estimate = _estimate_age(
        age_head=models.age_head,
        device=models.device,
        query_vec=query_vec,
        pool_size=len(fdi_used),
        open_set_decision=open_set_decision,
    )

    return {
        "stage": "search",
        "results": results_list,
        "confidence": _confidence_label(top1, top2),
        "top1_top2_gap": top1_top2_gap,
        "timings_ms": {},  # search-fragment is sub-ms; left empty intentionally
        "n_query_teeth": int(len(fdi_used)),
        "n_dropped": 0,
        "tooth_contributions": tooth_contributions,
        "ensemble": False,
        "ensemble_members": None,
        "open_set_score": open_set_score,
        "open_set_decision": open_set_decision,
        "open_set_threshold": (
            models.open_set_calibration["operating_point"]["threshold"]
            if models.open_set_calibration else None
        ),
        "query_provenance": query_provenance,
        "expected_person_id": expected_pid,
        "sim_top1_percentile": sim_top1_pct,
        "age_estimate": age_estimate,
    }


def run_fragment_search(
    query_id: str,
    tooth_indices: list[int],
    models: PipelineModels,
    config: PipelineConfig,
) -> dict:
    """Phase 9.5 — re-pool a subset of cached tooth embeddings and re-search.

    Loads the teeth.npz cached during the parent /api/identify run for query_id,
    selects the requested indices, mean-pools + L2-normalises, runs FAISS search,
    and re-applies the same open-set + percentile scoring as /api/identify.
    """
    query_dir = config.temp_dir / query_id
    npz_path = query_dir / "teeth.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"no cached embeddings for query_id={query_id}")
    data = np.load(npz_path, allow_pickle=True)
    all_embs: np.ndarray = data["embeddings"]
    all_fdi = [str(x) for x in data["fdi"]]
    all_fdi_conf = [float(x) for x in data["fdi_conf"]]

    if not tooth_indices:
        raise ValueError("tooth_indices must be non-empty")
    indices = [int(i) for i in tooth_indices if 0 <= int(i) < len(all_embs)]
    if not indices:
        raise ValueError("no valid tooth_indices in cache range")

    sub_embs = all_embs[indices]
    pooled = sub_embs.mean(axis=0)
    n = float(np.linalg.norm(pooled))
    query_vec = (pooled / n if n > 1e-12 else pooled).astype(np.float32)

    # Find the original upload path so provenance hashing still works.
    upload_path = query_dir / "upload.png"
    upload_for_provenance = upload_path if upload_path.exists() else None

    payload = _build_search_payload(
        query_vec=query_vec,
        fdi_used=[all_fdi[i] for i in indices],
        fdi_conf_used=[all_fdi_conf[i] for i in indices],
        models=models,
        config=config,
        panoramic_path=upload_for_provenance,
        sub_embs=sub_embs,
    )
    payload["query_id"] = query_id
    payload["tooth_indices"] = indices
    return payload


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


def compute_query_vector_sync(
    panoramic_path: Path,
    models: PipelineModels,
    mode: str = "segmentation",
) -> dict:
    """Phase 9.7 — non-streaming Stage A→E for the enrolment endpoint.

    Runs the same single-model path as ``run_pipeline`` (YOLO → bbox crops →
    FDI classification + dedup → ResNet embedder → mean+L2 pool) but as a
    plain blocking call with no SSE bookkeeping. ``/api/enrol`` calls this
    directly so the modal can show a spinner instead of building a duplicate
    SSE consumer.

    Intentionally does NOT touch the registry index, run open-set scoring, or
    write any temp files — those concerns belong to the caller (which may
    decide to enrol, reject as a duplicate, or skip the embedding entirely).

    Returns::

        {
            "query_vec": np.ndarray (D,) float32, L2-normed,
            "embeddings_arr": np.ndarray (n, D) float32, L2-normed per-tooth,
            "fdi_kept": list[str],
            "fdi_conf_kept": list[float],
            "bboxes_kept": np.ndarray (n, 4) float32,
            "n_teeth": int,
        }

    Raises ``ValueError`` with a user-facing message if YOLO finds no teeth
    or FDI deduplication leaves nothing.
    """
    cfg = models.config
    device = models.device
    if mode not in ("detection", "segmentation"):
        mode = cfg.default_mode

    # --- Stage A: YOLO (single-pass, no streaming) ---
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
        raise ValueError("YOLO returned no results — is this a panoramic X-ray?")
    res0 = results[0]
    boxes = res0.boxes
    if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
        raise ValueError("No teeth found in this image — is it a panoramic X-ray?")
    bboxes = boxes.xyxy.cpu().numpy()

    if mode == "segmentation":
        masks_obj = getattr(res0, "masks", None)
        if masks_obj is None or masks_obj.xy is None or len(masks_obj.xy) == 0:
            raise ValueError("Segmenter found no tooth masks.")
        polygons = [np.asarray(p) for p in masks_obj.xy]
        derived = []
        for poly in polygons:
            xs, ys = poly[:, 0], poly[:, 1]
            derived.append([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())])
        bboxes = np.asarray(derived)

    # --- Stage B: Crops ---
    pano = Image.open(panoramic_path).convert("RGB")
    crops: list[Image.Image] = []
    for (x1, y1, x2, y2) in bboxes:
        ex1, ey1, ex2, ey2 = _expand_bbox(
            (float(x1), float(y1), float(x2), float(y2)), pano.size, padding_ratio=0.1
        )
        crops.append(pano.crop((ex1, ey1, ex2, ey2)))

    # --- Stage C: FDI classification + dedup ---
    with torch.no_grad():
        logits_all = [models.fdi_classifier(_to_tensor(c, cfg.crop_size, device)) for c in crops]
    logits_tensor = torch.cat(logits_all, dim=0)
    probs = F.softmax(logits_tensor, dim=1).cpu().numpy()
    fdi_class_idx = probs.argmax(axis=1)
    fdi_confidences = probs.max(axis=1)
    fdi_labels = [models.fdi_label_inv[idx] for idx in fdi_class_idx]

    seen: dict[str, int] = {}
    keep_indices: list[int] = []
    for i, fdi in enumerate(fdi_labels):
        if fdi in seen and fdi_confidences[i] <= fdi_confidences[seen[fdi]]:
            continue
        if fdi in seen:
            keep_indices.remove(seen[fdi])
        seen[fdi] = i
        keep_indices.append(i)
    keep_indices.sort()

    if not keep_indices:
        raise ValueError("No teeth survived FDI assignment.")

    fdi_kept = [fdi_labels[i] for i in keep_indices]
    fdi_conf_kept = [float(fdi_confidences[i]) for i in keep_indices]
    crops_kept = [crops[i] for i in keep_indices]
    bboxes_kept = bboxes[keep_indices]

    # --- Stage D: Embeddings (single model — enrolment never uses the ensemble) ---
    embeddings = []
    with torch.no_grad():
        for crop, fdi in zip(crops_kept, fdi_kept):
            tensor = _to_tensor(crop, cfg.crop_size, device)
            if models.embedder_uses_metadata:
                fdi_idx = models.embedder_fdi_label_map.get(fdi, 0)
                fdi_t = torch.tensor([fdi_idx], dtype=torch.long, device=device)
                emb = models.embedder(tensor, fdi_t)
            else:
                emb = models.embedder(tensor)
            embeddings.append(emb.cpu().numpy()[0])
    embeddings_arr = np.stack(embeddings).astype(np.float32)

    # --- Stage E: mean + L2 ---
    pooled = embeddings_arr.mean(axis=0)
    n = float(np.linalg.norm(pooled))
    query_vec = (pooled / n if n > 1e-12 else pooled).astype(np.float32)

    return {
        "query_vec": query_vec,
        "embeddings_arr": embeddings_arr,
        "fdi_kept": fdi_kept,
        "fdi_conf_kept": fdi_conf_kept,
        "bboxes_kept": bboxes_kept,
        "n_teeth": int(len(fdi_kept)),
    }


async def run_pipeline(
    panoramic_path: Path,
    query_id: str,
    models: PipelineModels,
    mode: str = "segmentation",
    ensemble: bool = False,
    session_id: str | None = None,
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

    # Phase 9.5 — cache per-tooth embeddings + FDI + bboxes so the fragment
    # explorer can re-pool an arbitrary subset without re-running detection.
    try:
        np.savez(
            query_dir / "teeth.npz",
            embeddings=embeddings_arr,
            fdi=np.array(fdi_kept, dtype=object),
            fdi_conf=np.array(fdi_conf_kept, dtype=np.float32),
            bboxes=bboxes_kept,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[pipeline] WARN: failed to cache teeth.npz for {query_id}: {exc}")

    yield {
        "event": "stage_complete",
        "data": {
            "stage": "embed",
            "n_embeddings": int(len(embeddings_arr)),
            "elapsed_ms": round(timings["embed"], 1),
            "ensemble": ensemble,
            "ensemble_members": [m[0] for m in models.ensemble_models] if ensemble else None,
            # Phase 9.5 — per-tooth metadata so the frontend FragmentSelector
            # can render clickable tooth boxes on the annotated overlay.
            "per_tooth": [
                {
                    "index": i,
                    "fdi": fdi_kept[i],
                    "fdi_confidence": fdi_conf_kept[i],
                    "bbox": bboxes_kept[i].tolist(),
                }
                for i in range(len(fdi_kept))
            ],
            "query_id": query_id,
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

    # Phase 9.7 — merge session enrolments (if any) by raw cosine. Session
    # results carry a `is_session=True` flag downstream so the UI can badge
    # them. We assemble in-place rather than reordering arrays because the
    # downstream tooth-contribution / open-set blocks expect `sims` to remain
    # the canonical-only ranking (sim_top1 calibration was learned on canonical
    # data and would be invalid if session enrolments displaced it).
    session_meta_by_pid: dict[str, dict] = {}
    session_results: list[dict] = []
    if session_id is not None:
        try:
            import importlib
            sessions_module = importlib.import_module("backend.sessions")
            session_index = sessions_module.load_session_index(
                cfg.sessions_dir, session_id, dim=query_vec.shape[0]
            )
            if session_index is not None and len(session_index) > 0:
                session_meta = sessions_module.load_session_meta(
                    cfg.sessions_dir, session_id
                ) or {}
                for p in session_meta.get("persons", []):
                    session_meta_by_pid[p["person_id"]] = p
                k_session = min(cfg.top_k, len(session_index))
                ssims, sids = session_index.search(query_vec, k=k_session)
                for sim, sid in zip(ssims, sids):
                    m = session_meta_by_pid.get(sid, {})
                    session_results.append({
                        "person_id": sid,
                        "fake_name": m.get("fake_name", sid),
                        "n_teeth": int(m.get("n_teeth", 0)) or None,
                        "similarity": float(sim),
                        "is_session": True,
                    })
        except Exception as exc:  # noqa: BLE001
            # Session merge is best-effort. A bad session dir shouldn't take
            # down the whole identify response — log and continue with the
            # canonical-only ranking.
            print(f"[pipeline] session merge failed (session_id={session_id}): {exc}")

    # --- Stage G: Assemble result ---
    canonical_results = []
    for sim, person_id in zip(sims, neighbor_ids):
        meta = models.registry_meta.get(person_id, {})
        canonical_results.append({
            "person_id": person_id,
            "fake_name": meta.get("fake_name", person_id),
            "n_teeth": meta.get("n_teeth"),
            "similarity": float(sim),
            "is_session": False,
        })

    # Merge by raw cosine, take top_k. Session entries float to the top if
    # their similarity beats the canonical neighbours, which is exactly the
    # "self-match after enrolment" case the verify-by-re-query flow needs.
    merged = sorted(
        canonical_results + session_results,
        key=lambda r: r["similarity"],
        reverse=True,
    )[: cfg.top_k]
    results_list = []
    for rank, r in enumerate(merged):
        results_list.append({
            "rank": rank + 1,
            **r,
        })

    # Calibrated quantities — `top1_top2_gap`, `confidence`, `open_set_*`,
    # `similarity_percentile`, `age_estimate` — are all computed off the
    # CANONICAL top-1 / top-2 (sims[0], sims[1]). The Phase 8.6 thresholds
    # were learned on canonical pairs; extending them to session entries
    # would invalidate the calibration semantics. The session-merge contract
    # is "session entries may displace canonical entries in the visible
    # ranking, but calibrated trust never extends to them."
    canonical_top1 = float(sims[0])
    canonical_top2 = float(sims[1]) if len(sims) > 1 else 0.0
    top1_top2_gap = canonical_top1 - canonical_top2 if len(sims) > 1 else 1.0
    confidence = _confidence_label(canonical_top1, canonical_top2)

    # Phase 9.2 — Phase 8.6 calibrated open-set decision + provenance.
    # Always computed on the CANONICAL top-1 (sims[0]), never on a session
    # self-match — calibration was learned on the canonical 1,178-person
    # distribution and is not transferable to session enrolments.
    open_set_score, open_set_decision = _open_set_score(
        float(sims[0]), models.open_set_calibration,
    )
    query_provenance, expected_pid = _query_provenance(
        panoramic_path, models.panoramic_sha256_to_pid,
    )
    # Phase 9.5.1 — age estimate, gated on open_set_decision and pool size.
    age_estimate = _estimate_age(
        age_head=models.age_head,
        device=models.device,
        query_vec=query_vec,
        pool_size=int(len(embeddings_arr)),
        open_set_decision=open_set_decision,
    )
    # Phase 9.3 — empirical percentile of sim_top1 vs known-correct identifications.
    sim_top1_pct = _sim_top1_percentile(float(sims[0]), models.sim_top1_in_registry_sorted)
    # Percentile is well-defined ONLY for the canonical top-1 against the
    # canonical reference distribution (740 rank-1 hits from held-out
    # enrolment). Session entries get None. The canonical top-1 may sit at
    # any merged rank (a session entry can displace it to rank 2 etc), so
    # we identify it by person_id rather than by rank position.
    canonical_top1_pid = str(neighbor_ids[0]) if len(neighbor_ids) > 0 else None
    for r in results_list:
        if (
            not r.get("is_session")
            and r["person_id"] == canonical_top1_pid
        ):
            r["similarity_percentile"] = _sim_top1_percentile(
                r["similarity"], models.sim_top1_in_registry_sorted
            )
        else:
            r["similarity_percentile"] = None

    # Per-tooth contribution: dot each tooth's embedding against the top-1
    # gallery profile. In ensemble mode we average the per-model contributions.
    # Phase 9.7 — when the top-1 is a session enrolment, reconstruct from the
    # session index instead of the canonical one.
    tooth_contributions: list[dict] = []
    try:
        top1_person = results_list[0]["person_id"]
        top1_is_session = bool(results_list[0].get("is_session"))
        if top1_is_session and session_id is not None:
            import importlib
            sessions_module = importlib.import_module("backend.sessions")
            session_index = sessions_module.load_session_index(
                cfg.sessions_dir, session_id, dim=query_vec.shape[0]
            )
            if session_index is None:
                raise RuntimeError("session top-1 but no session index available")
            top1_faiss_idx = session_index.person_ids.index(top1_person)
            top1_vec = session_index.index.reconstruct(top1_faiss_idx)
            per_tooth_sims = embeddings_arr @ top1_vec
        elif ensemble:
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
            # Phase 9.3 — empirical percentile of top-1 sim against 740 in-registry refs.
            "sim_top1_percentile": sim_top1_pct,
            # Phase 9.4 — Phase 8.10 age estimate (sex head intentionally not wired).
            "age_estimate": age_estimate,
            # Phase 9.5 — query_id so the frontend can hit /api/search-fragment.
            "query_id": query_id,
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
