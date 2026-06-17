"""Rotation-canonicalised evaluator — same protocol as evaluate_pipeline.py
but with rotation canonicalisation applied to each detected tooth before embedding.

This wraps `evaluate_pipeline` to reuse all its cached YOLO + FDI work; only
the embedding step is replaced. For each cached Stage A/C output, we:

  1. Take the YOLO polygon for each kept tooth.
  2. Compute its canonical rotation via canonical_rotation_deg (same rule as
     the training-time rotnorm crop extractor — pure geometric).
  3. Apply that rotation to the panoramic, crop the polygon's tight bbox in
     the rotated frame (10% padding, matching training), and resize 224×224.
  4. Run the rotnorm embedder on the canonicalised crops.

We then run the same multi-tooth sweep + rotation-stress + held-out enrolment
protocol as the baseline against the new embedder, and compute paired-difference
bootstrap CIs vs the baseline.

Uses a separate cache namespace keyed on the embedder hash so it never
collides with the baseline cache.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import io
import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from backend.pipeline import PipelineConfig, PipelineModels, _to_tensor
from identification.evaluation.evaluate_pipeline import (
    PipelineCache,
    StageACOutput,
    _b64_to_crop,
    _build_test_panoramic_list,
    _evaluate_against_full_registry_paired,
    _evaluate_sweep_symmetric_paired,
    _file_hash,
    _per_fdi_breakdown,
    _paired_diff_bootstrap,
    _sanity_check_polygon_rotation,
    _sanity_check_registry_overlap,
    evaluate_heldout_enrolment,
    load_gt_polygons,
)
from identification.models.retrieval_index import RetrievalIndex
from identification.scripts.extract_crops_rotnorm import canonical_rotation_deg

warnings.filterwarnings("ignore", message=".*encountered in matmul.*", category=RuntimeWarning)
print = functools.partial(print, flush=True)  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Sanity check: canonical_rotation_deg matches cv2.warpAffine convention
# ---------------------------------------------------------------------------

def _sanity_check_canon_rotation() -> None:
    """Verify that applying canonical_rotation_deg via cv2.warpAffine actually
    brings an elongated polygon to vertical with the wider end at the bottom.
    Catches sign-convention / axis-ordering regressions that the visual checks
    might miss on near-symmetric teeth.
    """
    base_poly = np.array([
        [85, 20], [115, 20],     # narrow root tip
        [130, 200], [70, 200],   # wide crown
    ], dtype=np.float32)
    n_pass = 0
    n_total = 0
    # ±60° range covers the realistic deployment regime (patient positioning
    # variance + ±30° rotation-stress augmentation). Beyond ±90° the arch
    # convention deliberately fails (panoramic taken upside-down is not a
    # deployment case we handle, and would also invert the YOLO segmenter).
    for orig_rot in [-60, -45, -30, -15, 0, 15, 30, 45, 60, 90]:
        M0 = cv2.getRotationMatrix2D((100.0, 110.0), orig_rot, 1.0)
        rot_poly = (M0 @ np.hstack([base_poly, np.ones((4, 1))]).T).T.astype(np.float32)
        rec = canonical_rotation_deg(rot_poly, panoramic_h=400)
        cx, cy = rot_poly.mean(axis=0)
        M1 = cv2.getRotationMatrix2D((float(cx), float(cy)), rec, 1.0)
        canon = (M1 @ np.hstack([rot_poly, np.ones((4, 1))]).T).T
        bw = canon[:, 0].max() - canon[:, 0].min()
        bh = canon[:, 1].max() - canon[:, 1].min()
        vertical = bh > bw * 1.5
        crown_y = canon[2:4, 1].mean()
        root_y = canon[0:2, 1].mean()
        crown_at_bottom = crown_y > root_y
        if vertical and crown_at_bottom:
            n_pass += 1
        n_total += 1
    if n_pass < n_total:
        raise RuntimeError(
            f"canon-rotation sanity FAILED: {n_pass}/{n_total} orientations land "
            f"with the long axis vertical AND crown at the bottom"
        )
    print(f"  [sanity] canonical_rotation_deg ↔ cv2.warpAffine round-trip OK ({n_pass}/{n_total})")


# ---------------------------------------------------------------------------
# Canonicalise + re-embed
# ---------------------------------------------------------------------------

def _rotate_image_about_centroid(
    image: np.ndarray, centroid: tuple[float, float], deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate `image` (BGR uint8) about `centroid` by `deg` degrees.

    Returns (rotated_image, 2x3 affine matrix M used for the rotation).
    """
    h, w = image.shape[:2]
    cx, cy = centroid
    M = cv2.getRotationMatrix2D((float(cx), float(cy)), float(deg), 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )
    return rotated, M


def _resize_with_padding_cv(image: np.ndarray, target_size: int = 224) -> np.ndarray:
    """Mirror identification/scripts/extract_crops_rotnorm.py::resize_with_padding."""
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    y_off = (target_size - new_h) // 2
    x_off = (target_size - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def canonicalise_and_crop_from_polygon(
    panoramic_bgr: np.ndarray,
    polygon: np.ndarray,
    padding_ratio: float = 0.1,
    target_size: int = 224,
) -> np.ndarray:
    """Reproduce the training-time canonicalisation given a YOLO polygon.

    Steps mirror identification/scripts/extract_crops_rotnorm.py::rotate_and_crop:
      1. Compute the canonical rotation angle from the polygon shape.
      2. Rotate the full panoramic about the polygon centroid.
      3. Crop the polygon's tight bbox (in the rotated frame) with 10% padding.
      4. Resize-with-padding to 224×224.
    """
    if polygon.dtype != np.float32:
        polygon = polygon.astype(np.float32)
    rect = cv2.minAreaRect(polygon.reshape(-1, 1, 2))
    (cx, cy), _, _ = rect
    # Use the panoramic height for the arch-convention fallback when the
    # polygon's half-area signal is weak (same rule as training).
    rot_deg = canonical_rotation_deg(polygon, panoramic_h=panoramic_bgr.shape[0])

    rotated, M = _rotate_image_about_centroid(panoramic_bgr, (cx, cy), rot_deg)
    poly_h = np.hstack([polygon, np.ones((len(polygon), 1))]).astype(np.float32)
    rotated_poly = (M @ poly_h.T).T  # (N, 2)
    x1, y1 = rotated_poly.min(axis=0)
    x2, y2 = rotated_poly.max(axis=0)
    bw = x2 - x1
    bh = y2 - y1
    pad_x = bw * padding_ratio
    pad_y = bh * padding_ratio
    H, W = rotated.shape[:2]
    x1c = max(0, int(round(x1 - pad_x)))
    y1c = max(0, int(round(y1 - pad_y)))
    x2c = min(W, int(round(x2 + pad_x)))
    y2c = min(H, int(round(y2 + pad_y)))
    if x2c <= x1c or y2c <= y1c:
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)
    crop = rotated[y1c:y2c, x1c:x2c]
    return _resize_with_padding_cv(crop, target_size)


@dataclass
class CanonCache:
    """Disk cache for canonicalised embeddings, keyed on (stage_ac_key, embedder_hash)."""
    output_dir: Path
    embedder_hash: str

    def __post_init__(self):
        self.dir = self.output_dir / "cache" / "embeddings_rotnorm"
        self.dir.mkdir(parents=True, exist_ok=True)

    def emb_path(self, stage_ac_key: str) -> Path:
        return self.dir / f"{stage_ac_key}__emb{self.embedder_hash}.npy"

    def get_or_compute(
        self,
        stage_ac: StageACOutput,
        stage_ac_key: str,
        models: PipelineModels,
        panoramic_path: Path,
        rotation_deg: float,
    ) -> np.ndarray | None:
        p = self.emb_path(stage_ac_key)
        if p.exists():
            return np.load(p)
        # We need the panoramic in the SAME frame the polygons live in.
        # Polygons in the baseline cache live in the rotated panoramic frame (if any
        # rotation was applied at extraction time, the panoramic was rotated
        # BEFORE YOLO ran and the polygons are in that rotated coordinate
        # system). To re-canonicalise, we must rotate the original panoramic
        # by the same `rotation_deg` first.
        pano = cv2.imread(str(panoramic_path))
        if pano is None:
            return None
        if abs(rotation_deg) > 1e-6:
            # Mirror PIL's rotate(deg, expand=False, fillcolor=0). cv2's
            # warpAffine with center=(W/2, H/2) and the same sign convention
            # produces identical pixels (we verified empirically in the
            # baseline polygon-rotation sanity check, err=0.32px at 30°).
            h, w = pano.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), rotation_deg, 1.0)
            pano = cv2.warpAffine(pano, M, (w, h), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        # Build canonicalised crops one per kept tooth
        from PIL import Image
        embs = []
        with torch.no_grad():
            for i, poly_list in enumerate(stage_ac.polygons):
                poly = np.asarray(poly_list, dtype=np.float32)
                canon_crop = canonicalise_and_crop_from_polygon(
                    pano, poly, padding_ratio=0.1, target_size=224,
                )
                pil = Image.fromarray(cv2.cvtColor(canon_crop, cv2.COLOR_BGR2RGB))
                t = _to_tensor(pil, 224, models.device)
                if models.embedder_uses_metadata:
                    fdi = stage_ac.fdi_labels[i]
                    fdi_idx = models.embedder_fdi_label_map.get(fdi, 0)
                    ft = torch.tensor([fdi_idx], dtype=torch.long, device=models.device)
                    emb = models.embedder(t, ft)
                else:
                    emb = models.embedder(t)
                embs.append(emb.cpu().numpy()[0])
        if not embs:
            return None
        arr = np.stack(embs).astype(np.float32)
        np.save(p, arr)
        return arr


# ---------------------------------------------------------------------------
# Driver — mirrors evaluate_pipeline.py's main(), with canon embedding
# ---------------------------------------------------------------------------

def _stage_ac_key_for(
    image_id: str, rotation_deg: float, yolo_hash: str, fdi_hash: str,
    crop_size: int, yolo_conf: float, yolo_iou: float, yolo_imgsz: int,
) -> str:
    deg = int(round(rotation_deg * 100))
    return (
        f"{image_id}__rot{deg:+06d}__yolo{yolo_hash}__fdi{fdi_hash}"
        f"__c{crop_size}__yc{int(yolo_conf*1000)}__yi{int(yolo_iou*1000)}__ys{yolo_imgsz}"
    )


def _extract_canon_for_split(
    label: str,
    test_persons: list[tuple[str, str, Path, float]],
    canon_cache: CanonCache,
    stage_ac_cache: PipelineCache,
    models: PipelineModels,
    load_gt: bool,
) -> tuple[dict[str, np.ndarray], dict[str, StageACOutput], int]:
    """Extract canonicalised embeddings for each panoramic by reusing the
    baseline Stage A/C cache for YOLO+FDI work."""
    per_person: dict[str, np.ndarray] = {}
    stage_outputs: dict[str, StageACOutput] = {}
    n_failed = 0
    t0 = time.perf_counter()
    for i, (pid, image_id, pano_path, angle) in enumerate(test_persons):
        gt_polys = load_gt_polygons(image_id) if load_gt else {}
        # Reuses the baseline's YOLO+FDI cache (cheap)
        st = stage_ac_cache.get_stage_ac(models, pano_path, pid, image_id, angle, gt_polys)
        if st is None:
            n_failed += 1
            continue
        stage_key = _stage_ac_key_for(
            image_id, angle, stage_ac_cache.yolo_hash, stage_ac_cache.fdi_hash,
            stage_ac_cache.crop_size, stage_ac_cache.yolo_conf,
            stage_ac_cache.yolo_iou, stage_ac_cache.yolo_imgsz,
        )
        embs = canon_cache.get_or_compute(st, stage_key, models, pano_path, angle)
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
    parser.add_argument("--embedder", required=True,
                        help="Path to the rotnorm embedder checkpoint (best.pt).")
    parser.add_argument("--registry-dir", required=True,
                        help="Path to the rotnorm registry directory.")
    parser.add_argument("--phase8-baseline-dir", default="identification/runs/phase8_baseline",
                        help="Where the baseline Stage A/C cache lives (we reuse it).")
    parser.add_argument("--output-dir", default="identification/runs/phase8_rotnorm",
                        help="Where to write the new payload + canon embeddings cache.")
    parser.add_argument("--manifest", default="identification/data/manifest_clean.csv",
                        help="Manifest to read the test panoramic list from (split column).")
    parser.add_argument("--n-query-list", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--rotation-deg", type=float, default=30.0)
    parser.add_argument("--heldout-count", type=int, default=30)
    parser.add_argument("--heldout-trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-rotation", action="store_true")
    parser.add_argument("--skip-heldout", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    # Override the deployed embedder + registry with the rotnorm ones for this eval.
    config = PipelineConfig()
    # Normalise to absolute paths under PROJECT_ROOT so .relative_to() works later.
    emb_arg = Path(args.embedder)
    if not emb_arg.is_absolute():
        emb_arg = (PROJECT_ROOT / emb_arg).resolve()
    config.embedder = emb_arg
    reg_arg = Path(args.registry_dir)
    if not reg_arg.is_absolute():
        reg_arg = (PROJECT_ROOT / reg_arg).resolve()
    config.registry_dir = reg_arg
    models = PipelineModels(config=config)
    print("Loading pipeline with rotnorm embedder + registry...")
    models.load_all()

    yolo_hash = _file_hash(config.yolo_seg_weights)
    fdi_hash = _file_hash(config.fdi_classifier)
    embedder_hash = _file_hash(config.embedder)
    print(f"YOLO seg hash: {yolo_hash}, FDI hash: {fdi_hash}, rotnorm embedder hash: {embedder_hash}")

    # Baseline Stage A/C cache (read-only reuse)
    baseline_dir = (PROJECT_ROOT / args.phase8_baseline_dir).resolve()
    stage_ac_cache = PipelineCache(
        output_dir=baseline_dir,
        yolo_hash=yolo_hash,
        fdi_hash=fdi_hash,
        embedder_hash="UNUSED_we_override_embedding_cache",
        crop_size=config.crop_size,
        yolo_conf=config.yolo_conf,
        yolo_iou=config.yolo_iou,
        yolo_imgsz=config.yolo_imgsz,
        scratch_dir=config.temp_dir / "phase8_rotnorm_scratch",
    )

    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    canon_cache = CanonCache(output_dir=output_dir, embedder_hash=embedder_hash)

    test_persons_all = _build_test_panoramic_list(PROJECT_ROOT / args.manifest)
    if args.limit:
        test_persons_all = test_persons_all[: args.limit]
    print(f"Test panoramics: {len(test_persons_all)}")

    print("Sanity checks...")
    _sanity_check_polygon_rotation()
    _sanity_check_registry_overlap(
        [p for p, _, _ in test_persons_all], models.registry_index,
    )
    _sanity_check_canon_rotation()
    # Verify the registry was built with this exact embedder (BLOCKER #3 fix).
    reg_meta_path = config.registry_dir / "registry_meta.json"
    if reg_meta_path.exists():
        with open(reg_meta_path) as f:
            reg_meta = json.load(f)
        rebuilt_with = reg_meta.get("checkpoint")
        if rebuilt_with and not str(config.embedder).endswith(rebuilt_with):
            print(f"  [sanity] WARNING registry was built with '{rebuilt_with}' but "
                  f"embedder is '{config.embedder}' — embeddings may not match registry!")
        else:
            print(f"  [sanity] registry built with embedder hash matches: {rebuilt_with}")

    seed_root = np.random.SeedSequence(args.seed)
    baseline_rng, rotation_rng, heldout_rng, angle_rng, bootstrap_rng = (
        np.random.default_rng(s) for s in seed_root.spawn(5)
    )

    # --- Baseline upright ---
    baseline_per_person_r1: dict[int, dict[str, float]] = {}
    baseline_permutations: dict[int, list[dict[str, np.ndarray]]] = {}
    if not args.skip_baseline:
        print("\n[baseline] canon-embedding upright Stage A/C cache...")
        upright_persons = [(p, i, path, 0.0) for p, i, path in test_persons_all]
        per_person, stage_outputs, n_failed = _extract_canon_for_split(
            "baseline", upright_persons, canon_cache, stage_ac_cache, models, load_gt=True,
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
                  f"R5={s['rank5_mean']:.4f}")

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

        print("[baseline] per-FDI breakdown...")
        per_fdi = _per_fdi_breakdown(stage_outputs, per_person, models.registry_index)

        for s in sweep_sym:
            if s.get("skipped"):
                continue
            baseline_per_person_r1[s["n_query"]] = {
                pid: r1 for pid, r1 in zip(s["pids"], s["per_person_r1"])
            }

        baseline_payload = {
            "label": "yolo_eval_rotnorm",
            "rotation_deg": 0.0,
            "n_persons_attempted": len(upright_persons),
            "n_persons_usable": len(per_person),
            "n_persons_failed": n_failed,
            "embedder_checkpoint": str(config.embedder.relative_to(PROJECT_ROOT)),
            "registry_dir": str(config.registry_dir.relative_to(PROJECT_ROOT)),
            "embedder_hash": embedder_hash,
            "registry_size": len(models.registry_index),
            "n_query_list": args.n_query_list,
            "n_trials": args.n_trials,
            "sweep_symmetric": sweep_sym,
            "sweep_full_registry": sweep_reg,
            "per_fdi": per_fdi,
        }
        with open(output_dir / "yolo_eval.json", "w") as f:
            json.dump(baseline_payload, f, indent=2)
        print(f"[baseline] saved → {output_dir/'yolo_eval.json'}")

    # --- Rotation-stress ---
    if not args.skip_rotation:
        print(f"\n[rotation_stress] drawing per-person angles in ±{args.rotation_deg:.0f}°...")
        rotated_persons: list[tuple[str, str, Path, float]] = []
        angle_by_pid: dict[str, float] = {}
        for pid, image_id, pano_path in test_persons_all:
            angle = float(angle_rng.uniform(-args.rotation_deg, args.rotation_deg))
            rotated_persons.append((pid, image_id, pano_path, angle))
            angle_by_pid[pid] = angle

        print("[rotation_stress] canon-embedding rotated Stage A/C cache...")
        per_person, stage_outputs, n_failed = _extract_canon_for_split(
            "rotation_stress", rotated_persons, canon_cache, stage_ac_cache, models, load_gt=True,
        )

        print("[rotation_stress] symmetric sweep (REUSING baseline permutations for pairing)...")
        # Same logic as evaluate_pipeline.py rotation symmetric sweep
        sweep_sym_rot: list[dict] = []
        per_person_r1_rot: dict[int, dict[str, float]] = {}
        for n_q in args.n_query_list:
            base_perms = baseline_permutations.get(n_q, []) if not args.skip_baseline else []
            eligible_pids = [pid for pid in per_person.keys() if len(per_person[pid]) >= n_q + 1]
            if not base_perms:
                from identification.evaluation.evaluate_pipeline import _draw_paired_subsets
                base_perms = _draw_paired_subsets(per_person, eligible_pids, n_q, args.n_trials, rotation_rng)
            if len(eligible_pids) < 5:
                sweep_sym_rot.append({"n_query": n_q, "n_persons": len(eligible_pids), "skipped": True})
                continue
            n_trials_eff = len(base_perms)
            match_r1 = np.zeros((n_trials_eff, len(eligible_pids)), dtype=bool)
            match_r5 = np.zeros_like(match_r1)
            truly_paired: set[str] = set()
            from identification.evaluation.evaluate_pipeline import _mean_pool
            for t, perm in enumerate(base_perms):
                queries, galleries, used = [], [], []
                for pid in eligible_pids:
                    arr = per_person[pid]
                    if pid in perm and len(perm[pid]) == len(arr):
                        idx = perm[pid]
                        q_idx = idx[:n_q]
                        g_idx = idx[n_q:]
                        truly_paired.add(pid)
                    else:
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
                    raise RuntimeError(f"rot sym n_q={n_q} trial={t}: non-finite Q/G")
                sim = Q @ G.T
                if not np.isfinite(sim).all():
                    raise RuntimeError(f"rot sym n_q={n_q} trial={t}: non-finite sim")
                ranked = np.argsort(-sim, axis=1)
                pids_arr = np.array(used)
                ranked_labels = pids_arr[ranked]
                mat = ranked_labels == pids_arr[:, None]
                idx_map = {pid: k for k, pid in enumerate(eligible_pids)}
                for u_i, pid in enumerate(used):
                    k = idx_map[pid]
                    match_r1[t, k] = mat[u_i, 0]
                    match_r5[t, k] = mat[u_i, :5].any()
            per_person_r1 = match_r1.mean(axis=0)
            rank1 = float(per_person_r1.mean())
            rank5 = float(match_r5.mean(axis=0).mean())
            n_boot = 1000
            n_e = len(eligible_pids)
            boot_r1 = np.empty(n_boot)
            for b in range(n_boot):
                sel = bootstrap_rng.integers(0, n_e, size=n_e)
                boot_r1[b] = per_person_r1[sel].mean()
            ci_low, ci_high = np.percentile(boot_r1, [2.5, 97.5])
            sweep_sym_rot.append({
                "n_query": n_q,
                "n_persons": n_e,
                "n_persons_truly_paired": len(truly_paired),
                "rank1_mean": rank1,
                "rank1_ci95_low": float(ci_low),
                "rank1_ci95_high": float(ci_high),
                "rank5_mean": rank5,
                "per_person_r1": per_person_r1.tolist(),
                "pids": eligible_pids,
                "truly_paired_pids": sorted(truly_paired),
            })
            per_person_r1_rot[n_q] = {pid: r for pid, r in zip(eligible_pids, per_person_r1) if pid in truly_paired}
            print(f"  rot sym n={n_q:>2}: R1={rank1:.4f} [{ci_low:.3f}, {ci_high:.3f}]")

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

        rot_payload = {
            "label": "rotation_stress_rotnorm",
            "rotation_deg_max": args.rotation_deg,
            "per_person_angle": angle_by_pid,
            "n_persons_attempted": len(rotated_persons),
            "n_persons_usable": len(per_person),
            "n_persons_failed": n_failed,
            "n_query_list": args.n_query_list,
            "n_trials": args.n_trials,
            "sweep_symmetric_rotated": sweep_sym_rot,
            "sweep_full_registry_rotated": sweep_reg_rot,
            "paired_diff_vs_baseline": paired_diff,
        }
        with open(output_dir / "rotation_stress.json", "w") as f:
            json.dump(rot_payload, f, indent=2)
        print(f"[rotation_stress] saved → {output_dir/'rotation_stress.json'}")

    # --- Heldout ---
    if not args.skip_heldout:
        print("\n[heldout_enrol] reusing upright canonicalised embeddings...")
        upright_persons = [(p, i, path, 0.0) for p, i, path in test_persons_all]
        per_person, _, _ = _extract_canon_for_split(
            "heldout_enrol", upright_persons, canon_cache, stage_ac_cache, models, load_gt=False,
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
        print(f"[heldout_enrol] saved → {output_dir/'heldout_enrol.json'}")

    print("\nRotnorm evaluation complete.")


if __name__ == "__main__":
    main()
