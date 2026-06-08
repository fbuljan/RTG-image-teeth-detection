"""
Phase 8.5 — adversarial blur stress test.

For each K in {1, 3, 5}, two corruption slices:
  (a) "blur random K teeth"        — pick K teeth uniformly at random per person
  (b) "blur top-K-weighted teeth"  — pick the K teeth with highest weights under cfg

"Blur" implementation choice: replace the corrupted tooth's embedding with the
MEAN of the OTHER teeth's embeddings in the same image (smooth degradation —
the tooth becomes a "non-discriminative" but in-distribution vector). Then
L2-renormalise that replaced vector so its norm matches the rest.

For each (K, mode) we compute Recall@1 @ n=16 (symmetric query/gallery split,
same protocol as fit_aggregator.py) under three aggregation rules:
  - mean    : uniform mean-pool
  - w_recomp: weighted, with cfg recomputed on the corrupted batch
  - w_frozen: weighted, with weights frozen at the CLEAN per-tooth scores
              (i.e., the corruption changes only the embeddings, not the weights)

Pre-registered pass criterion #6 (from design-review synthesis):
  weight-frozen-at-clean stress within 3pp of clean weighted at top-K=4 blur.
(We sweep K in {1, 3, 5} and report all three plus the clean reference.)

Outputs:
  identification/runs/phase8_weighted/adversarial/adversarial_blur.json
"""

from __future__ import annotations

import argparse
import functools
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from identification.evaluation.weighted_aggregation import (
    PerToothFeatures,
    WeightConfig,
    build_fdi_label_map,
    compute_weights,
    effective_n_teeth,
    extract_features,
    weighted_person_embedding,
)
from identification.evaluation.fit_aggregator import (
    bootstrap_r1,
    load_val_cache,  # generic cache loader — works for any cache dir
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
print = functools.partial(print, flush=True)


# --------------------------------------------------------------------------- #
# Corruption                                                                  #
# --------------------------------------------------------------------------- #

def _blur_embeddings(embs: np.ndarray, blur_idx: np.ndarray) -> np.ndarray:
    """Replace embeddings at `blur_idx` with the mean of the OTHER embeddings.

    If all teeth are blurred we fall back to a small-noise vector to avoid a
    degenerate zero embedding. The replacement is L2-renormalised to the
    median norm of the non-blurred embeddings (preserves embedding-space scale).
    """
    T = embs.shape[0]
    if T == 0 or len(blur_idx) == 0:
        return embs.copy()
    mask = np.zeros(T, dtype=bool)
    mask[blur_idx] = True
    keep = ~mask

    out = embs.copy()
    if keep.any():
        replacement = embs[keep].mean(axis=0)
        # Normalise to the median norm of kept embeddings (typical scale)
        ref_norm = float(np.median(np.linalg.norm(embs[keep], axis=1)))
        rep_norm = float(np.linalg.norm(replacement))
        if rep_norm > 1e-12:
            replacement = replacement * (ref_norm / rep_norm)
    else:
        # Edge case: every tooth blurred. Use a tiny random vector.
        rng = np.random.default_rng(0)
        replacement = rng.normal(0, 1e-3, size=embs.shape[1]).astype(np.float32)

    for i in blur_idx:
        out[i] = replacement
    return out


# --------------------------------------------------------------------------- #
# Aggregation helpers (subset-aware, matches fit_aggregator.py)              #
# --------------------------------------------------------------------------- #

def _subset_features(features: PerToothFeatures, idx: np.ndarray) -> PerToothFeatures:
    return PerToothFeatures(
        image_id=features.image_id,
        person_id=features.person_id,
        fdi=[features.fdi[i] for i in idx],
        fdi_idx=features.fdi_idx[idx],
        yolo_logit=features.yolo_logit[idx],
        log_norm_area=features.log_norm_area[idx],
        low_conf_flag=features.low_conf_flag[idx],
        log_n_teeth=features.log_n_teeth,
        n_teeth=len(idx),
    )


def _mean_pool(embs: np.ndarray) -> np.ndarray:
    if embs.shape[0] == 0:
        return np.zeros(embs.shape[1], dtype=np.float32)
    pooled = embs.mean(axis=0)
    nrm = np.linalg.norm(pooled)
    return pooled / nrm if nrm > 1e-12 else pooled


def _weighted_frozen(
    embs: np.ndarray,
    clean_weights: np.ndarray,
) -> np.ndarray:
    """Aggregate `embs` using pre-computed weights (no recompute on corrupted data)."""
    if embs.shape[0] == 0:
        return np.zeros(embs.shape[1], dtype=np.float32)
    w = clean_weights
    if w.shape[0] != embs.shape[0]:
        w = np.full(embs.shape[0], 1.0 / embs.shape[0], dtype=np.float32)
    # Renormalise (subset of weights does not sum to 1)
    s = float(w.sum())
    if s <= 0:
        w = np.full(embs.shape[0], 1.0 / embs.shape[0], dtype=np.float32)
    else:
        w = w / s
    pooled = (w[:, None] * embs).sum(axis=0)
    nrm = np.linalg.norm(pooled)
    return pooled / nrm if nrm > 1e-12 else pooled


# --------------------------------------------------------------------------- #
# Adversarial R1 evaluation                                                   #
# --------------------------------------------------------------------------- #

def evaluate_adversarial_R1(
    persons: List[Tuple[PerToothFeatures, np.ndarray]],
    cfg: WeightConfig,
    K: int,
    mode: str,                       # "random" or "top_weighted"
    n_query: int = 16,
    n_trials: int = 5,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Compute per-person R1 vectors under three aggregation rules.

    Returns a dict with keys: "mean", "w_recomp", "w_frozen".
    Each value is a per-person R1 vector of length n_eligible.
    """
    rng = np.random.default_rng(seed)
    eligible = [(f, e) for f, e in persons if f.n_teeth >= n_query + 1]
    n = len(eligible)
    if n < 5:
        return {k: np.array([]) for k in ("mean", "w_recomp", "w_frozen")}

    # Pre-compute clean per-tooth weights (full image, NOT subset)
    clean_weights_full = [compute_weights(f, cfg) for f, _ in eligible]

    match_mean = np.zeros((n_trials, n), dtype=bool)
    match_wrec = np.zeros((n_trials, n), dtype=bool)
    match_wfrz = np.zeros((n_trials, n), dtype=bool)

    for t in range(n_trials):
        Q_mean, G_mean = [], []
        Q_wrec, G_wrec = [], []
        Q_wfrz, G_wfrz = [], []

        for pi, (f, e) in enumerate(eligible):
            perm = rng.permutation(f.n_teeth)
            q_idx, g_idx = perm[:n_query], perm[n_query:]

            # Pick blur indices PER SIDE (query and gallery independently)
            # so that the corruption is local to each retrieval slot.
            k_eff = min(K, len(q_idx))
            if mode == "random":
                q_blur = rng.choice(q_idx, size=k_eff, replace=False)
            elif mode == "top_weighted":
                # On the subset, recompute clean weights and pick top-K
                q_subf = _subset_features(f, q_idx)
                q_subw = compute_weights(q_subf, cfg)
                order = np.argsort(-q_subw)
                q_blur = q_idx[order[:k_eff]]
            else:
                raise ValueError(f"Unknown mode {mode}")

            k_eff_g = min(K, len(g_idx))
            if mode == "random":
                g_blur = rng.choice(g_idx, size=k_eff_g, replace=False)
            else:
                g_subf = _subset_features(f, g_idx)
                g_subw = compute_weights(g_subf, cfg)
                order = np.argsort(-g_subw)
                g_blur = g_idx[order[:k_eff_g]]

            # Corrupt embeddings (full-image-indexed)
            e_corrupt = _blur_embeddings(e, np.concatenate([q_blur, g_blur]))

            q_embs = e_corrupt[q_idx]
            g_embs = e_corrupt[g_idx]

            # --- aggregation rule 1: mean pool ---
            Q_mean.append(_mean_pool(q_embs))
            G_mean.append(_mean_pool(g_embs))

            # --- aggregation rule 2: weighted, recomputed on corrupt batch ---
            # (weights depend only on FEATURES, not embeddings, so they're the
            #  same as clean subset weights — corruption affects only embs.)
            q_subf = _subset_features(f, q_idx)
            g_subf = _subset_features(f, g_idx)
            Q_wrec.append(weighted_person_embedding(q_embs, q_subf, cfg))
            G_wrec.append(weighted_person_embedding(g_embs, g_subf, cfg))

            # --- aggregation rule 3: weighted, frozen at clean ---
            # "Clean" here means weights computed from the FULL image's features
            # (the deployment-time weights), restricted to the subset indices and
            # renormalised.
            cw = clean_weights_full[pi]
            Q_wfrz.append(_weighted_frozen(q_embs, cw[q_idx]))
            G_wfrz.append(_weighted_frozen(g_embs, cw[g_idx]))

        for name, Q, G, match in (
            ("mean", Q_mean, G_mean, match_mean),
            ("w_recomp", Q_wrec, G_wrec, match_wrec),
            ("w_frozen", Q_wfrz, G_wfrz, match_wfrz),
        ):
            Q = np.stack(Q)
            G = np.stack(G)
            if not (np.isfinite(Q).all() and np.isfinite(G).all()):
                continue
            sim = Q @ G.T
            if not np.isfinite(sim).all():
                continue
            ranked = np.argsort(-sim, axis=1)
            for i in range(n):
                match[t, i] = (ranked[i, 0] == i)

    return {
        "mean": match_mean.mean(axis=0),
        "w_recomp": match_wrec.mean(axis=0),
        "w_frozen": match_wfrz.mean(axis=0),
    }


def evaluate_clean_R1(
    persons: List[Tuple[PerToothFeatures, np.ndarray]],
    cfg: WeightConfig,
    n_query: int = 16,
    n_trials: int = 5,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Clean (no blur) reference for mean / weighted (recompute == frozen here)."""
    return evaluate_adversarial_R1(
        persons, cfg, K=0, mode="random",
        n_query=n_query, n_trials=n_trials, seed=seed,
    )


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def _summarise(pp: np.ndarray, seed: int) -> Dict[str, float]:
    pt, lo, hi = bootstrap_r1(pp, n_boot=1000, seed=seed)
    return {"point": pt, "ci_low": lo, "ci_high": hi, "n": int(pp.size)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weight-config",
        default="identification/runs/phase8_weighted/weight_config.json",
        help="Path to fitted WeightConfig JSON (from fit_aggregator.py).",
    )
    parser.add_argument(
        "--cache-dir",
        default="identification/runs/phase8_baseline/cache/stage_ac",
        help="Stage A/C cache directory (test split, upright rotation).",
    )
    parser.add_argument(
        "--embedding-dir",
        default="identification/runs/phase8_baseline/cache/embeddings",
        help="Per-image embedding cache directory.",
    )
    parser.add_argument(
        "--embedder",
        default="identification/runs/embedding_fdi_init_v1/best.pt",
        help="Embedder weights (used only to compute cache hash key).",
    )
    parser.add_argument(
        "--output-dir",
        default="identification/runs/phase8_weighted/adversarial",
    )
    parser.add_argument("--n-query", type=int, default=16)
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--K-values", type=int, nargs="+", default=[1, 3, 5],
        help="Number of teeth to blur per person, per slice.",
    )
    args = parser.parse_args()

    cfg_path = PROJECT_ROOT / args.weight_config
    if not cfg_path.exists():
        raise FileNotFoundError(f"WeightConfig not found at {cfg_path}; run fit_aggregator.py first.")
    cfg = WeightConfig.load(cfg_path)
    print(f"Loaded WeightConfig: α={cfg.alpha} γ={cfg.gamma} η={cfg.eta} μ={cfg.mu} T={cfg.T} "
          f"|β|_max={float(np.abs(cfg.beta_fdi).max()):.3f}")

    # Embedder hash
    import hashlib
    h = hashlib.sha256()
    with open(PROJECT_ROOT / args.embedder, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    embedder_hash = h.hexdigest()[:16]
    print(f"Embedder hash: {embedder_hash}")

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test cache (upright)
    persons = load_val_cache(
        cache_dir=PROJECT_ROOT / args.cache_dir,
        embedding_dir=PROJECT_ROOT / args.embedding_dir,
        embedder_hash=embedder_hash,
        rotation_filter=0.0,
    )
    if not persons:
        raise RuntimeError("No persons loaded; check cache paths.")
    print(f"Eligible persons (n_teeth ≥ {args.n_query + 1}): "
          f"{sum(1 for f, _ in persons if f.n_teeth >= args.n_query + 1)} / {len(persons)}")

    table: Dict[str, dict] = {}

    # --- Clean reference (K=0) ---
    t0 = time.perf_counter()
    print("\n=== Clean reference (K=0) ===")
    clean = evaluate_clean_R1(persons, cfg, n_query=args.n_query,
                              n_trials=args.n_trials, seed=args.seed)
    for k, pp in clean.items():
        s = _summarise(pp, seed=args.seed + 1)
        table[f"clean__{k}"] = s
        print(f"  {k:10s}: R1={s['point']:.4f} [{s['ci_low']:.4f}, {s['ci_high']:.4f}]  n={s['n']}")

    # --- Blur sweeps ---
    median_ent_clean = float(np.median([
        effective_n_teeth(f, cfg) for f, _ in persons if f.n_teeth >= args.n_query + 1
    ]))

    for K in args.K_values:
        for mode in ("random", "top_weighted"):
            slice_name = f"blur_{mode}_K{K}"
            print(f"\n=== {slice_name} ===")
            r = evaluate_adversarial_R1(
                persons, cfg, K=K, mode=mode,
                n_query=args.n_query, n_trials=args.n_trials,
                seed=args.seed + K,
            )
            for k, pp in r.items():
                s = _summarise(pp, seed=args.seed + 1)
                table[f"{slice_name}__{k}"] = s
                print(f"  {k:10s}: R1={s['point']:.4f} [{s['ci_low']:.4f}, {s['ci_high']:.4f}]  n={s['n']}")

    elapsed = time.perf_counter() - t0
    print(f"\nElapsed: {elapsed:.1f}s")

    # --- Headline + pass-criterion #6 (within 3pp of clean weighted, K=4 proxy) ---
    clean_w = table["clean__w_recomp"]["point"]
    clean_m = table["clean__mean"]["point"]
    print("\n=== Headline ===")
    print(f"  Clean    : mean={clean_m:.4f}  weighted={clean_w:.4f}  (Δ={clean_w - clean_m:+.4f})")

    for K in args.K_values:
        for mode in ("random", "top_weighted"):
            base = f"blur_{mode}_K{K}"
            m = table[f"{base}__mean"]["point"]
            wr = table[f"{base}__w_recomp"]["point"]
            wf = table[f"{base}__w_frozen"]["point"]
            print(
                f"  K={K} {mode:13s}: mean={m:.4f}  w_recomp={wr:.4f}  w_frozen={wf:.4f}  "
                f"(Δw_vs_mean={wr - m:+.4f}  drop_from_clean_w={wr - clean_w:+.4f})"
            )

    # Approximate pass-criterion #6 at K=4 by linear interpolation between K=3, K=5
    drop_pp_at_K4 = None
    if 3 in args.K_values and 5 in args.K_values:
        wf3 = table["blur_top_weighted_K3__w_frozen"]["point"]
        wf5 = table["blur_top_weighted_K5__w_frozen"]["point"]
        wf4_est = 0.5 * (wf3 + wf5)
        drop_pp_at_K4 = (clean_w - wf4_est) * 100.0
        print(
            f"\n  Pass-criterion #6 check (top-weighted, w_frozen):\n"
            f"    interpolated K=4 estimate = {wf4_est:.4f}  drop from clean = {drop_pp_at_K4:+.2f}pp"
        )
        if drop_pp_at_K4 <= 3.0:
            print(f"    PASS: drop ≤ 3pp")
        else:
            print(f"    FAIL: drop > 3pp")

    summary = {
        "weight_config": cfg.to_payload(),
        "embedder_hash": embedder_hash,
        "n_query": args.n_query,
        "n_trials": args.n_trials,
        "K_values": list(args.K_values),
        "blur_strategy": "replace_with_mean_of_other_teeth_renormed_to_median_norm",
        "median_effective_n_teeth_clean": median_ent_clean,
        "table": table,
        "criterion6_drop_pp_at_K4_top_weighted_frozen": drop_pp_at_K4,
        "criterion6_pass": (drop_pp_at_K4 is not None and drop_pp_at_K4 <= 3.0),
    }
    out_path = output_dir / "adversarial_blur.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
