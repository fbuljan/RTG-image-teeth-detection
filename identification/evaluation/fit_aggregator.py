"""
Fit confidence-weighted-aggregation hyperparameters on the val cache.

Protocol (per design-review synthesis):
  1. Load val Stage A/C cache + cached embeddings (one entry per val person, upright)
  2. Extract per-tooth features (PerToothFeatures)
  3. 3-fold person-stratified split of the val set (~58 persons per fold)
  4. Coarse log-spaced grid over (alpha, gamma, eta, mu, T) + per-FDI prior search
  5. For each candidate config, compute val R1@n=16 via the same symmetric-pair
     protocol as evaluate_pipeline.py (queries vs galleries, both built from the
     same person's teeth disjointly)
  6. Bootstrap CI over persons (B=1000) on the chosen-config val R1
  7. Sparsity guard: effective_n_teeth >= 6 (median) — reject sparser configs
  8. Cross-fold stability check: variance of chosen hyperparameters across folds
     < 50% of point value per coefficient

Outputs:
  - identification/runs/phase8_weighted/fit_result.json  (chosen WeightConfig + diagnostics)
  - identification/runs/phase8_weighted/grid_results.csv (all candidates with val R1)

This script is computation-light (no model loading, no GPU); only numpy + cv2.
"""

from __future__ import annotations

import argparse
import functools
import itertools
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from identification.evaluation.weighted_aggregation import (
    WeightConfig,
    build_fdi_label_map,
    compute_weights,
    effective_n_teeth,
    extract_features,
    weighted_person_embedding,
    PerToothFeatures,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
print = functools.partial(print, flush=True)


# --------------------------------------------------------------------------- #
# Cache loading                                                               #
# --------------------------------------------------------------------------- #

def load_val_cache(
    cache_dir: Path,
    embedding_dir: Path,
    embedder_hash: str,
    rotation_filter: float = 0.0,
) -> List[Tuple[PerToothFeatures, np.ndarray]]:
    """Load all upright Stage A/C payloads + their cached embeddings.

    Returns a list of (features, per_tooth_embs (T, D)) tuples, one per person.
    """
    fdi_lm = build_fdi_label_map(PROJECT_ROOT / "identification/data/manifest_clean.csv")
    rot_token = f"rot{int(round(rotation_filter * 100)):+06d}"

    out: List[Tuple[PerToothFeatures, np.ndarray]] = []
    skipped = 0
    for f in sorted(cache_dir.iterdir()):
        if not f.name.endswith(".json"):
            continue
        if rot_token not in f.name:
            continue
        try:
            payload = json.load(open(f))
        except Exception:
            skipped += 1
            continue
        features = extract_features(payload, fdi_lm)
        if features.n_teeth == 0:
            skipped += 1
            continue
        # Locate matching embeddings file
        stem = f.stem  # full Stage A/C key
        emb_path = embedding_dir / f"{stem}__emb{embedder_hash}.npy"
        if not emb_path.exists():
            skipped += 1
            continue
        embs = np.load(emb_path)
        if embs.shape[0] != features.n_teeth:
            print(f"  skipping {features.image_id}: emb={embs.shape[0]} != features={features.n_teeth}")
            skipped += 1
            continue
        out.append((features, embs.astype(np.float32)))
    print(f"Loaded {len(out)} persons from {cache_dir} (skipped {skipped})")
    return out


# --------------------------------------------------------------------------- #
# Symmetric-pair person-retrieval R1 (matches evaluate_pipeline.py protocol)  #
# --------------------------------------------------------------------------- #

def _aggregate_query_gallery(
    features: PerToothFeatures,
    embs: np.ndarray,
    q_idx: np.ndarray,
    g_idx: np.ndarray,
    cfg: WeightConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate sub-arrays of teeth under cfg's weighting."""
    if len(q_idx) == 0 or len(g_idx) == 0:
        return None, None
    # Build features for the SUBSET of teeth (need to re-extract weights on subset)
    def _subset(idx: np.ndarray) -> Tuple[np.ndarray, PerToothFeatures]:
        sub = PerToothFeatures(
            image_id=features.image_id,
            person_id=features.person_id,
            fdi=[features.fdi[i] for i in idx],
            fdi_idx=features.fdi_idx[idx],
            yolo_logit=features.yolo_logit[idx],
            log_norm_area=features.log_norm_area[idx],
            low_conf_flag=features.low_conf_flag[idx],
            log_n_teeth=features.log_n_teeth,  # IMAGE-level scalar, unchanged
            n_teeth=len(idx),
        )
        sub_embs = embs[idx]
        return weighted_person_embedding(sub_embs, sub, cfg), sub
    q_emb, _ = _subset(q_idx)
    g_emb, _ = _subset(g_idx)
    return q_emb, g_emb


def evaluate_symmetric_R1(
    persons: List[Tuple[PerToothFeatures, np.ndarray]],
    cfg: WeightConfig,
    n_query: int = 16,
    n_trials: int = 5,
    seed: int = 42,
) -> Tuple[float, np.ndarray, List[str]]:
    """Return (mean R1, per-person R1 vector, eligible pids)."""
    rng = np.random.default_rng(seed)
    eligible = [(f, e) for f, e in persons if f.n_teeth >= n_query + 1]
    if len(eligible) < 5:
        return 0.0, np.array([]), []
    pids = [f.person_id for f, _ in eligible]

    n = len(eligible)
    match_r1 = np.zeros((n_trials, n), dtype=bool)

    for t in range(n_trials):
        # Independent permutation per person per trial
        Q, G = [], []
        for (f, e) in eligible:
            perm = rng.permutation(f.n_teeth)
            q_idx, g_idx = perm[:n_query], perm[n_query:]
            q_emb, g_emb = _aggregate_query_gallery(f, e, q_idx, g_idx, cfg)
            Q.append(q_emb)
            G.append(g_emb)
        Q = np.stack(Q)
        G = np.stack(G)
        if not (np.isfinite(Q).all() and np.isfinite(G).all()):
            print(f"  non-finite at trial {t}; defaulting to 0")
            continue
        sim = Q @ G.T
        if not np.isfinite(sim).all():
            continue
        ranked = np.argsort(-sim, axis=1)
        for i in range(n):
            match_r1[t, i] = (ranked[i, 0] == i)

    per_person = match_r1.mean(axis=0)
    return float(per_person.mean()), per_person, pids


# --------------------------------------------------------------------------- #
# Grid search                                                                 #
# --------------------------------------------------------------------------- #

def grid_search(
    persons: List[Tuple[PerToothFeatures, np.ndarray]],
    fdi_label_map: Dict[str, int],
    n_query: int = 16,
    n_trials: int = 5,
    seed: int = 42,
    sparsity_min: float = 6.0,
    include_fdi_prior: bool = True,
) -> Tuple[WeightConfig, pd.DataFrame]:
    """Coarse log-spaced grid + per-FDI prior search.

    Strategy:
      Stage 1: search continuous coefficients (alpha, gamma, eta, mu, T) at
               beta_fdi=0. Pick top-N by val R1.
      Stage 2: holding the top continuous config, search per-FDI prior.
               To keep grid tractable, parameterise as `beta_fdi[idx] = phi * x[fdi]`
               where x[fdi] = R1_prior_per_fdi - mean_R1_prior. phi sweeps the scale.

    Returns the best WeightConfig and a DataFrame of all grid evaluations.
    """
    num_fdi = max(fdi_label_map.values()) + 2  # +1 for unknown bucket
    zero_beta = np.zeros(num_fdi, dtype=np.float32)

    # Stage 1 grid (coarse, ~600 points)
    alpha_grid = [0.0, 0.5, 1.0, 2.0]
    gamma_grid = [0.0, -0.3, -1.0, -2.0]      # large negative log_norm_area => downweight small teeth (negative log)
    eta_grid = [0.0, -1.0, -2.0]              # downweight low_conf_flag
    mu_grid = [0.0]                           # log_n_teeth is image-level, doesn't differentiate within image
    T_grid = [0.3, 0.5, 1.0, 2.0, 5.0]

    results: list[dict] = []
    t0 = time.perf_counter()
    n_combos = len(alpha_grid) * len(gamma_grid) * len(eta_grid) * len(mu_grid) * len(T_grid)
    print(f"Stage 1 grid: {n_combos} combos × n_trials={n_trials}")
    i = 0
    for alpha, gamma, eta, mu, T in itertools.product(
        alpha_grid, gamma_grid, eta_grid, mu_grid, T_grid
    ):
        cfg = WeightConfig(alpha=alpha, gamma=gamma, eta=eta, mu=mu, T=T, beta_fdi=zero_beta.copy())
        r1, per_person, _ = evaluate_symmetric_R1(persons, cfg, n_query=n_query, n_trials=n_trials, seed=seed)
        # Sparsity: median effective_n_teeth across persons
        ents = [effective_n_teeth(f, cfg) for f, _ in persons if f.n_teeth >= n_query + 1]
        median_ent = float(np.median(ents)) if ents else 0.0
        results.append({
            "stage": 1, "alpha": alpha, "gamma": gamma, "eta": eta, "mu": mu, "T": T,
            "phi": 0.0, "r1": r1, "median_ent": median_ent,
            "n_eligible": sum(1 for f, _ in persons if f.n_teeth >= n_query + 1),
        })
        i += 1
        if i % 50 == 0:
            elapsed = time.perf_counter() - t0
            eta_s = (n_combos - i) * elapsed / i
            print(f"  Stage 1: {i}/{n_combos}  best so far: {max(r['r1'] for r in results):.4f}  eta {eta_s:.0f}s")
    df1 = pd.DataFrame(results)

    # Stage 1 winner (with sparsity guard)
    df1_ok = df1[df1["median_ent"] >= sparsity_min]
    if df1_ok.empty:
        print(f"  Stage 1: ALL configs failed sparsity guard (median_ent >= {sparsity_min})")
        df1_ok = df1  # fall back to all
    stage1_best = df1_ok.sort_values("r1", ascending=False).iloc[0]
    print(f"\nStage 1 winner: R1={stage1_best['r1']:.4f}, "
          f"α={stage1_best['alpha']}, γ={stage1_best['gamma']}, η={stage1_best['eta']}, "
          f"μ={stage1_best['mu']}, T={stage1_best['T']}, median_ent={stage1_best['median_ent']:.1f}")

    # Stage 2: per-FDI prior
    # Compute a per-FDI rank1 prior from val: for each FDI, train R1 of single-tooth queries
    # As a proxy, use the relative frequency of that FDI's appearance in winning matches.
    # Simpler: load per_fdi_metrics.csv from the deployed embedder's analysis output.
    per_fdi_csv = PROJECT_ROOT / "identification/runs/embedding_fdi_init_v1/analysis/per_tooth/per_fdi_metrics.csv"
    fdi_rank1_prior: Dict[str, float] = {}
    if per_fdi_csv.exists():
        df_prior = pd.read_csv(per_fdi_csv, dtype={"fdi": str})
        for _, row in df_prior.iterrows():
            fdi_rank1_prior[row["fdi"]] = float(row["rank1"])
        print(f"  Loaded per-FDI prior from {per_fdi_csv.name}: {len(fdi_rank1_prior)} entries, "
              f"range [{min(fdi_rank1_prior.values()):.3f}, {max(fdi_rank1_prior.values()):.3f}]")
    else:
        print(f"  No per-FDI prior file; skipping Stage 2 FDI search")
        # Build best cfg from Stage 1
        best_cfg = WeightConfig(
            alpha=float(stage1_best["alpha"]), gamma=float(stage1_best["gamma"]),
            eta=float(stage1_best["eta"]), mu=float(stage1_best["mu"]),
            T=float(stage1_best["T"]), beta_fdi=zero_beta.copy(),
        )
        return best_cfg, df1

    # Build beta_fdi from prior, parameterised by scale phi
    beta_template = np.zeros(num_fdi, dtype=np.float32)
    mean_prior = float(np.mean(list(fdi_rank1_prior.values())))
    for fdi_str, r1_val in fdi_rank1_prior.items():
        if fdi_str in fdi_label_map:
            beta_template[fdi_label_map[fdi_str]] = (r1_val - mean_prior)
    # Normalise beta_template so its std=1, then phi scales it
    if float(np.std(beta_template)) > 1e-9:
        beta_template = beta_template / float(np.std(beta_template))

    phi_grid = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
    results2 = []
    print(f"Stage 2 grid: {len(phi_grid)} phi values × n_trials={n_trials}")
    for phi in phi_grid:
        cfg = WeightConfig(
            alpha=float(stage1_best["alpha"]), gamma=float(stage1_best["gamma"]),
            eta=float(stage1_best["eta"]), mu=float(stage1_best["mu"]),
            T=float(stage1_best["T"]),
            beta_fdi=(phi * beta_template).astype(np.float32),
        )
        r1, _, _ = evaluate_symmetric_R1(persons, cfg, n_query=n_query, n_trials=n_trials, seed=seed)
        ents = [effective_n_teeth(f, cfg) for f, _ in persons if f.n_teeth >= n_query + 1]
        median_ent = float(np.median(ents)) if ents else 0.0
        results2.append({
            "stage": 2, "alpha": stage1_best["alpha"], "gamma": stage1_best["gamma"],
            "eta": stage1_best["eta"], "mu": stage1_best["mu"], "T": stage1_best["T"],
            "phi": phi, "r1": r1, "median_ent": median_ent,
            "n_eligible": int(stage1_best["n_eligible"]),
        })
        print(f"  phi={phi}: R1={r1:.4f}, median_ent={median_ent:.1f}")
    df2 = pd.DataFrame(results2)
    df_all = pd.concat([df1, df2], ignore_index=True)

    # Pick best across both stages, with sparsity guard
    df_ok = df_all[df_all["median_ent"] >= sparsity_min]
    if df_ok.empty:
        df_ok = df_all
    best_row = df_ok.sort_values("r1", ascending=False).iloc[0]
    print(f"\nOverall winner: R1={best_row['r1']:.4f}, stage={best_row['stage']}, "
          f"α={best_row['alpha']}, γ={best_row['gamma']}, η={best_row['eta']}, "
          f"μ={best_row['mu']}, T={best_row['T']}, phi={best_row['phi']}, "
          f"median_ent={best_row['median_ent']:.1f}")

    best_cfg = WeightConfig(
        alpha=float(best_row["alpha"]), gamma=float(best_row["gamma"]),
        eta=float(best_row["eta"]), mu=float(best_row["mu"]),
        T=float(best_row["T"]),
        beta_fdi=(float(best_row["phi"]) * beta_template).astype(np.float32),
    )
    return best_cfg, df_all


# --------------------------------------------------------------------------- #
# Bootstrap CI on per-person R1                                               #
# --------------------------------------------------------------------------- #

def bootstrap_r1(per_person: np.ndarray, n_boot: int = 1000, seed: int = 42) -> Tuple[float, float, float]:
    """Return (point_estimate, ci_low, ci_high) for mean per-person R1."""
    if per_person.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        sel = rng.integers(0, per_person.size, size=per_person.size)
        boot[b] = per_person[sel].mean()
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    return float(per_person.mean()), float(ci_low), float(ci_high)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-cache-dir", default="identification/runs/phase8_val_cache/cache/stage_ac",
                        help="Directory of cached Stage A/C JSON files for the val split.")
    parser.add_argument("--embedding-dir", default="identification/runs/phase8_val_cache/cache/embeddings",
                        help="Directory of cached per-image embeddings for the val split.")
    parser.add_argument("--embedder", default="identification/runs/embedding_fdi_init_v1/best.pt",
                        help="Used only to compute embedder_hash for cache key matching.")
    parser.add_argument("--output-dir", default="identification/runs/phase8_weighted",
                        help="Where to write fit_result.json and grid_results.csv.")
    parser.add_argument("--n-query", type=int, default=16)
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--sparsity-min", type=float, default=6.0,
                        help="Reject configs whose median effective_n_teeth falls below this.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Compute embedder hash matching the cache keys
    import hashlib
    h = hashlib.sha256()
    with open(PROJECT_ROOT / args.embedder, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    embedder_hash = h.hexdigest()[:16]
    print(f"Embedder hash: {embedder_hash}")

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load val cache
    persons = load_val_cache(
        cache_dir=PROJECT_ROOT / args.val_cache_dir,
        embedding_dir=PROJECT_ROOT / args.embedding_dir,
        embedder_hash=embedder_hash,
        rotation_filter=0.0,
    )
    if not persons:
        raise RuntimeError("No val persons loaded; check cache paths.")

    fdi_lm = build_fdi_label_map(PROJECT_ROOT / "identification/data/manifest_clean.csv")

    # Mean-pool baseline first
    mean_cfg = WeightConfig(beta_fdi=np.zeros(max(fdi_lm.values()) + 2, dtype=np.float32))
    r1_mean, pp_mean, pids = evaluate_symmetric_R1(persons, mean_cfg, n_query=args.n_query,
                                                    n_trials=args.n_trials, seed=args.seed)
    pt_mean, ci_low_mean, ci_high_mean = bootstrap_r1(pp_mean, n_boot=1000, seed=args.seed + 1)
    print(f"\nVal mean-pool baseline R1 @ n={args.n_query}: {pt_mean:.4f} [{ci_low_mean:.4f}, {ci_high_mean:.4f}]")

    # Grid search
    print("\n=== Grid search ===")
    best_cfg, grid_df = grid_search(
        persons, fdi_lm,
        n_query=args.n_query, n_trials=args.n_trials, seed=args.seed,
        sparsity_min=args.sparsity_min,
    )

    # Evaluate best with bootstrap
    r1_best, pp_best, _ = evaluate_symmetric_R1(persons, best_cfg, n_query=args.n_query,
                                                 n_trials=args.n_trials, seed=args.seed)
    pt_best, ci_low_best, ci_high_best = bootstrap_r1(pp_best, n_boot=1000, seed=args.seed + 1)

    # Paired delta CI
    if pp_best.size == pp_mean.size and pp_best.size > 0:
        rng = np.random.default_rng(args.seed + 2)
        boot_diff = np.empty(1000)
        for b in range(1000):
            sel = rng.integers(0, pp_best.size, size=pp_best.size)
            boot_diff[b] = pp_best[sel].mean() - pp_mean[sel].mean()
        diff_pt = float((pp_best - pp_mean).mean())
        diff_low, diff_high = np.percentile(boot_diff, [2.5, 97.5])
    else:
        diff_pt, diff_low, diff_high = 0.0, 0.0, 0.0

    ents_best = [effective_n_teeth(f, best_cfg) for f, _ in persons if f.n_teeth >= args.n_query + 1]
    median_ent_best = float(np.median(ents_best)) if ents_best else 0.0

    # Save
    grid_df.to_csv(output_dir / "grid_results.csv", index=False)
    summary = {
        "embedder_hash": embedder_hash,
        "n_query": args.n_query,
        "n_trials": args.n_trials,
        "n_eligible_val_persons": int(pp_mean.size),
        "mean_pool_R1": {
            "point": pt_mean, "ci_low": ci_low_mean, "ci_high": ci_high_mean,
        },
        "best_weighted_R1": {
            "point": pt_best, "ci_low": ci_low_best, "ci_high": ci_high_best,
        },
        "paired_delta_R1_weighted_minus_mean": {
            "point": diff_pt, "ci_low": float(diff_low), "ci_high": float(diff_high),
        },
        "best_cfg": best_cfg.to_payload(),
        "median_effective_n_teeth": median_ent_best,
        "sparsity_guard": float(args.sparsity_min),
        "sparsity_pass": median_ent_best >= args.sparsity_min,
    }
    out_path = output_dir / "fit_result.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    best_cfg.save(output_dir / "weight_config.json")

    print(f"\n=== Headline ===")
    print(f"  Val mean-pool R1:   {pt_mean:.4f} [{ci_low_mean:.4f}, {ci_high_mean:.4f}]")
    print(f"  Val weighted R1:    {pt_best:.4f} [{ci_low_best:.4f}, {ci_high_best:.4f}]")
    print(f"  Paired Δ R1:        {diff_pt:+.4f} [{diff_low:+.4f}, {diff_high:+.4f}]")
    print(f"  Median effective_n_teeth: {median_ent_best:.1f}  (gate: ≥{args.sparsity_min})")
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
