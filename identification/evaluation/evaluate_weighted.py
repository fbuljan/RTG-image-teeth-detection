"""
Phase 8.5 — TEST-time evaluation of confidence-weighted aggregation.

This script applies a fitted ``WeightConfig`` (from
``identification/runs/phase8_weighted/weight_config.json``) to the cached test
Stage A/C payloads and per-image embeddings produced by Phase 8.0
(``identification/runs/phase8_baseline/cache/``). It reports R1/R5/R10 with
bootstrap CIs for two regimes that mirror ``evaluate_pipeline.py``:

  1. Symmetric-pair sweep (queries vs galleries, both built from the SAME
     person's teeth, disjoint subsets) — replaces ``_mean_pool`` with
     ``weighted_person_embedding``.
  2. Full-registry sweep (query vs the deployed 1,178-person registry)
     against either the GT-built registry (``identification/registry/``) or the
     YOLO-built registry
     (``identification/registry_ensemble_yolo/embedding_fdi_init_v1/``).

CRITICAL: this v1 of the script is the cheap *asymmetric* probe. The query
side is aggregated with the new weighted formula but the registry is left as
the deployed mean-pool registry. The Phase 8.5 design-review warned that this
creates a distribution mismatch and the symmetric re-aggregation is required
before declaring a pass. The script logs this fact prominently.

To support the pre-registered paired-bootstrap criterion (3), the same
permutations + bootstrap seed are used to evaluate the MEAN-POOL baseline on
the very same persons. The result is written to ``paired_delta_vs_mean.json``.

Outputs (under ``--output-dir``):
  - ``yolo_eval.json``              (same shape as Phase 8.0 yolo_eval.json)
  - ``mean_pool_eval.json``         (paired comparator)
  - ``paired_delta_vs_mean.json``   (Δ R1 with bootstrap CI per n_query)

Usage:
    python -m identification.evaluation.evaluate_weighted \
        --weight-config identification/runs/phase8_weighted/weight_config.json \
        --registry-source yolo \
        --output-dir identification/runs/phase8_weighted/eval_test
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

warnings.filterwarnings(
    "ignore", message=".*encountered in matmul.*", category=RuntimeWarning,
)

from identification.evaluation.weighted_aggregation import (
    PerToothFeatures,
    WeightConfig,
    build_fdi_label_map,
    compute_weights,
    effective_n_teeth,
    extract_features,
    weighted_person_embedding,
)
from identification.models.retrieval_index import RetrievalIndex

PROJECT_ROOT = Path(__file__).resolve().parents[2]
print = functools.partial(print, flush=True)


# --------------------------------------------------------------------------- #
# Cache loading                                                               #
# --------------------------------------------------------------------------- #

def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def load_test_cache(
    cache_dir: Path,
    embedding_dir: Path,
    embedder_hash: str,
    rotation_filter: float = 0.0,
) -> List[Tuple[PerToothFeatures, np.ndarray]]:
    """Load upright Stage A/C payloads + cached per-tooth embeddings.

    One (features, per_tooth_embs) tuple per person. Filters by the rotation
    token embedded in the cache filename (``rot+00000`` for upright).
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
        emb_path = embedding_dir / f"{f.stem}__emb{embedder_hash}.npy"
        if not emb_path.exists():
            skipped += 1
            continue
        embs = np.load(emb_path)
        if embs.shape[0] != features.n_teeth:
            print(
                f"  skipping {features.image_id}: emb={embs.shape[0]} "
                f"!= features={features.n_teeth}"
            )
            skipped += 1
            continue
        out.append((features, embs.astype(np.float32)))
    print(f"Loaded {len(out)} test persons from {cache_dir} (skipped {skipped})")
    return out


# --------------------------------------------------------------------------- #
# Subset aggregation under a WeightConfig                                     #
# --------------------------------------------------------------------------- #

def _aggregate_subset(
    features: PerToothFeatures,
    embs: np.ndarray,
    idx: np.ndarray,
    cfg: WeightConfig,
) -> np.ndarray:
    """Aggregate a subset of teeth (by index) into a single L2-normalised emb."""
    if len(idx) == 0:
        return np.zeros(embs.shape[1], dtype=np.float32)
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
    return weighted_person_embedding(embs[idx], sub, cfg)


def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v.astype(np.float32)
    return (v / n).astype(np.float32)


# --------------------------------------------------------------------------- #
# Paired permutation sampling (shared across cfgs for like-for-like bootstrap) #
# --------------------------------------------------------------------------- #

def _draw_paired_permutations(
    persons: List[Tuple[PerToothFeatures, np.ndarray]],
    n_query: int,
    n_trials: int,
    rng: np.random.Generator,
) -> Tuple[List[str], List[Dict[str, np.ndarray]]]:
    """Return (eligible_pids, per-trial pid→permutation).

    A person is eligible if it has >= n_query + 1 teeth. Permutations are
    drawn ONCE and reused by every WeightConfig under test so the paired
    bootstrap is comparing the same tooth assignments per person per trial.
    """
    eligible = [(f, e) for f, e in persons if f.n_teeth >= n_query + 1]
    pids = [f.person_id for f, _ in eligible]
    person_by_pid = {f.person_id: (f, e) for f, e in eligible}
    trials: List[Dict[str, np.ndarray]] = []
    for _ in range(n_trials):
        trial: Dict[str, np.ndarray] = {}
        for pid in pids:
            f, _ = person_by_pid[pid]
            trial[pid] = rng.permutation(f.n_teeth)
        trials.append(trial)
    return pids, trials


# --------------------------------------------------------------------------- #
# Symmetric-pair sweep under arbitrary cfg                                    #
# --------------------------------------------------------------------------- #

def _sweep_symmetric(
    persons: List[Tuple[PerToothFeatures, np.ndarray]],
    cfg: WeightConfig,
    n_query_list: List[int],
    n_trials: int,
    sweep_rng: np.random.Generator,
    bootstrap_rng: np.random.Generator,
    shared_perms: Dict[int, Tuple[List[str], List[Dict[str, np.ndarray]]]] | None = None,
) -> Tuple[List[dict], Dict[int, Tuple[List[str], List[Dict[str, np.ndarray]]]]]:
    """Symmetric sweep returning per-n_query results + the permutations used.

    Pass ``shared_perms`` to reuse permutations drawn by a prior call (e.g.
    weighted then mean-pool) — this is what makes the paired delta valid.
    """
    person_by_pid = {f.person_id: (f, e) for f, e in persons}
    results: List[dict] = []
    perms_out: Dict[int, Tuple[List[str], List[Dict[str, np.ndarray]]]] = {}
    for n_q in n_query_list:
        if shared_perms is not None and n_q in shared_perms:
            pids, trial_perms = shared_perms[n_q]
        else:
            pids, trial_perms = _draw_paired_permutations(
                persons, n_q, n_trials, sweep_rng,
            )
        perms_out[n_q] = (pids, trial_perms)
        n_eligible = len(pids)
        if n_eligible < 5:
            results.append({"n_query": n_q, "n_persons": n_eligible, "skipped": True})
            continue

        match_r1 = np.zeros((len(trial_perms), n_eligible), dtype=bool)
        match_r5 = np.zeros_like(match_r1)
        match_r10 = np.zeros_like(match_r1)
        ap_arr = np.zeros((len(trial_perms), n_eligible), dtype=np.float64)

        for t, perm in enumerate(trial_perms):
            queries, galleries = [], []
            for pid in pids:
                f, e = person_by_pid[pid]
                idx = perm[pid]
                q_idx = idx[:n_q]
                g_idx = idx[n_q:]
                queries.append(_aggregate_subset(f, e, q_idx, cfg))
                galleries.append(_aggregate_subset(f, e, g_idx, cfg))
            Q = np.stack(queries)
            G = np.stack(galleries)
            if not (np.isfinite(Q).all() and np.isfinite(G).all()):
                raise RuntimeError(
                    f"sym sweep n_q={n_q} trial={t}: non-finite Q/G"
                )
            sim = Q @ G.T
            if not np.isfinite(sim).all():
                raise RuntimeError(
                    f"sym sweep n_q={n_q} trial={t}: non-finite sim"
                )
            ranked = np.argsort(-sim, axis=1)
            pids_arr = np.array(pids)
            mat = pids_arr[ranked] == pids_arr[:, None]
            match_r1[t] = mat[:, 0]
            match_r5[t] = mat[:, :5].any(axis=1)
            match_r10[t] = mat[:, :10].any(axis=1)
            first_pos = np.argmax(mat, axis=1)
            valid = mat.any(axis=1)
            ap_arr[t] = np.where(valid, 1.0 / (first_pos + 1), 0.0)

        per_person_r1 = match_r1.mean(axis=0)
        per_person_ap = ap_arr.mean(axis=0)
        rank1 = float(per_person_r1.mean())
        rank5 = float(match_r5.mean(axis=0).mean())
        rank10 = float(match_r10.mean(axis=0).mean())
        mAP = float(per_person_ap.mean())

        n_boot = 1000
        boot_r1 = np.empty(n_boot)
        for b in range(n_boot):
            sel = bootstrap_rng.integers(0, n_eligible, size=n_eligible)
            boot_r1[b] = per_person_r1[sel].mean()
        ci_low, ci_high = np.percentile(boot_r1, [2.5, 97.5])

        results.append({
            "n_query": n_q,
            "method": "weighted" if not cfg.is_mean_pool() else "mean",
            "n_persons": n_eligible,
            "n_trials": len(trial_perms),
            "rank1_mean": rank1,
            "rank1_ci95_low": float(ci_low),
            "rank1_ci95_high": float(ci_high),
            "rank5_mean": rank5,
            "rank10_mean": rank10,
            "mAP_mean": mAP,
            "per_person_r1": per_person_r1.tolist(),
            "pids": pids,
        })
    return results, perms_out


# --------------------------------------------------------------------------- #
# Full-registry sweep under arbitrary cfg                                     #
# --------------------------------------------------------------------------- #

def _sweep_full_registry(
    persons: List[Tuple[PerToothFeatures, np.ndarray]],
    cfg: WeightConfig,
    registry_index: RetrievalIndex,
    n_query_list: List[int],
    perms_by_nq: Dict[int, Tuple[List[str], List[Dict[str, np.ndarray]]]],
    fallback_rng: np.random.Generator,
    bootstrap_rng: np.random.Generator,
) -> List[dict]:
    """Query the deployed registry. Reuses the symmetric sweep's permutations.

    NOTE: the registry was built with mean-pool. This is the *asymmetric*
    probe — see module docstring.
    """
    person_by_pid = {f.person_id: (f, e) for f, e in persons}
    n_reg = len(registry_index)
    results: List[dict] = []
    for n_q in n_query_list:
        eligible = [pid for pid, (f, e) in person_by_pid.items() if f.n_teeth >= n_q]
        if len(eligible) < 5:
            results.append({"n_query": n_q, "n_persons": len(eligible), "skipped": True})
            continue
        # Build pid→perm lookup (eligible set here may differ from sym sweep
        # because we don't need the +1 gallery slack).
        pid_perm_lookup: Dict[str, np.ndarray] = {}
        if n_q in perms_by_nq:
            _, trial_perms = perms_by_nq[n_q]
        else:
            trial_perms = []
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
                f, e = person_by_pid[pid]
                if perm is not None and pid in perm and len(perm[pid]) == f.n_teeth:
                    idx = perm[pid][:n_q]
                else:
                    idx = fallback_rng.permutation(f.n_teeth)[:n_q]
                q = _aggregate_subset(f, e, idx, cfg)
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
        for b in range(n_boot):
            sel = bootstrap_rng.integers(0, n_eligible, size=n_eligible)
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


# --------------------------------------------------------------------------- #
# Paired delta R1                                                             #
# --------------------------------------------------------------------------- #

def _paired_delta(
    weighted_results: List[dict],
    mean_results: List[dict],
    bootstrap_rng: np.random.Generator,
    sweep_label: str,
) -> List[dict]:
    """Paired bootstrap of (weighted − mean) per-person R1, n_query by n_query."""
    out: List[dict] = []
    by_nq_w = {r["n_query"]: r for r in weighted_results if not r.get("skipped")}
    by_nq_m = {r["n_query"]: r for r in mean_results if not r.get("skipped")}
    for n_q in sorted(set(by_nq_w) & set(by_nq_m)):
        w = by_nq_w[n_q]
        m = by_nq_m[n_q]
        common = [p for p in w["pids"] if p in set(m["pids"])]
        if not common:
            out.append({"n_query": n_q, "sweep": sweep_label, "skipped": True})
            continue
        w_map = dict(zip(w["pids"], w["per_person_r1"]))
        m_map = dict(zip(m["pids"], m["per_person_r1"]))
        delta = np.array([w_map[p] - m_map[p] for p in common], dtype=np.float64)
        n = len(common)
        n_boot = 1000
        boot = np.empty(n_boot)
        for b in range(n_boot):
            sel = bootstrap_rng.integers(0, n, size=n)
            boot[b] = delta[sel].mean()
        ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
        out.append({
            "n_query": n_q,
            "sweep": sweep_label,
            "n_persons_paired": n,
            "delta_r1_mean": float(delta.mean()),
            "delta_r1_ci95_low": float(ci_low),
            "delta_r1_ci95_high": float(ci_high),
            "weighted_r1": w["rank1_mean"],
            "mean_r1": m["rank1_mean"],
        })
    return out


# --------------------------------------------------------------------------- #
# Sparsity diagnostic                                                          #
# --------------------------------------------------------------------------- #

def _sparsity_summary(
    persons: List[Tuple[PerToothFeatures, np.ndarray]],
    cfg: WeightConfig,
    n_query: int = 16,
) -> dict:
    ents = [
        effective_n_teeth(f, cfg)
        for f, _ in persons if f.n_teeth >= n_query + 1
    ]
    if not ents:
        return {"n": 0, "median": 0.0, "p10": 0.0, "p90": 0.0}
    return {
        "n": len(ents),
        "median": float(np.median(ents)),
        "p10": float(np.percentile(ents, 10)),
        "p90": float(np.percentile(ents, 90)),
    }


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

REGISTRY_SOURCES = {
    "yolo": "identification/registry_ensemble_yolo/embedding_fdi_init_v1",
    "gt": "identification/registry",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weight-config",
        default="identification/runs/phase8_weighted/weight_config.json",
        help="Path to fitted WeightConfig JSON (output of fit_aggregator.py).",
    )
    parser.add_argument(
        "--cache-dir",
        default="identification/runs/phase8_baseline/cache/stage_ac",
        help="Stage A/C cache dir (one JSON per (image, rotation)).",
    )
    parser.add_argument(
        "--embedding-dir",
        default="identification/runs/phase8_baseline/cache/embeddings",
        help="Per-image embedding cache dir (one .npy per (image, rotation, embedder)).",
    )
    parser.add_argument(
        "--embedder",
        default="identification/runs/embedding_fdi_init_v1/best.pt",
        help="Embedder checkpoint (used only for cache-key hash).",
    )
    parser.add_argument(
        "--registry-source",
        choices=list(REGISTRY_SOURCES.keys()),
        default="yolo",
        help="Which registry to query (yolo=82.6%% baseline; gt=GT-built).",
    )
    parser.add_argument(
        "--registry-dir",
        default=None,
        help="Explicit registry dir; overrides --registry-source.",
    )
    parser.add_argument(
        "--output-dir",
        default="identification/runs/phase8_weighted/eval_test",
    )
    parser.add_argument("--n-query-list", type=int, nargs="+",
                        default=[1, 2, 4, 8, 16])
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-full-registry", action="store_true",
        help="Skip the deployed-registry sweep (only do symmetric R1).",
    )
    args = parser.parse_args()

    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load WeightConfig + sanity ---
    weight_cfg_path = (PROJECT_ROOT / args.weight_config).resolve()
    cfg = WeightConfig.load(weight_cfg_path)
    print(f"Loaded WeightConfig from {weight_cfg_path}")
    print(f"  alpha={cfg.alpha}, gamma={cfg.gamma}, eta={cfg.eta}, mu={cfg.mu}, T={cfg.T}")
    print(f"  beta_fdi: dim={cfg.beta_fdi.shape[0]}, |max|={np.abs(cfg.beta_fdi).max():.3f}")

    # --- Embedder hash for cache key matching ---
    embedder_path = (PROJECT_ROOT / args.embedder).resolve()
    embedder_hash = _file_hash(embedder_path)
    print(f"Embedder hash: {embedder_hash} ({embedder_path.name})")

    # --- Load test cache ---
    persons = load_test_cache(
        cache_dir=(PROJECT_ROOT / args.cache_dir).resolve(),
        embedding_dir=(PROJECT_ROOT / args.embedding_dir).resolve(),
        embedder_hash=embedder_hash,
        rotation_filter=0.0,
    )
    if not persons:
        raise RuntimeError("No test persons loaded; check cache paths.")

    # --- RNGs (split into independent streams so seeds are stable) ---
    seed_seq = np.random.SeedSequence(args.seed)
    sweep_rng, fallback_rng, bootstrap_rng, bootstrap_rng_paired = (
        np.random.default_rng(s) for s in seed_seq.spawn(4)
    )

    # --- Symmetric weighted sweep (also captures shared permutations) ---
    print("\n[weighted] symmetric sweep...")
    t0 = time.perf_counter()
    weighted_sym, shared_perms = _sweep_symmetric(
        persons, cfg, args.n_query_list, args.n_trials,
        sweep_rng, bootstrap_rng, shared_perms=None,
    )
    for s in weighted_sym:
        if s.get("skipped"):
            continue
        print(
            f"  sym n={s['n_query']:>2}: R1={s['rank1_mean']:.4f} "
            f"[{s['rank1_ci95_low']:.3f}, {s['rank1_ci95_high']:.3f}], "
            f"R5={s['rank5_mean']:.4f}, mAP={s['mAP_mean']:.4f}"
        )

    # --- Symmetric mean-pool comparator on SHARED PERMS for paired Δ ---
    print("\n[mean-pool comparator] symmetric sweep (shared permutations)...")
    num_fdi = cfg.beta_fdi.shape[0]
    mean_cfg = WeightConfig(beta_fdi=np.zeros(num_fdi, dtype=np.float32))
    # Re-seed bootstrap_rng for mean so its CIs are independent of the
    # weighted sweep's bootstrap stream BUT the per-person R1 is paired via
    # shared_perms. We deliberately spawn a separate stream for the mean
    # sweep's per-person bootstrap.
    bootstrap_rng_mean = np.random.default_rng(args.seed + 9001)
    mean_sym, _ = _sweep_symmetric(
        persons, mean_cfg, args.n_query_list, args.n_trials,
        sweep_rng, bootstrap_rng_mean, shared_perms=shared_perms,
    )
    for s in mean_sym:
        if s.get("skipped"):
            continue
        print(
            f"  sym n={s['n_query']:>2}: R1={s['rank1_mean']:.4f} "
            f"[{s['rank1_ci95_low']:.3f}, {s['rank1_ci95_high']:.3f}]"
        )

    # --- Full-registry sweep (optional, slow) ---
    sweep_reg_weighted: List[dict] = []
    sweep_reg_mean: List[dict] = []
    paired_delta_reg: List[dict] = []
    registry_dir = None
    n_registry = 0
    if not args.skip_full_registry:
        if args.registry_dir is not None:
            registry_dir = (PROJECT_ROOT / args.registry_dir).resolve()
        else:
            registry_dir = (PROJECT_ROOT / REGISTRY_SOURCES[args.registry_source]).resolve()
        index_stem = registry_dir / "index"
        if not (index_stem.with_suffix(".faiss")).exists():
            raise FileNotFoundError(
                f"Registry index missing: {index_stem.with_suffix('.faiss')}"
            )
        registry_index = RetrievalIndex.load(str(index_stem))
        n_registry = len(registry_index)
        print(f"\n[registry] {args.registry_source}: {registry_dir} "
              f"({n_registry} persons)")

        print("[weighted] full-registry sweep (ASYMMETRIC: registry is mean-pool)...")
        sweep_reg_weighted = _sweep_full_registry(
            persons, cfg, registry_index, args.n_query_list,
            shared_perms, fallback_rng, bootstrap_rng,
        )
        for s in sweep_reg_weighted:
            if s.get("skipped"):
                continue
            print(
                f"  reg n={s['n_query']:>2}: R1={s['rank1_mean']:.4f} "
                f"[{s['rank1_ci95_low']:.3f}, {s['rank1_ci95_high']:.3f}], "
                f"sim_med={s['sim_top1_median']:.3f}"
            )

        print("[mean-pool comparator] full-registry sweep (shared permutations)...")
        fallback_rng_mean = np.random.default_rng(args.seed + 7777)
        sweep_reg_mean = _sweep_full_registry(
            persons, mean_cfg, registry_index, args.n_query_list,
            shared_perms, fallback_rng_mean, bootstrap_rng_mean,
        )
        for s in sweep_reg_mean:
            if s.get("skipped"):
                continue
            print(
                f"  reg n={s['n_query']:>2}: R1={s['rank1_mean']:.4f} "
                f"[{s['rank1_ci95_low']:.3f}, {s['rank1_ci95_high']:.3f}]"
            )

        paired_delta_reg = _paired_delta(
            sweep_reg_weighted, sweep_reg_mean,
            bootstrap_rng_paired, sweep_label="full_registry",
        )

    paired_delta_sym = _paired_delta(
        weighted_sym, mean_sym,
        bootstrap_rng_paired, sweep_label="symmetric",
    )

    # --- Sparsity diagnostics ---
    sparsity = _sparsity_summary(persons, cfg, n_query=16)
    sparsity_mean = _sparsity_summary(persons, mean_cfg, n_query=16)
    print(f"\nSparsity (effective_n_teeth, n_query=16, eligible persons):")
    print(f"  weighted: median={sparsity['median']:.1f} "
          f"[p10={sparsity['p10']:.1f}, p90={sparsity['p90']:.1f}]  n={sparsity['n']}")
    print(f"  mean:     median={sparsity_mean['median']:.1f}  (sanity: should ≈ teeth/person)")

    # --- Write yolo_eval.json (weighted, same shape as Phase 8.0) ---
    elapsed = time.perf_counter() - t0
    weighted_payload = {
        "label": "phase8.5_weighted_eval",
        "rotation_deg": 0.0,
        "asymmetric_registry": True,
        "asymmetric_registry_note": (
            "Registry is mean-pool; only the QUERY side uses weighted aggregation. "
            "Per Phase 8.5 design-review, a symmetric re-aggregation of the "
            "registry is required before the pre-registered pass criteria can "
            "be evaluated. This file is the cheap first-cut probe."
        ),
        "n_persons_usable": len(persons),
        "mean_teeth_per_person": float(
            np.mean([f.n_teeth for f, _ in persons])
        ) if persons else 0.0,
        "embedder_hash": embedder_hash,
        "embedder_checkpoint": str(embedder_path.relative_to(PROJECT_ROOT)),
        "weight_config_path": str(weight_cfg_path.relative_to(PROJECT_ROOT)),
        "weight_config": cfg.to_payload(),
        "registry_source": args.registry_source if not args.skip_full_registry else None,
        "registry_dir": (
            str(registry_dir.relative_to(PROJECT_ROOT))
            if registry_dir is not None else None
        ),
        "registry_size": n_registry,
        "n_query_list": args.n_query_list,
        "n_trials": args.n_trials,
        "sweep_symmetric": weighted_sym,
        "sweep_full_registry": sweep_reg_weighted,
        "sparsity_effective_n_teeth": sparsity,
        "elapsed_s": elapsed,
    }
    out_yolo = output_dir / "yolo_eval.json"
    with open(out_yolo, "w") as f:
        json.dump(weighted_payload, f, indent=2)
    print(f"\n[weighted] saved → {out_yolo}")

    # --- Write mean_pool_eval.json (comparator, same shape) ---
    mean_payload = {
        "label": "phase8.5_mean_pool_comparator",
        "rotation_deg": 0.0,
        "n_persons_usable": len(persons),
        "embedder_hash": embedder_hash,
        "embedder_checkpoint": str(embedder_path.relative_to(PROJECT_ROOT)),
        "registry_source": args.registry_source if not args.skip_full_registry else None,
        "registry_dir": (
            str(registry_dir.relative_to(PROJECT_ROOT))
            if registry_dir is not None else None
        ),
        "registry_size": n_registry,
        "n_query_list": args.n_query_list,
        "n_trials": args.n_trials,
        "sweep_symmetric": mean_sym,
        "sweep_full_registry": sweep_reg_mean,
        "sparsity_effective_n_teeth": sparsity_mean,
    }
    out_mean = output_dir / "mean_pool_eval.json"
    with open(out_mean, "w") as f:
        json.dump(mean_payload, f, indent=2)
    print(f"[mean-pool comparator] saved → {out_mean}")

    # --- Write paired_delta_vs_mean.json ---
    paired_payload = {
        "label": "phase8.5_paired_delta",
        "n_persons_usable": len(persons),
        "weight_config_path": str(weight_cfg_path.relative_to(PROJECT_ROOT)),
        "registry_source": args.registry_source if not args.skip_full_registry else None,
        "registry_dir": (
            str(registry_dir.relative_to(PROJECT_ROOT))
            if registry_dir is not None else None
        ),
        "asymmetric_registry": True,
        "asymmetric_registry_note": (
            "Full-registry Δ uses an asymmetric (weighted query vs mean-pool "
            "registry) comparison. Treat this as a sanity probe, not a "
            "pre-registered-criterion result."
        ),
        "n_query_list": args.n_query_list,
        "paired_delta_symmetric": paired_delta_sym,
        "paired_delta_full_registry": paired_delta_reg,
    }
    out_paired = output_dir / "paired_delta_vs_mean.json"
    with open(out_paired, "w") as f:
        json.dump(paired_payload, f, indent=2)
    print(f"[paired Δ] saved → {out_paired}")

    # --- Headline ---
    def _grab_r1(results: List[dict], n_q: int) -> str:
        for r in results:
            if r.get("n_query") == n_q and not r.get("skipped"):
                return (
                    f"{r['rank1_mean']:.4f} "
                    f"[{r['rank1_ci95_low']:.3f}, {r['rank1_ci95_high']:.3f}]"
                )
        return "—"

    print("\n=== Headline (n_query=16) ===")
    print(f"  symmetric weighted R1: {_grab_r1(weighted_sym, 16)}")
    print(f"  symmetric mean R1:     {_grab_r1(mean_sym, 16)}")
    for d in paired_delta_sym:
        if d.get("n_query") == 16:
            print(f"  paired Δ (sym):        {d['delta_r1_mean']:+.4f} "
                  f"[{d['delta_r1_ci95_low']:+.4f}, {d['delta_r1_ci95_high']:+.4f}]")
    if not args.skip_full_registry:
        print(f"  full-reg weighted R1:  {_grab_r1(sweep_reg_weighted, 16)}")
        print(f"  full-reg mean R1:      {_grab_r1(sweep_reg_mean, 16)}")
        for d in paired_delta_reg:
            if d.get("n_query") == 16:
                print(f"  paired Δ (reg):        {d['delta_r1_mean']:+.4f} "
                      f"[{d['delta_r1_ci95_low']:+.4f}, {d['delta_r1_ci95_high']:+.4f}]")
    print(f"  sparsity median (weighted, n_q=16): {sparsity['median']:.1f}")
    print(f"\n[!] ASYMMETRIC REGISTRY: the full-registry numbers above are the "
          f"cheap first-cut probe; the deployed registry was built with "
          f"mean-pool. A symmetric re-aggregation is required before the "
          f"Phase 8.5 pre-registered criteria can be evaluated.")


if __name__ == "__main__":
    main()
