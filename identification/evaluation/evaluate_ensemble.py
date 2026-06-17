"""Score-level ensemble of all four trained embedders.

For each of the four embedders (raw / masked / metadata / FDI-init), extract
embeddings on the eval split, then run the standard symmetric multi-tooth
sweep and the forensic 1-vs-aggregated protocol with several combination
strategies on the per-model similarity matrices:

    - mean:       uniform-weighted average of per-model cosine similarities
    - max:        per-pair max across models
    - weighted:   grid-searched weights (best on val), applied to test
    - borda:      rank-fusion (sum of within-model ranks; smaller = better)

Saves a metrics.json compatible with the existing person_retrieval format so
the model card / figure generator can read it without changes.

Usage:
    python -m identification.evaluation.evaluate_ensemble
    python -m identification.evaluation.evaluate_ensemble --split test
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from identification.evaluation.evaluate_embedding import (
    build_eval_dataset,
    extract_embeddings,
    load_checkpoint,
)
from identification.models.embedding_model import ToothEmbeddingModelWithMetadata
from identification.models.person_aggregation import aggregate_mean

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINTS: dict[str, str] = {
    "baseline": "identification/runs/embedding_triplet_v1/best.pt",
    "masked": "identification/runs/embedding_triplet_masked_v1/best.pt",
    "metadata": "identification/runs/embedding_metadata_v1/best.pt",
    "fdi_init": "identification/runs/embedding_fdi_init_v1/best.pt",
}


# ---------------------------------------------------------------------------
# Embedding extraction (aligned across models)
# ---------------------------------------------------------------------------


def build_canonical_label_map(manifest_path: str, split: str) -> dict[str, int]:
    """Person_id -> contiguous int, sorted, computed once and shared by all models."""
    df = pd.read_csv(manifest_path, dtype=str)
    df = df[df["split"] == split]
    persons = sorted(df["person_id"].unique().tolist())
    return {pid: i for i, pid in enumerate(persons)}


def extract_all_models(
    split: str,
    device: str,
    batch_size: int = 64,
    manifest_override: str | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Run each checkpoint on the same split and return aligned per-model embeddings.

    Returns:
        embeddings_by_model: {model_name: (N, 128)} all using the same row order.
        labels: (N,) integer person ID per row (from the canonical label_map).
    """
    # Use the first checkpoint's manifest path as the canonical one — all four
    # were trained on identical splits so this is safe.
    first_ckpt_path = next(iter(CHECKPOINTS.values()))
    _, cfg, _, _ = load_checkpoint(str(PROJECT_ROOT / first_ckpt_path), device)
    manifest_path = manifest_override or cfg["data"]["manifest"]
    label_map = build_canonical_label_map(manifest_path, split)

    embeddings_by_model: dict[str, np.ndarray] = {}
    labels_ref: np.ndarray | None = None

    for name, ckpt_path in CHECKPOINTS.items():
        print(f"\n=== Extracting embeddings: {name} ===")
        full_path = str(PROJECT_ROOT / ckpt_path)
        model, cfg, _, ckpt = load_checkpoint(full_path, device)
        uses_metadata = isinstance(model, ToothEmbeddingModelWithMetadata)
        dataset = build_eval_dataset(cfg, split, label_map, ckpt, uses_metadata,
                                       manifest_override=manifest_override)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        emb, lab = extract_embeddings(model, loader, device)
        embeddings_by_model[name] = emb
        if labels_ref is None:
            labels_ref = lab
        else:
            # The NaN-filter in extract_embeddings could in principle drop different
            # rows per model. Detect that and bail out — we need aligned arrays.
            if len(lab) != len(labels_ref) or not np.array_equal(lab, labels_ref):
                raise RuntimeError(
                    f"Embedding row alignment broke for model '{name}'. "
                    "NaN-filtering dropped different samples; aborting."
                )
        # Free model from MPS memory
        del model
        if device == "mps":
            torch.mps.empty_cache()

    assert labels_ref is not None
    return embeddings_by_model, labels_ref


# ---------------------------------------------------------------------------
# Per-model aggregation into (queries, gallery_profiles) matrices
# ---------------------------------------------------------------------------


def aggregate_split(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_query: int,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Produce per-person aggregated query and gallery vectors via mean pooling.

    Returns query_embs, gallery_embs, person_labels — or None if no person has
    enough teeth.
    """
    unique = np.unique(labels)
    eligible = [p for p in unique if (labels == p).sum() > n_query]
    if len(eligible) < 5:
        return None

    query_embs, gallery_embs, person_labels = [], [], []
    for person in eligible:
        idx = np.where(labels == person)[0]
        shuffled = rng.permutation(idx)
        q_idx, g_idx = shuffled[:n_query], shuffled[n_query:]
        query_embs.append(aggregate_mean(embeddings[q_idx]))
        gallery_embs.append(aggregate_mean(embeddings[g_idx]))
        person_labels.append(person)

    return (
        np.stack(query_embs),
        np.stack(gallery_embs),
        np.asarray(person_labels),
    )


def split_indices(
    labels: np.ndarray,
    n_query: int,
    rng: np.random.RandomState,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray] | None:
    """Return per-person (query_idx, gallery_idx) so every model uses the same split.

    Returns query_idx_per_person, gallery_idx_per_person, person_labels.
    """
    unique = np.unique(labels)
    eligible = [p for p in unique if (labels == p).sum() > n_query]
    if len(eligible) < 5:
        return None
    q_list, g_list, person_labels = [], [], []
    for person in eligible:
        idx = np.where(labels == person)[0]
        shuffled = rng.permutation(idx)
        q_list.append(shuffled[:n_query])
        g_list.append(shuffled[n_query:])
        person_labels.append(person)
    return q_list, g_list, np.asarray(person_labels)


def aggregate_with_indices(
    embeddings: np.ndarray,
    q_idx_list: Sequence[np.ndarray],
    g_idx_list: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate per-person using fixed indices (shared across models)."""
    q = np.stack([aggregate_mean(embeddings[idx]) for idx in q_idx_list])
    g = np.stack([aggregate_mean(embeddings[idx]) for idx in g_idx_list])
    return q, g


# ---------------------------------------------------------------------------
# Combination strategies on the per-model similarity matrices
# ---------------------------------------------------------------------------


def combine_mean(sim_matrices: Sequence[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(sim_matrices, axis=0), axis=0)


def combine_max(sim_matrices: Sequence[np.ndarray]) -> np.ndarray:
    return np.max(np.stack(sim_matrices, axis=0), axis=0)


def combine_weighted(sim_matrices: Sequence[np.ndarray], weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / max(weights.sum(), 1e-12)
    stacked = np.stack(sim_matrices, axis=0)  # (M, n, n)
    return np.tensordot(weights, stacked, axes=([0], [0]))


def combine_borda(sim_matrices: Sequence[np.ndarray]) -> np.ndarray:
    """Borda rank fusion. Returns a 'similarity-like' matrix where bigger is better.

    For each model, convert each row to a rank (higher similarity → higher rank score),
    sum across models. The result is in [0, n_rows * n_models] so we can argsort it.
    """
    score = np.zeros_like(sim_matrices[0], dtype=np.float64)
    n = sim_matrices[0].shape[1]
    for sim in sim_matrices:
        ranks = (-sim).argsort(axis=1).argsort(axis=1)  # 0 = best, n-1 = worst
        score += (n - 1 - ranks)  # invert so larger = better
    return score


# ---------------------------------------------------------------------------
# Metric computation given a "combined" similarity-like matrix
# ---------------------------------------------------------------------------


def rank_metrics(combined: np.ndarray, person_labels: np.ndarray) -> dict[str, float]:
    """Compute Rank-1/5/10 + mAP given a (n, n) combined-score matrix.

    The diagonal `combined[i, i]` is the correct match for query i — each person
    has exactly one gallery profile, built from disjoint teeth (so the diagonal
    must NOT be excluded).
    """
    ranked = np.argsort(-combined, axis=1)
    ranked_labels = person_labels[ranked]
    query_col = person_labels[:, None]
    matches = ranked_labels == query_col

    rank1 = float(matches[:, 0].mean())
    rank5 = float(matches[:, :5].any(axis=1).mean())
    rank10 = float(matches[:, :10].any(axis=1).mean())

    first_pos = np.argmax(matches, axis=1)
    valid = matches.any(axis=1)
    aps = np.where(valid, 1.0 / (first_pos + 1), 0.0)
    mAP = float(aps.mean())
    return {"rank1": rank1, "rank5": rank5, "rank10": rank10, "mAP": mAP}


# ---------------------------------------------------------------------------
# Symmetric sweep
# ---------------------------------------------------------------------------


def sweep(
    embeddings_by_model: dict[str, np.ndarray],
    labels: np.ndarray,
    n_query_list: Sequence[int],
    n_trials: int,
    rng_seed: int,
    weights: dict[str, float] | None = None,
) -> dict[str, list[dict]]:
    """Run the symmetric multi-tooth sweep for each combination strategy.

    Returns {strategy_name: [per-n_query result dicts]}.
    """
    rng = np.random.RandomState(rng_seed)
    model_order = list(embeddings_by_model.keys())
    if weights is not None:
        w = np.array([weights[m] for m in model_order], dtype=np.float64)
    else:
        w = None

    strategies = {
        "mean": lambda mats: combine_mean(mats),
        "max": lambda mats: combine_max(mats),
        "borda": lambda mats: combine_borda(mats),
    }
    if w is not None:
        strategies["weighted"] = lambda mats, ww=w: combine_weighted(mats, ww)

    results: dict[str, list[dict]] = {name: [] for name in strategies}

    for n_query in n_query_list:
        per_strategy_trials: dict[str, list[dict]] = {name: [] for name in strategies}
        for trial in range(n_trials):
            split = split_indices(labels, n_query, rng)
            if split is None:
                continue
            q_idx_list, g_idx_list, person_labels = split

            # Per-model aggregated query/gallery + similarity matrix
            sim_matrices: list[np.ndarray] = []
            for name in model_order:
                q, g = aggregate_with_indices(embeddings_by_model[name], q_idx_list, g_idx_list)
                sim_matrices.append(q @ g.T)

            for s_name, combiner in strategies.items():
                combined = combiner(sim_matrices)
                m = rank_metrics(combined, person_labels)
                m["n_persons"] = len(person_labels)
                per_strategy_trials[s_name].append(m)

        for s_name, trial_metrics in per_strategy_trials.items():
            if not trial_metrics:
                continue
            df = pd.DataFrame(trial_metrics)
            results[s_name].append({
                "n_query": int(n_query),
                "method": s_name,
                "n_persons": int(df["n_persons"].iloc[0]),
                "n_trials": int(len(df)),
                "rank1_mean": float(df["rank1"].mean()),
                "rank1_std": float(df["rank1"].std()),
                "rank5_mean": float(df["rank5"].mean()),
                "rank5_std": float(df["rank5"].std()),
                "rank10_mean": float(df["rank10"].mean()),
                "rank10_std": float(df["rank10"].std()),
                "mAP_mean": float(df["mAP"].mean()),
                "mAP_std": float(df["mAP"].std()),
            })

    return results


# ---------------------------------------------------------------------------
# Weight grid search on a held-out split (val)
# ---------------------------------------------------------------------------


def grid_search_weights(
    embeddings_by_model: dict[str, np.ndarray],
    labels: np.ndarray,
    n_query: int,
    n_trials: int,
    rng_seed: int,
) -> dict[str, float]:
    """Brute-force coarse grid over weight tuples that sum to ~1.

    Each weight is in {0.0, 0.1, 0.2, ..., 1.0}; only tuples summing close to 1
    are considered. Picks weights that maximise mean Rank-1 across `n_trials`.
    """
    model_order = list(embeddings_by_model.keys())
    rng = np.random.RandomState(rng_seed)
    # Reuse the same trial splits across all weight candidates so the comparison
    # is paired rather than re-sampled.
    trial_splits: list[tuple[list[np.ndarray], list[np.ndarray], np.ndarray]] = []
    while len(trial_splits) < n_trials:
        s = split_indices(labels, n_query, rng)
        if s is None:
            break
        trial_splits.append(s)
    if not trial_splits:
        return {m: 0.25 for m in model_order}

    # Precompute the 4 sim matrices per trial (so each weight candidate is fast)
    precomp: list[list[np.ndarray]] = []
    person_labels_list: list[np.ndarray] = []
    for q_idx_list, g_idx_list, person_labels in trial_splits:
        sim_matrices: list[np.ndarray] = []
        for name in model_order:
            q, g = aggregate_with_indices(embeddings_by_model[name], q_idx_list, g_idx_list)
            sim_matrices.append(q @ g.T)
        precomp.append(sim_matrices)
        person_labels_list.append(person_labels)

    weight_steps = np.arange(0, 11) / 10.0  # 0.0 .. 1.0 step 0.1
    best: tuple[float, np.ndarray] = (-1.0, np.array([0.25] * len(model_order)))
    for combo in product(weight_steps, repeat=len(model_order)):
        total = sum(combo)
        if abs(total - 1.0) > 1e-6:
            continue
        w = np.array(combo, dtype=np.float64)
        rank1s = []
        for sim_matrices, person_labels in zip(precomp, person_labels_list):
            combined = combine_weighted(sim_matrices, w)
            rank1s.append(rank_metrics(combined, person_labels)["rank1"])
        mean_rank1 = float(np.mean(rank1s))
        if mean_rank1 > best[0]:
            best = (mean_rank1, w)

    return {name: float(weight) for name, weight in zip(model_order, best[1])}


# ---------------------------------------------------------------------------
# Forensic 1-vs-aggregated protocol
# ---------------------------------------------------------------------------


def forensic_one_tooth(
    embeddings_by_model: dict[str, np.ndarray],
    labels: np.ndarray,
    n_trials: int,
    rng_seed: int,
    weights: dict[str, float] | None = None,
) -> dict[str, dict]:
    rng = np.random.RandomState(rng_seed)
    model_order = list(embeddings_by_model.keys())
    if weights is not None:
        w = np.array([weights[m] for m in model_order], dtype=np.float64)
    else:
        w = None

    strategies = {
        "mean": combine_mean,
        "max": combine_max,
        "borda": combine_borda,
    }
    if w is not None:
        strategies["weighted"] = lambda mats, ww=w: combine_weighted(mats, ww)

    per_strategy: dict[str, list[dict]] = {name: [] for name in strategies}

    unique = np.unique(labels)
    eligible = [p for p in unique if (labels == p).sum() >= 2]
    if len(eligible) < 5:
        return {name: {} for name in strategies}

    for trial in range(n_trials):
        # Build trial-specific 1-query and gallery splits, shared across models.
        q_idx_list: list[int] = []
        g_idx_list: list[np.ndarray] = []
        person_labels: list[int] = []
        for person in eligible:
            idx = np.where(labels == person)[0]
            shuffled = rng.permutation(idx)
            q_idx_list.append(int(shuffled[0]))
            g_idx_list.append(shuffled[1:])
            person_labels.append(person)
        person_labels_arr = np.asarray(person_labels)
        sim_matrices: list[np.ndarray] = []
        for name in model_order:
            emb = embeddings_by_model[name]
            queries = emb[q_idx_list]  # (n_persons, dim) — raw single-tooth query
            galleries = np.stack([aggregate_mean(emb[idx]) for idx in g_idx_list])
            sim_matrices.append(queries @ galleries.T)
        for s_name, combiner in strategies.items():
            combined = combiner(sim_matrices)
            m = rank_metrics(combined, person_labels_arr)
            m["n_persons"] = len(person_labels_arr)
            per_strategy[s_name].append(m)

    out: dict[str, dict] = {}
    for s_name, trials in per_strategy.items():
        if not trials:
            continue
        df = pd.DataFrame(trials)
        out[s_name] = {
            "n_query": 1,
            "method": f"single_query_{s_name}_gallery",
            "n_persons": int(df["n_persons"].iloc[0]),
            "n_trials": int(len(df)),
            "rank1_mean": float(df["rank1"].mean()),
            "rank1_std": float(df["rank1"].std()),
            "rank5_mean": float(df["rank5"].mean()),
            "rank5_std": float(df["rank5"].std()),
            "rank10_mean": float(df["rank10"].mean()),
            "rank10_std": float(df["rank10"].std()),
            "mAP_mean": float(df["mAP"].mean()),
            "mAP_std": float(df["mAP"].std()),
        }
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--val-split", default="val",
                        help="Split for grid-searching weights.")
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--n-query-list", nargs="+", type=int,
                        default=[1, 2, 4, 8, 16])
    parser.add_argument("--weight-search-n-query", type=int, default=8,
                        help="n_query at which to grid-search weights on val.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="identification/runs/ensemble_v1/analysis/person_retrieval")
    parser.add_argument("--manifest", default=None,
                        help="Override the embedder's training manifest. Use "
                             "identification/data/manifest_yolo.csv to evaluate the "
                             "ensemble on YOLO-extracted crops (matching the demo).")
    args = parser.parse_args()

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Ensemble of: {list(CHECKPOINTS.keys())}")

    # 1. Extract embeddings on val (for weight search) and test (for final metrics).
    print(f"\n--- Extracting on val for weight search ---")
    val_embs, val_labels = extract_all_models(args.val_split, device,
                                                manifest_override=args.manifest)
    print(f"\n--- Grid-searching weights on val (n_query={args.weight_search_n_query}) ---")
    weights = grid_search_weights(val_embs, val_labels,
                                    args.weight_search_n_query, args.n_trials, args.seed)
    print(f"Best val weights: {weights}")

    print(f"\n--- Extracting on {args.split} for final metrics ---")
    test_embs, test_labels = extract_all_models(args.split, device,
                                                  manifest_override=args.manifest)

    # 2. Symmetric sweep
    print("\n--- Sweep ---")
    sweep_results = sweep(test_embs, test_labels, args.n_query_list, args.n_trials,
                            args.seed, weights=weights)
    sweep_flat: list[dict] = []
    for s_name, rows in sweep_results.items():
        for r in rows:
            print(f"  [{s_name:8s}] n_query={r['n_query']:>2}  R1={r['rank1_mean']:.4f}  R5={r['rank5_mean']:.4f}  mAP={r['mAP_mean']:.4f}")
            sweep_flat.append(r)

    # 3. Forensic 1-vs-aggregated
    print("\n--- Forensic 1-tooth ---")
    forensic = forensic_one_tooth(test_embs, test_labels, args.n_trials, args.seed,
                                    weights=weights)
    for s_name, res in forensic.items():
        print(f"  [{s_name:8s}] R1={res['rank1_mean']:.4f}  R5={res['rank5_mean']:.4f}")

    # 4. Save
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "checkpoints": CHECKPOINTS,
        "split": args.split,
        "val_split": args.val_split,
        "n_trials": args.n_trials,
        "weights": weights,
        "sweep": sweep_flat,
        "forensic_1tooth": list(forensic.values()),
    }
    out_path = output_dir / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
