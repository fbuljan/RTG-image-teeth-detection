"""
Multi-tooth person retrieval evaluation.

Tests the practical forensic scenario:
- Gallery: each test person represented by an aggregated embedding from K teeth
- Query: N teeth from a different test person → aggregated → matched against gallery

Reports Rank-K and mAP as a function of:
- Number of query teeth (n_query)
- Aggregation method (mean, max, weighted)

Usage:
    python -m identification.evaluation.evaluate_person_retrieval --checkpoint path/to/best.pt
    python -m identification.evaluation.evaluate_person_retrieval --checkpoint path/to/best.pt --n-trials 10
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from identification.models.person_aggregation import (
    AGGREGATORS,
    aggregate_by_person,
)


def evaluate_query_vs_gallery(query_emb: np.ndarray, query_label,
                                gallery_emb: np.ndarray, gallery_labels: np.ndarray):
    """For one query embedding, compute Rank-1, Rank-5, Rank-10, AP."""
    # Cosine similarity (both already L2-normalized at extraction)
    sims = gallery_emb @ query_emb  # (n_gallery,)
    ranked = np.argsort(-sims)
    ranked_labels = gallery_labels[ranked]
    matches = ranked_labels == query_label

    rank1 = float(matches[0])
    rank5 = float(matches[:5].any())
    rank10 = float(matches[:10].any())

    positions = np.where(matches)[0]
    if len(positions) == 0:
        ap = 0.0
    else:
        precisions = np.arange(1, len(positions) + 1) / (positions + 1)
        ap = float(precisions.mean())

    return rank1, rank5, rank10, ap


def evaluate_multi_tooth(embeddings, labels, n_query, n_trials, method, rng):
    """
    Evaluate the n-tooth-query / aggregated-gallery setup.

    Protocol per trial:
        1. For each person, randomly select n_query teeth as query, rest as gallery
        2. Aggregate gallery teeth into one person profile per person
        3. Aggregate query teeth into one query embedding per person
        4. Each query is matched against all gallery profiles
        5. Compute Rank-K and AP per query

    Returns dict with mean and std of metrics across trials.
    """
    aggregator = AGGREGATORS[method]
    unique_persons = np.unique(labels)

    # Filter to persons with > n_query teeth (need at least 1 in gallery)
    eligible = [p for p in unique_persons if (labels == p).sum() > n_query]

    if len(eligible) < 5:
        return None  # not enough persons for retrieval

    trial_metrics = []
    for trial in range(n_trials):
        query_embs = []  # (n_persons, dim) — aggregated queries
        gallery_embs = []  # (n_persons, dim) — aggregated gallery
        person_labels = []

        for person in eligible:
            idx = np.where(labels == person)[0]
            shuffled = rng.permutation(idx)
            q_idx = shuffled[:n_query]
            g_idx = shuffled[n_query:]

            query_embs.append(aggregator(embeddings[q_idx]))
            gallery_embs.append(aggregator(embeddings[g_idx]))
            person_labels.append(person)

        query_embs = np.stack(query_embs)
        gallery_embs = np.stack(gallery_embs)
        person_labels = np.array(person_labels)

        # Compute pair-wise similarities. Each person has 1 query + 1 gallery profile,
        # built from disjoint teeth, so gallery[i] (profile of person i) IS the correct
        # match for query[i] — the diagonal must NOT be excluded.
        sim = query_embs @ gallery_embs.T  # (n, n)

        ranked = np.argsort(-sim, axis=1)
        ranked_labels = person_labels[ranked]
        query_labels_col = person_labels[:, None]
        matches = ranked_labels == query_labels_col

        rank1 = float(matches[:, 0].mean())
        rank5 = float(matches[:, :5].any(axis=1).mean())
        rank10 = float(matches[:, :10].any(axis=1).mean())

        # mAP per query (only one positive per query in gallery)
        first_pos = np.argmax(matches, axis=1)
        # If no match, argmax returns 0 but matches[:, 0] is False
        valid = matches.any(axis=1)
        aps = np.where(valid, 1.0 / (first_pos + 1), 0.0)
        mAP = float(aps.mean())

        trial_metrics.append({
            "rank1": rank1,
            "rank5": rank5,
            "rank10": rank10,
            "mAP": mAP,
            "n_persons": len(eligible),
        })

    df = pd.DataFrame(trial_metrics)
    return {
        "n_query": n_query,
        "method": method,
        "n_persons": int(df["n_persons"].iloc[0]),
        "n_trials": n_trials,
        "rank1_mean": float(df["rank1"].mean()),
        "rank1_std": float(df["rank1"].std()),
        "rank5_mean": float(df["rank5"].mean()),
        "rank5_std": float(df["rank5"].std()),
        "rank10_mean": float(df["rank10"].mean()),
        "rank10_std": float(df["rank10"].std()),
        "mAP_mean": float(df["mAP"].mean()),
        "mAP_std": float(df["mAP"].std()),
    }


def evaluate_single_tooth_vs_aggregated_gallery(embeddings, labels, n_trials, method, rng):
    """
    Realistic forensic scenario: query is ONE tooth, gallery is aggregated from many.

    Protocol per trial:
        1. For each person, randomly hold out 1 tooth as query, rest as gallery
        2. Aggregate gallery teeth into one profile per person
        3. Each query (1 tooth) is matched against all gallery profiles

    This is the asymmetric scenario the user wants for the thesis.
    """
    aggregator = AGGREGATORS[method]
    unique_persons = np.unique(labels)
    eligible = [p for p in unique_persons if (labels == p).sum() >= 2]

    if len(eligible) < 5:
        return None

    trial_metrics = []
    for trial in range(n_trials):
        query_embs = []
        gallery_embs = []
        person_labels = []

        for person in eligible:
            idx = np.where(labels == person)[0]
            shuffled = rng.permutation(idx)
            q_idx = shuffled[0]
            g_idx = shuffled[1:]

            query_embs.append(embeddings[q_idx])
            gallery_embs.append(aggregator(embeddings[g_idx]))
            person_labels.append(person)

        query_embs = np.stack(query_embs)
        gallery_embs = np.stack(gallery_embs)
        person_labels = np.array(person_labels)

        # Same construction as above: gallery[i] is the unique correct match for query[i].
        sim = query_embs @ gallery_embs.T

        ranked = np.argsort(-sim, axis=1)
        ranked_labels = person_labels[ranked]
        query_labels_col = person_labels[:, None]
        matches = ranked_labels == query_labels_col

        rank1 = float(matches[:, 0].mean())
        rank5 = float(matches[:, :5].any(axis=1).mean())
        rank10 = float(matches[:, :10].any(axis=1).mean())

        first_pos = np.argmax(matches, axis=1)
        valid = matches.any(axis=1)
        aps = np.where(valid, 1.0 / (first_pos + 1), 0.0)
        mAP = float(aps.mean())

        trial_metrics.append({"rank1": rank1, "rank5": rank5, "rank10": rank10, "mAP": mAP})

    df = pd.DataFrame(trial_metrics)
    return {
        "n_query": 1,
        "method": f"single_query_{method}_gallery",
        "n_persons": len(eligible),
        "n_trials": n_trials,
        "rank1_mean": float(df["rank1"].mean()),
        "rank1_std": float(df["rank1"].std()),
        "rank5_mean": float(df["rank5"].mean()),
        "rank5_std": float(df["rank5"].std()),
        "rank10_mean": float(df["rank10"].mean()),
        "rank10_std": float(df["rank10"].std()),
        "mAP_mean": float(df["mAP"].mean()),
        "mAP_std": float(df["mAP"].std()),
    }


def plot_n_query_curve(results_df, output_path):
    """Plot Rank-1 / Rank-5 / mAP as a function of n_query teeth."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for method in results_df["method"].unique():
        sub = results_df[results_df["method"] == method].sort_values("n_query")
        for metric, color in [("rank1_mean", "tab:blue"), ("rank5_mean", "tab:orange"), ("mAP_mean", "tab:green")]:
            std_col = metric.replace("_mean", "_std")
            label = f"{method} {metric.replace('_mean', '')}"
            ax.errorbar(sub["n_query"], sub[metric], yerr=sub[std_col],
                        marker="o", capsize=3, label=label)

    ax.set_xlabel("Number of query teeth")
    ax.set_ylabel("Score")
    ax.set_title("Multi-tooth retrieval: metrics vs n_query")
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--methods", nargs="+", default=["mean", "max"],
                        choices=list(AGGREGATORS.keys()))
    parser.add_argument("--n-query-list", nargs="+", type=int,
                        default=[1, 2, 4, 8, 16])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, label_map, ckpt = load_checkpoint(args.checkpoint, device)

    uses_metadata = isinstance(model, ToothEmbeddingModelWithMetadata)
    dataset = build_eval_dataset(cfg, args.split, label_map, ckpt, uses_metadata)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    print(f"Evaluating {args.split}: {len(dataset)} samples, {len(label_map)} persons")

    print("Extracting embeddings...")
    embeddings, labels = extract_embeddings(model, loader, device)

    rng = np.random.RandomState(args.seed)

    # Sweep n_query × method
    all_results = []
    print("\n=== Sweep: n_query × aggregation method ===")
    for method in args.methods:
        for n_q in args.n_query_list:
            print(f"  n_query={n_q}, method={method}", end=" ... ")
            res = evaluate_multi_tooth(embeddings, labels, n_q, args.n_trials, method, rng)
            if res is None:
                print("skipped (insufficient persons)")
                continue
            print(f"R1={res['rank1_mean']:.4f}±{res['rank1_std']:.4f}, "
                  f"R5={res['rank5_mean']:.4f}, mAP={res['mAP_mean']:.4f}")
            all_results.append(res)

    # Single-query / aggregated-gallery (forensic scenario)
    print("\n=== Forensic scenario: 1 query tooth vs aggregated gallery ===")
    forensic_results = []
    for method in args.methods:
        res = evaluate_single_tooth_vs_aggregated_gallery(embeddings, labels, args.n_trials, method, rng)
        if res is None:
            continue
        print(f"  query=1, gallery_method={method}: R1={res['rank1_mean']:.4f}±{res['rank1_std']:.4f}, "
              f"R5={res['rank5_mean']:.4f}, mAP={res['mAP_mean']:.4f}")
        forensic_results.append(res)

    # Output
    if args.output_dir is None:
        output_dir = Path(args.checkpoint).parent / "analysis" / "person_retrieval"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(all_results + forensic_results)
    df.to_csv(output_dir / "person_retrieval.csv", index=False)

    # Plot the symmetric sweep
    if all_results:
        sweep_df = pd.DataFrame(all_results)
        plot_n_query_curve(sweep_df, output_dir / "metrics_vs_nquery.png")

    # Save metrics.json for headline numbers
    summary = {
        "checkpoint": args.checkpoint,
        "n_trials": args.n_trials,
        "sweep": all_results,
        "forensic_1tooth": forensic_results,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
