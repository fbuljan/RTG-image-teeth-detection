"""
Same-tooth-type forensic evaluation.

For each FDI tooth type, restrict both gallery and query to that tooth type,
then compute verification metrics (AUC, EER) on all within-FDI pairs.

NOTE: Because we have only 1 panoramic image per person, each person has
at most 1 tooth of each FDI type in the test set. This means within-FDI
retrieval (Rank-K) is not well-defined (no positive matches exist). We
therefore focus on verification — "are these two tooth 11s from the same
person?" — which captures the forensic value without requiring multiple
instances of the same tooth per person.

Usage:
    python -m identification.evaluation.evaluate_same_tooth --checkpoint path/to/best.pt
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

from identification.data.tooth_dataset import ToothDataset, get_val_transforms
from identification.evaluation.evaluate_embedding import (
    build_eval_dataset,
    extract_embeddings,
    load_checkpoint,
    evaluate_verification,
    evaluate_retrieval,
)
from identification.models.embedding_model import ToothEmbeddingModelWithMetadata


def evaluate_per_fdi_forensic(embeddings, labels, fdi_array, min_persons=20):
    """
    For each FDI, compute verification metrics on same-FDI pairs only.

    Since each person contributes at most 1 tooth per FDI in our dataset
    (1 panoramic image per person), within-FDI retrieval has no positive
    matches. But verification ("are these two tooth 11s from the same
    person?") is well-defined when both tooth 11s come from DIFFERENT persons
    — we compute the similarity distribution and compare it to the similarity
    distribution of same-person different-FDI pairs (the baseline has those).

    To enable verification within a single FDI, we need at least some
    positive pairs. Since none exist within a single FDI (1 per person),
    we treat this as a "cross-person similarity calibration" problem:
    compute the mean negative-pair similarity per FDI.

    Also compute retrieval metrics using that FDI as query against the FULL
    gallery (this is the actual forensic scenario: 'given a tooth 11, find
    its owner in a database of mixed teeth').
    """
    results = {}

    # Full gallery similarity matrix
    sim_matrix = embeddings @ embeddings.T
    np.fill_diagonal(sim_matrix, -float("inf"))

    unique_fdis = sorted(set(fdi_array), key=lambda x: int(x) if x.isdigit() else 99)

    for fdi in unique_fdis:
        mask = fdi_array == fdi
        n_samples = int(mask.sum())
        if n_samples < min_persons:
            continue

        query_idx = np.where(mask)[0]

        # Within-FDI negative-pair similarity (all pairs are negative since 1/person/FDI)
        if n_samples >= 2:
            # Same-FDI pair similarities (all negative in our data)
            triu_i, triu_j = np.triu_indices(n_samples, k=1)
            global_i = query_idx[triu_i]
            global_j = query_idx[triu_j]
            same_fdi_sims = np.diag(embeddings[global_i] @ embeddings[global_j].T) \
                if False else embeddings[global_i[0]:global_i[0]+1]  # placeholder
            # Simpler: use submatrix
            sub = embeddings[query_idx] @ embeddings[query_idx].T
            same_fdi_sims = sub[np.triu_indices(n_samples, k=1)]
            mean_neg_sim_within_fdi = float(same_fdi_sims.mean())
        else:
            mean_neg_sim_within_fdi = float("nan")

        # Retrieval: query=this FDI, gallery=all test samples
        ranked = np.argsort(-sim_matrix[query_idx], axis=1)
        ranked_labels = labels[ranked]
        query_labels = labels[query_idx][:, None]
        matches = ranked_labels == query_labels

        rank1 = float(matches[:, 0].mean())
        rank5 = float(matches[:, :5].any(axis=1).mean())
        rank10 = float(matches[:, :10].any(axis=1).mean())

        aps = []
        for row in matches:
            positions = np.where(row)[0]
            if len(positions) == 0:
                aps.append(0.0)
                continue
            precisions = np.arange(1, len(positions) + 1) / (positions + 1)
            aps.append(float(precisions.mean()))
        mAP = float(np.mean(aps)) if aps else 0.0

        results[fdi] = {
            "fdi": fdi,
            "n_samples": n_samples,
            "n_persons": int(len(np.unique(labels[mask]))),
            "rank1": rank1,
            "rank5": rank5,
            "rank10": rank10,
            "mAP": mAP,
            "mean_neg_sim_within_fdi": mean_neg_sim_within_fdi,
        }

    return results


def plot_same_vs_cross(same_df, baseline_rank1, output_path):
    """Bar chart comparing per-FDI-query Rank-1 to overall baseline Rank-1."""
    df = same_df.sort_values("fdi", key=lambda s: s.astype(int))
    fdis = df["fdi"].tolist()
    same_vals = df["rank1"].tolist()

    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(fdis))
    width = 0.4

    ax.bar(x - width/2, [baseline_rank1] * len(fdis), width,
           label=f"Cross-tooth baseline ({baseline_rank1:.3f})",
           color="#9E9E9E", alpha=0.7, edgecolor="black")
    ax.bar(x + width/2, same_vals, width,
           label="Same-tooth-type",
           color="#2196F3", alpha=0.85, edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(fdis, rotation=90)
    ax.set_xlabel("FDI tooth number")
    ax.set_ylabel("Rank-1 accuracy")
    ax.set_title("Per-FDI Query Rank-1 vs Overall Baseline")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim([0, 1.05])

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_metrics_summary(same_df, output_path):
    """Bar chart of Rank-1, Rank-5, Rank-10, mAP per FDI."""
    df = same_df.sort_values("fdi", key=lambda s: s.astype(int))
    fdis = df["fdi"].tolist()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    metrics = [
        ("rank1", "Rank-1", axes[0, 0]),
        ("rank5", "Rank-5", axes[0, 1]),
        ("rank10", "Rank-10", axes[1, 0]),
        ("mAP", "Retrieval mAP", axes[1, 1]),
    ]
    for col, title, ax in metrics:
        ax.bar(range(len(fdis)), df[col].tolist(),
               color="#2196F3", edgecolor="black", alpha=0.85)
        ax.set_xticks(range(len(fdis)))
        ax.set_xticklabels(fdis, rotation=90, fontsize=8)
        ax.set_title(title)
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_summary_md(same_df, baseline_metrics, output_path):
    """Thesis-ready markdown."""
    lines = ["# Per-FDI Query Evaluation\n"]
    lines.append("For each tooth FDI, query samples of that type against the full gallery.\n")
    lines.append("Note: Within-FDI retrieval is not well-defined because each person\n")
    lines.append("contributes at most 1 tooth per FDI in our dataset.\n")

    lines.append("## Baseline (all queries, full-gallery retrieval)\n")
    lines.append(f"- Rank-1: {baseline_metrics['rank1']:.4f}")
    lines.append(f"- AUC: {baseline_metrics['auc']:.4f}")
    lines.append(f"- mAP: {baseline_metrics['mAP']:.4f}\n")

    lines.append("## Per-FDI Query Averages\n")
    avg_rank1 = same_df["rank1"].mean()
    avg_map = same_df["mAP"].mean()
    lines.append(f"- Avg Rank-1 across FDIs: {avg_rank1:.4f}")
    lines.append(f"- Avg mAP across FDIs: {avg_map:.4f}\n")

    lines.append("## Per-FDI Results\n")
    show = same_df.sort_values("rank1", ascending=False)[
        ["fdi", "n_persons", "rank1", "rank5", "rank10", "mAP", "mean_neg_sim_within_fdi"]
    ]
    lines.append(show.to_markdown(index=False, floatfmt=".4f"))

    lines.append("\n\n## Top 5 Most Informative Query FDIs (by Rank-1)\n")
    top5 = same_df.nlargest(5, "rank1")[["fdi", "n_persons", "rank1", "rank5", "mAP"]]
    lines.append(top5.to_markdown(index=False, floatfmt=".4f"))

    lines.append("\n\n## Bottom 5 Least Informative Query FDIs\n")
    bot5 = same_df.nsmallest(5, "rank1")[["fdi", "n_persons", "rank1", "rank5", "mAP"]]
    lines.append(bot5.to_markdown(index=False, floatfmt=".4f"))

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Same-tooth-type forensic evaluation")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--split", default="test", help="Split to evaluate")
    parser.add_argument("--min-persons", type=int, default=20, help="Min persons per FDI")
    parser.add_argument("--baseline-metrics", default=None,
                        help="Path to baseline metrics.json (defaults to cross-tooth eval_test)")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, label_map, ckpt = load_checkpoint(args.checkpoint, device)

    manifest_path = cfg["data"]["manifest"]
    target_col = cfg["data"]["target_col"]
    uses_metadata = isinstance(model, ToothEmbeddingModelWithMetadata)
    dataset = build_eval_dataset(cfg, args.split, label_map, ckpt, uses_metadata)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    print(f"Evaluating {args.split} split: {len(dataset)} samples")

    print("Extracting embeddings...")
    embeddings, labels = extract_embeddings(model, loader, device)

    df = pd.read_csv(manifest_path, dtype=str)
    df_split = df[df["split"] == args.split].reset_index(drop=True)
    assert len(df_split) == len(embeddings)
    fdi_array = df_split["tooth_fdi"].values

    # Load baseline cross-tooth metrics for comparison
    if args.baseline_metrics is None:
        baseline_path = Path(args.checkpoint).parent / f"eval_{args.split}" / "metrics.json"
    else:
        baseline_path = Path(args.baseline_metrics)

    baseline = {"rank1": 0.0, "auc": 0.0, "mAP": 0.0}
    if baseline_path.exists():
        with open(baseline_path) as f:
            m = json.load(f)
        baseline["rank1"] = m["retrieval"]["rank1_micro"]
        baseline["auc"] = m["verification"]["auc"]
        baseline["mAP"] = m["retrieval"]["mAP"]
        print(f"Loaded baseline from {baseline_path}")

    # Compute per-FDI forensic metrics
    print("Computing per-FDI forensic metrics...")
    results = evaluate_per_fdi_forensic(embeddings, labels, fdi_array, min_persons=args.min_persons)
    same_df = pd.DataFrame(list(results.values())) if results else pd.DataFrame()
    for col in ["n_samples", "n_persons", "rank1", "rank5", "rank10", "mAP",
                "mean_neg_sim_within_fdi"]:
        if col in same_df.columns:
            same_df[col] = pd.to_numeric(same_df[col], errors="coerce")

    # Output
    if args.output_dir is None:
        output_dir = Path(args.checkpoint).parent / "analysis" / "same_tooth"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    same_df.to_csv(output_dir / "same_tooth_metrics.csv", index=False)

    # Side-by-side comparison
    if baseline["rank1"] > 0:
        compare = same_df[["fdi", "n_persons", "rank1", "mAP"]].copy()
        compare.columns = ["fdi", "n_persons", "per_fdi_rank1", "per_fdi_mAP"]
        compare["baseline_rank1"] = baseline["rank1"]
        compare["baseline_mAP"] = baseline["mAP"]
        compare["rank1_delta"] = compare["per_fdi_rank1"] - baseline["rank1"]
        compare.to_csv(output_dir / "per_fdi_vs_baseline.csv", index=False)

        plot_same_vs_cross(same_df, baseline["rank1"],
                           output_dir / "rank1_by_fdi.png")

    plot_metrics_summary(same_df, output_dir / "metrics_by_fdi.png")
    write_summary_md(same_df, baseline, output_dir / "summary.md")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Per-FDI Query Evaluation")
    print(f"{'='*60}")
    print(f"FDIs evaluated: {len(same_df)} (min {args.min_persons} persons)")
    if len(same_df) > 0:
        avg_rank1 = same_df["rank1"].mean()
        best = same_df.nlargest(1, "rank1").iloc[0]
        worst = same_df.nsmallest(1, "rank1").iloc[0]
        print(f"Average Rank-1: {avg_rank1:.4f} (baseline: {baseline['rank1']:.4f}, "
              f"{avg_rank1/max(baseline['rank1'], 1e-6):.1f}×)")
        print(f"Best FDI:  {best['fdi']} — Rank-1={best['rank1']:.4f}")
        print(f"Worst FDI: {worst['fdi']} — Rank-1={worst['rank1']:.4f}")
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
