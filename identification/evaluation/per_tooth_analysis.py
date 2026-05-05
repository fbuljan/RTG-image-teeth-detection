"""
Per-tooth-type performance analysis.

Breaks down verification/retrieval metrics by FDI tooth type, quadrant, jaw,
and anatomical category (incisor/canine/premolar/molar).

Unlike evaluate_same_tooth.py, this evaluates the model on the FULL test set
once, then partitions results by FDI for analysis — it does NOT restrict
gallery/query pairs to the same tooth type (that's 4.4).

Usage:
    python -m identification.evaluation.per_tooth_analysis --checkpoint path/to/best.pt
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


# FDI → anatomical category
FDI_CATEGORY = {}
for fdi in [11, 12, 21, 22, 31, 32, 41, 42]:
    FDI_CATEGORY[str(fdi)] = "incisor"
for fdi in [13, 23, 33, 43]:
    FDI_CATEGORY[str(fdi)] = "canine"
for fdi in [14, 15, 24, 25, 34, 35, 44, 45]:
    FDI_CATEGORY[str(fdi)] = "premolar"
for fdi in [16, 17, 18, 26, 27, 28, 36, 37, 38, 46, 47, 48]:
    FDI_CATEGORY[str(fdi)] = "molar"
# Deciduous (5x-8x) — categorize similarly but mark as deciduous
for quad in [5, 6, 7, 8]:
    for pos in range(1, 6):
        fdi = quad * 10 + pos
        if pos <= 2:
            FDI_CATEGORY[str(fdi)] = "deciduous_incisor"
        elif pos == 3:
            FDI_CATEGORY[str(fdi)] = "deciduous_canine"
        else:
            FDI_CATEGORY[str(fdi)] = "deciduous_molar"


def compute_metrics_for_subset(embeddings, labels, min_persons=5):
    """Compute verification + retrieval metrics on a subset. Returns dict or None."""
    unique_persons = np.unique(labels)
    if len(unique_persons) < min_persons:
        return None
    if len(labels) < min_persons * 2:
        return None

    ver = evaluate_verification(embeddings, labels)
    ret = evaluate_retrieval(embeddings, labels)

    return {
        "n_samples": len(labels),
        "n_persons": int(len(unique_persons)),
        "auc": ver["auc"],
        "eer": ver["eer"],
        "rank1_micro": ret["rank1_micro"],
        "rank1_macro": ret["rank1_macro"],
        "rank5": ret["rank5"],
        "rank10": ret["rank10"],
        "mAP": ret["mAP"],
    }


def compute_per_fdi_as_query(embeddings, labels, fdi_array, min_samples=30):
    """
    For each FDI, use that FDI's samples as queries against the FULL gallery
    (all test samples). Returns per-FDI Rank-K accuracy.

    This measures: "if I have a tooth of type T, how well can I find the owner
    in the full database?"
    """
    results = {}
    # Full-gallery similarity matrix (needed once)
    sim_matrix = embeddings @ embeddings.T
    np.fill_diagonal(sim_matrix, -float("inf"))

    unique_fdis = sorted(set(fdi_array), key=lambda x: int(x) if x.isdigit() else 99)
    for fdi in unique_fdis:
        mask = fdi_array == fdi
        n_samples = int(mask.sum())
        if n_samples < min_samples:
            continue

        query_idx = np.where(mask)[0]
        # For each query, find rank of its correct matches in the full gallery
        ranked = np.argsort(-sim_matrix[query_idx], axis=1)
        ranked_labels = labels[ranked]
        query_labels = labels[query_idx][:, None]
        matches = ranked_labels == query_labels

        rank1 = float(matches[:, 0].mean())
        rank5 = float(matches[:, :5].any(axis=1).mean())
        rank10 = float(matches[:, :10].any(axis=1).mean())

        # mAP per query
        aps = []
        for row in matches:
            positions = np.where(row)[0]
            if len(positions) == 0:
                aps.append(0.0)
                continue
            precisions = np.arange(1, len(positions) + 1) / (positions + 1)
            aps.append(float(precisions.mean()))
        mAP = float(np.mean(aps)) if aps else 0.0

        # Average positive similarity (for verification-like signal)
        # Positive pairs for this FDI = same person (across all FDIs)
        same_person_pairs = []
        for i, q_idx in enumerate(query_idx):
            matches_to_same = ranked[i][matches[i]]
            if len(matches_to_same) > 0:
                same_person_pairs.append(sim_matrix[q_idx, matches_to_same].mean())
        avg_pos_sim = float(np.mean(same_person_pairs)) if same_person_pairs else float("nan")

        results[fdi] = {
            "fdi": fdi,
            "category": FDI_CATEGORY.get(fdi, "unknown"),
            "n_samples": n_samples,
            "rank1": rank1,
            "rank5": rank5,
            "rank10": rank10,
            "mAP": mAP,
            "avg_pos_sim": avg_pos_sim,
        }

    return results


def compute_per_group(embeddings, labels, group_array, min_samples=50):
    """Compute metrics per group (e.g., per quadrant, per jaw)."""
    results = {}
    unique_groups = sorted(set(group_array))

    for group in unique_groups:
        mask = group_array == group
        if mask.sum() < min_samples:
            continue
        subset_emb = embeddings[mask]
        subset_lbl = labels[mask]
        metrics = compute_metrics_for_subset(subset_emb, subset_lbl)
        if metrics is None:
            continue
        metrics["group"] = str(group)
        results[str(group)] = metrics

    return results


def plot_metric_by_fdi(per_fdi_df, metric, title, output_path):
    """Bar chart of a metric across FDIs, colored by category."""
    df = per_fdi_df.sort_values("fdi", key=lambda s: s.astype(int))
    categories = df["category"].tolist()

    # Color per category
    category_colors = {
        "incisor": "#2196F3",
        "canine": "#4CAF50",
        "premolar": "#FF9800",
        "molar": "#F44336",
        "deciduous_incisor": "#90CAF9",
        "deciduous_canine": "#A5D6A7",
        "deciduous_molar": "#FFCC80",
        "unknown": "#9E9E9E",
    }
    colors = [category_colors.get(c, "#9E9E9E") for c in categories]

    fig, ax = plt.subplots(figsize=(16, 6))
    fdis = df["fdi"].tolist()
    values = df[metric].tolist()

    ax.bar(range(len(fdis)), values, color=colors, edgecolor="black", alpha=0.85)
    ax.set_xticks(range(len(fdis)))
    ax.set_xticklabels(fdis, rotation=90)
    ax.set_xlabel("FDI tooth number")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=col, label=cat)
        for cat, col in category_colors.items()
        if cat in set(categories)
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_summary_md(per_fdi_df, per_cat_df, per_quadrant_df, per_jaw_df, output_path):
    """Thesis-ready markdown summary."""
    lines = ["# Per-Tooth-Type Analysis — Summary\n"]

    # Top/bottom 5 FDIs by Rank-1
    lines.append("## Top 5 Query FDIs by Rank-1 (full-gallery retrieval)\n")
    top5 = per_fdi_df.nlargest(5, "rank1")[["fdi", "category", "n_samples", "rank1", "rank5", "mAP"]]
    lines.append(top5.to_markdown(index=False, floatfmt=".4f"))
    lines.append("\n\n## Bottom 5 Query FDIs by Rank-1\n")
    bot5 = per_fdi_df.nsmallest(5, "rank1")[["fdi", "category", "n_samples", "rank1", "rank5", "mAP"]]
    lines.append(bot5.to_markdown(index=False, floatfmt=".4f"))

    # By category
    lines.append("\n\n## By Anatomical Category\n")
    lines.append(per_cat_df[["group", "n_samples", "n_persons", "rank1_micro", "auc", "mAP"]].to_markdown(index=False, floatfmt=".4f"))

    # By quadrant
    lines.append("\n\n## By Quadrant\n")
    lines.append(per_quadrant_df[["group", "n_samples", "n_persons", "rank1_micro", "auc", "mAP"]].to_markdown(index=False, floatfmt=".4f"))

    # By jaw
    lines.append("\n\n## By Jaw\n")
    lines.append(per_jaw_df[["group", "n_samples", "n_persons", "rank1_micro", "auc", "mAP"]].to_markdown(index=False, floatfmt=".4f"))

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Per-tooth-type performance analysis")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--split", default="test", help="Split to analyze")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--min-samples", type=int, default=30, help="Min samples per FDI to include")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, label_map, ckpt = load_checkpoint(args.checkpoint, device)

    manifest_path = cfg["data"]["manifest"]
    target_col = cfg["data"]["target_col"]
    uses_metadata = isinstance(model, ToothEmbeddingModelWithMetadata)
    dataset = build_eval_dataset(cfg, args.split, label_map, ckpt, uses_metadata)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    print(f"Analyzing {args.split} split: {len(dataset)} samples")

    # Extract embeddings
    print("Extracting embeddings...")
    embeddings, labels = extract_embeddings(model, loader, device)

    # Align with manifest metadata
    df = pd.read_csv(manifest_path, dtype=str)
    df_split = df[df["split"] == args.split].reset_index(drop=True)
    assert len(df_split) == len(embeddings), f"Manifest/embeddings mismatch: {len(df_split)} vs {len(embeddings)}"

    fdi_array = df_split["tooth_fdi"].values
    quadrant_array = df_split["quadrant"].values
    jaw_array = df_split["jaw"].values
    category_array = np.array([FDI_CATEGORY.get(f, "unknown") for f in fdi_array])

    def to_df(results_dict):
        if not results_dict:
            return pd.DataFrame()
        df = pd.DataFrame(list(results_dict.values()))
        # Coerce numeric columns (they may be object dtype after the dict loop)
        for col in ["n_samples", "n_persons", "auc", "eer", "rank1", "rank1_micro",
                    "rank1_macro", "rank5", "rank10", "mAP", "avg_pos_sim"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    print("Computing per-FDI metrics (as query, against full gallery)...")
    per_fdi_df = to_df(compute_per_fdi_as_query(embeddings, labels, fdi_array, min_samples=args.min_samples))

    print("Computing per-category metrics...")
    per_cat_df = to_df(compute_per_group(embeddings, labels, category_array, min_samples=50))

    print("Computing per-quadrant metrics...")
    per_quad_df = to_df(compute_per_group(embeddings, labels, quadrant_array, min_samples=50))

    print("Computing per-jaw metrics...")
    per_jaw_df = to_df(compute_per_group(embeddings, labels, jaw_array, min_samples=50))

    # Output directory
    if args.output_dir is None:
        output_dir = Path(args.checkpoint).parent / "analysis" / "per_tooth"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save CSVs
    per_fdi_df.to_csv(output_dir / "per_fdi_metrics.csv", index=False)
    per_cat_df.to_csv(output_dir / "per_category_metrics.csv", index=False)
    per_quad_df.to_csv(output_dir / "per_quadrant_metrics.csv", index=False)
    per_jaw_df.to_csv(output_dir / "per_jaw_metrics.csv", index=False)

    # Plots
    if len(per_fdi_df) > 0:
        plot_metric_by_fdi(per_fdi_df, "rank1",
                           "Rank-1 by query FDI (full-gallery retrieval)",
                           output_dir / "rank1_by_fdi.png")
        plot_metric_by_fdi(per_fdi_df, "rank5",
                           "Rank-5 by query FDI (full-gallery retrieval)",
                           output_dir / "rank5_by_fdi.png")
        plot_metric_by_fdi(per_fdi_df, "mAP",
                           "mAP by query FDI (full-gallery retrieval)",
                           output_dir / "map_by_fdi.png")

    # Summary markdown
    write_summary_md(per_fdi_df, per_cat_df, per_quad_df, per_jaw_df, output_dir / "summary.md")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Per-Tooth Analysis — {args.split} split")
    print(f"{'='*60}")
    if len(per_fdi_df) > 0:
        print(f"FDIs analyzed: {len(per_fdi_df)} (min {args.min_samples} samples)")
        best = per_fdi_df.nlargest(1, "rank1").iloc[0]
        worst = per_fdi_df.nsmallest(1, "rank1").iloc[0]
        print(f"  Best FDI:  {best['fdi']} ({best['category']}) — Rank-1={best['rank1']:.4f}")
        print(f"  Worst FDI: {worst['fdi']} ({worst['category']}) — Rank-1={worst['rank1']:.4f}")
    if len(per_cat_df) > 0:
        print(f"\nBy category:")
        for _, row in per_cat_df.iterrows():
            print(f"  {row['group']:20s}: Rank-1={row['rank1_micro']:.4f}, AUC={row['auc']:.4f} (n={row['n_samples']})")
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
