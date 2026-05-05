"""
Subgroup analysis by demographics and clinical attributes.

Evaluates embedding quality on subgroups defined by sex, age, eruption status,
and root completion status. Uses bootstrap to estimate 95% confidence intervals
where sample sizes are marginal.

Usage:
    python -m identification.evaluation.subgroup_analysis --checkpoint path/to/best.pt
"""

import argparse
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
    evaluate_retrieval,
    evaluate_verification,
)
from identification.models.embedding_model import ToothEmbeddingModelWithMetadata


def age_bucket(age_str):
    """Group ages into 3 buckets."""
    try:
        age = float(age_str)
    except (ValueError, TypeError):
        return None
    if age <= 10:
        return "6-10"
    elif age <= 13:
        return "11-13"
    else:
        return "14-18"


def compute_subgroup_metrics(embeddings, labels, min_samples=100, min_persons=10):
    """Compute verification + retrieval for a subgroup."""
    if len(labels) < min_samples:
        return None
    unique_persons = np.unique(labels)
    if len(unique_persons) < min_persons:
        return None

    ver = evaluate_verification(embeddings, labels)
    ret = evaluate_retrieval(embeddings, labels)
    return {
        "n_samples": len(labels),
        "n_persons": int(len(unique_persons)),
        "auc": ver["auc"],
        "eer": ver["eer"],
        "rank1_micro": ret["rank1_micro"],
        "rank5": ret["rank5"],
        "rank10": ret["rank10"],
        "mAP": ret["mAP"],
    }


def bootstrap_rank1_ci(embeddings, labels, n_boot=50, alpha=0.05, seed=42):
    """Bootstrap confidence interval for Rank-1."""
    if len(labels) < 20:
        return None, None
    rng = np.random.RandomState(seed)
    n = len(labels)
    boot_vals = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        try:
            ret = evaluate_retrieval(embeddings[idx], labels[idx])
            boot_vals.append(ret["rank1_micro"])
        except Exception:
            continue
    if len(boot_vals) < 10:
        return None, None
    low = np.percentile(boot_vals, 100 * alpha / 2)
    high = np.percentile(boot_vals, 100 * (1 - alpha / 2))
    return float(low), float(high)


def analyze_subgroup(embeddings, labels, group_array, group_name,
                     min_samples=100, min_persons=10, bootstrap=False):
    """Analyze one subgroup variable. Returns DataFrame rows."""
    results = []
    # Filter out None/NaN first, then sort (cannot sort mixed None + str)
    valid = [g for g in set(group_array) if g is not None and str(g) != "nan" and str(g) != ""]
    unique_groups = sorted(valid, key=str)

    for group in unique_groups:
        mask = np.array([g == group for g in group_array])
        if mask.sum() == 0:
            continue
        sub_emb = embeddings[mask]
        sub_lbl = labels[mask]
        metrics = compute_subgroup_metrics(sub_emb, sub_lbl, min_samples, min_persons)
        if metrics is None:
            print(f"  Skipping {group_name}={group}: insufficient samples "
                  f"({mask.sum()} samples, {len(np.unique(sub_lbl))} persons)")
            continue

        row = {"subgroup_type": group_name, "group": str(group)}
        row.update(metrics)

        if bootstrap:
            ci_low, ci_high = bootstrap_rank1_ci(sub_emb, sub_lbl)
            row["rank1_ci_low"] = ci_low
            row["rank1_ci_high"] = ci_high

        results.append(row)

    return pd.DataFrame(results)


def plot_subgroup_bar(df, metric, group_col, title, output_path):
    """Bar chart for a subgroup with optional CI error bars."""
    if len(df) == 0:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    groups = df[group_col].tolist()
    values = df[metric].tolist()

    if "rank1_ci_low" in df.columns and df["rank1_ci_low"].notna().any() and metric == "rank1_micro":
        lower = [v - l if not pd.isna(l) else 0 for v, l in zip(values, df["rank1_ci_low"])]
        upper = [h - v if not pd.isna(h) else 0 for v, h in zip(values, df["rank1_ci_high"])]
        ax.bar(range(len(groups)), values,
               yerr=[lower, upper], color="#2196F3",
               edgecolor="black", alpha=0.85, capsize=5)
    else:
        ax.bar(range(len(groups)), values, color="#2196F3",
               edgecolor="black", alpha=0.85)

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=30)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate with sample count
    for i, (val, n) in enumerate(zip(values, df["n_samples"])):
        ax.text(i, val + 0.01, f"n={n}", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Subgroup analysis by demographics")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--bootstrap", action="store_true", help="Compute bootstrap CIs (slower)")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, label_map, ckpt = load_checkpoint(args.checkpoint, device)

    manifest_path = cfg["data"]["manifest"]
    target_col = cfg["data"]["target_col"]
    uses_metadata = isinstance(model, ToothEmbeddingModelWithMetadata)
    dataset = build_eval_dataset(cfg, args.split, label_map, ckpt, uses_metadata)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    print(f"Analyzing {args.split} split: {len(dataset)} samples")

    print("Extracting embeddings...")
    embeddings, labels = extract_embeddings(model, loader, device)

    df = pd.read_csv(manifest_path, dtype=str)
    df_split = df[df["split"] == args.split].reset_index(drop=True)
    assert len(df_split) == len(embeddings)

    # Output
    if args.output_dir is None:
        output_dir = Path(args.checkpoint).parent / "analysis" / "subgroups"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Overall test set metrics for reference
    overall = compute_subgroup_metrics(embeddings, labels, min_samples=0, min_persons=0)
    print(f"\nOverall test metrics: Rank-1={overall['rank1_micro']:.4f}, AUC={overall['auc']:.4f}")

    all_results = []

    # 1. Sex
    print("\n--- Sex ---")
    sex_array = df_split["sex"].values
    sex_df = analyze_subgroup(embeddings, labels, sex_array, "sex",
                               min_samples=200, min_persons=20, bootstrap=args.bootstrap)
    if len(sex_df) > 0:
        sex_df.to_csv(output_dir / "subgroup_sex.csv", index=False)
        plot_subgroup_bar(sex_df, "rank1_micro", "group", "Rank-1 by Sex",
                          output_dir / "subgroup_sex.png")
        all_results.append(sex_df)
        print(sex_df[["group", "n_samples", "n_persons", "rank1_micro", "auc"]].to_string(index=False))

    # 2. Age bucket
    print("\n--- Age Bucket ---")
    age_array = np.array([age_bucket(a) for a in df_split["age"].values])
    age_df = analyze_subgroup(embeddings, labels, age_array, "age_bucket",
                               min_samples=200, min_persons=20, bootstrap=args.bootstrap)
    if len(age_df) > 0:
        age_df.to_csv(output_dir / "subgroup_age.csv", index=False)
        plot_subgroup_bar(age_df, "rank1_micro", "group", "Rank-1 by Age Group",
                          output_dir / "subgroup_age.png")
        all_results.append(age_df)
        print(age_df[["group", "n_samples", "n_persons", "rank1_micro", "auc"]].to_string(index=False))

    # 3. Is deciduous
    print("\n--- Deciduous vs Permanent ---")
    dec_array = df_split["is_deciduous"].values
    dec_df = analyze_subgroup(embeddings, labels, dec_array, "is_deciduous",
                               min_samples=100, min_persons=10, bootstrap=args.bootstrap)
    if len(dec_df) > 0:
        dec_df.to_csv(output_dir / "subgroup_deciduous.csv", index=False)
        plot_subgroup_bar(dec_df, "rank1_micro", "group", "Rank-1 by Dentition (False=Permanent, True=Deciduous)",
                          output_dir / "subgroup_deciduous.png")
        all_results.append(dec_df)
        print(dec_df[["group", "n_samples", "n_persons", "rank1_micro", "auc"]].to_string(index=False))

    # 4. Erupted (600 subset)
    print("\n--- Erupted (600 subset) ---")
    erupt_array = df_split["erupted"].values
    # Replace nan/empty with None to skip
    erupt_array = np.array([e if e in ("True", "False") else None for e in erupt_array])
    erupt_df = analyze_subgroup(embeddings, labels, erupt_array, "erupted",
                                 min_samples=50, min_persons=10, bootstrap=args.bootstrap)
    if len(erupt_df) > 0:
        erupt_df.to_csv(output_dir / "subgroup_erupted.csv", index=False)
        plot_subgroup_bar(erupt_df, "rank1_micro", "group", "Rank-1 by Eruption Status (subset, n≈600 images)",
                          output_dir / "subgroup_erupted.png")
        all_results.append(erupt_df)
        print(erupt_df[["group", "n_samples", "n_persons", "rank1_micro", "auc"]].to_string(index=False))

    # 5. Root complete (600 subset)
    print("\n--- Root Complete (600 subset) ---")
    root_array = df_split["root_complete"].values
    root_array = np.array([r if r in ("True", "False") else None for r in root_array])
    root_df = analyze_subgroup(embeddings, labels, root_array, "root_complete",
                                min_samples=50, min_persons=10, bootstrap=args.bootstrap)
    if len(root_df) > 0:
        root_df.to_csv(output_dir / "subgroup_root.csv", index=False)
        plot_subgroup_bar(root_df, "rank1_micro", "group", "Rank-1 by Root Completion (subset, n≈600 images)",
                          output_dir / "subgroup_root.png")
        all_results.append(root_df)
        print(root_df[["group", "n_samples", "n_persons", "rank1_micro", "auc"]].to_string(index=False))

    # Combined summary
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(output_dir / "all_subgroups.csv", index=False)

        # Summary MD
        lines = ["# Subgroup Analysis — Summary\n"]
        lines.append(f"**Overall test Rank-1:** {overall['rank1_micro']:.4f}, AUC: {overall['auc']:.4f}\n")
        for sub_df, name in zip(all_results,
                                 ["Sex", "Age Bucket", "Dentition", "Eruption (subset)", "Root (subset)"]):
            if len(sub_df) == 0:
                continue
            lines.append(f"\n## {name}\n")
            cols = ["group", "n_samples", "n_persons", "rank1_micro", "rank5", "auc", "mAP"]
            lines.append(sub_df[cols].to_markdown(index=False, floatfmt=".4f"))
            lines.append("\n")
        with open(output_dir / "summary.md", "w") as f:
            f.write("\n".join(lines))

    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
