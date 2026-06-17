"""
Generate thesis-quality figures from saved metrics.

Reads metrics.json files from various run/analysis directories and produces
final comparison plots. All figures saved to identification/docs/figures/.

Figures generated:
1. Model comparison bar chart (AUC, Rank-1, mAP across all models)
2. Multi-tooth aggregation curve (Rank-1 vs n_query for baseline + FDI-init)
3. Single-tooth vs aggregated-gallery scaling (rank1 across n_query teeth)
4. Per-FDI Rank-1 (baseline)
5. Per-category Rank-1 + AUC (baseline)
6. Subgroup metrics (sex, age, deciduous, eruption, root)

Usage:
    python -m identification.scripts.generate_figures
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "identification" / "docs" / "figures"


# Models to compare. Each entry: (label, run_directory)
MODELS = [
    ("Baseline (raw)", "embedding_triplet_v1"),
    ("Masked", "embedding_triplet_masked_v1"),
    ("Metadata", "embedding_metadata_v1"),
    ("FDI-init", "embedding_fdi_init_v1"),
]


def load_json(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def fig_model_comparison():
    """Bar chart comparing models on AUC, Rank-1, mAP."""
    rows = []
    for label, run_dir in MODELS:
        metrics = load_json(REPO_ROOT / "identification" / "runs" / run_dir / "eval_test" / "metrics.json")
        if metrics is None:
            continue
        rows.append({
            "model": label,
            "AUC": metrics["verification"]["auc"],
            "EER": metrics["verification"]["eer"],
            "Rank-1": metrics["retrieval"]["rank1_micro"],
            "Rank-5": metrics["retrieval"]["rank5"],
            "mAP": metrics["retrieval"]["mAP"],
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = [("AUC", "Verification AUC"), ("Rank-1", "Single-tooth Rank-1"), ("mAP", "Retrieval mAP")]
    for ax, (col, title) in zip(axes, metrics):
        bars = ax.bar(df["model"], df[col], color=["#2196F3", "#4CAF50", "#FF9800", "#F44336"][:len(df)],
                      edgecolor="black", alpha=0.85)
        ax.set_title(title)
        ax.set_ylim(0, max(1.0, df[col].max() * 1.15))
        ax.set_xticklabels(df["model"], rotation=20, ha="right")
        ax.grid(True, alpha=0.3, axis="y")
        for b, v in zip(bars, df[col]):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f"{v:.3f}", ha="center", fontsize=9)

    fig.suptitle("Single-tooth metrics across embedding models")
    fig.tight_layout()
    fig.savefig(DOCS_DIR / "01_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 01_model_comparison.png")


def fig_multi_tooth_curve():
    """Multi-tooth aggregation: Rank-1 vs n_query for all 4 models + classification baseline."""
    fig, ax = plt.subplots(figsize=(11, 6))

    model_styles = [
        ("Baseline (raw)", "embedding_triplet_v1", "tab:blue", "o"),
        ("Masked", "embedding_triplet_masked_v1", "tab:green", "^"),
        ("Metadata", "embedding_metadata_v1", "tab:orange", "v"),
        ("FDI-init", "embedding_fdi_init_v1", "tab:red", "D"),
    ]

    for label, run_dir, color, marker in model_styles:
        metrics = load_json(REPO_ROOT / "identification" / "runs" / run_dir / "analysis" / "person_retrieval" / "metrics.json")
        if metrics is None:
            continue
        sweep = pd.DataFrame(metrics["sweep"])
        mean_sub = sweep[sweep["method"] == "mean"].sort_values("n_query")
        if len(mean_sub) == 0:
            continue
        ax.errorbar(mean_sub["n_query"], mean_sub["rank1_mean"],
                    yerr=mean_sub["rank1_std"], marker=marker, capsize=4,
                    label=label, color=color, linewidth=2, markersize=7)

    # Reference: classification baseline (FDI features)
    fdi_classifier = load_json(REPO_ROOT / "identification" / "runs" / "tooth_fdi_raw" / "analysis" / "as_embedding" / "metrics.json")
    if fdi_classifier is not None:
        sweep = pd.DataFrame(fdi_classifier["multi_tooth_sweep"])
        sweep_sorted = sweep.sort_values("n_query")
        ax.errorbar(sweep_sorted["n_query"], sweep_sorted["rank1_mean"],
                    yerr=sweep_sorted["rank1_std"], marker="s", capsize=4,
                    label="Classification baseline\n(FDI classifier features)",
                    color="tab:gray", linestyle="--", markersize=7)

    ax.set_xlabel("Number of teeth held out as query (gallery has the rest)")
    ax.set_ylabel("Rank-1 accuracy")
    ax.set_title("Multi-tooth aggregation: Rank-1 vs query size (mean pooling)")
    ax.set_xticks([1, 2, 4, 8, 16, 24, 32])
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    # Annotate peak
    ax.axvline(x=16, color="black", linestyle=":", alpha=0.3)
    ax.text(16.3, 0.05, "peak at\nn_query=16", fontsize=8, color="black", alpha=0.7)
    fig.tight_layout()
    fig.savefig(DOCS_DIR / "02_multi_tooth_aggregation.png", dpi=150)
    plt.close(fig)
    print("  ✓ 02_multi_tooth_aggregation.png")


def fig_per_fdi_baseline():
    """Per-FDI Rank-1 for the baseline model."""
    csv_path = REPO_ROOT / "identification" / "runs" / "embedding_triplet_v1" / "analysis" / "per_tooth" / "per_fdi_metrics.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if "fdi" not in df.columns or df.empty:
        return
    df["fdi"] = df["fdi"].astype(str)
    df = df.sort_values("fdi", key=lambda s: s.astype(int))

    color_map = {
        "incisor": "#2196F3", "canine": "#4CAF50", "premolar": "#FF9800", "molar": "#F44336",
        "deciduous_incisor": "#90CAF9", "deciduous_canine": "#A5D6A7",
        "deciduous_molar": "#FFCC80", "unknown": "#9E9E9E",
    }
    colors = [color_map.get(c, "#9E9E9E") for c in df["category"]]

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(range(len(df)), df["rank1"], color=colors, edgecolor="black", alpha=0.85)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["fdi"], rotation=90, fontsize=9)
    ax.set_xlabel("FDI tooth number")
    ax.set_ylabel("Rank-1 accuracy")
    ax.set_title("Per-FDI Rank-1 (baseline model, full-gallery retrieval)")
    ax.grid(True, alpha=0.3, axis="y")

    from matplotlib.patches import Patch
    legend = [Patch(facecolor=col, label=cat) for cat, col in color_map.items()
              if cat in set(df["category"])]
    ax.legend(handles=legend, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(DOCS_DIR / "03_per_fdi_rank1.png", dpi=150)
    plt.close(fig)
    print("  ✓ 03_per_fdi_rank1.png")


def fig_per_category():
    """Per-anatomical-category Rank-1 + AUC for baseline."""
    csv_path = REPO_ROOT / "identification" / "runs" / "embedding_triplet_v1" / "analysis" / "per_tooth" / "per_category_metrics.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        return

    df = df.sort_values("rank1_micro", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, col, title in [(axes[0], "rank1_micro", "Rank-1 by tooth category"),
                            (axes[1], "auc", "Verification AUC by tooth category")]:
        ax.barh(df["group"], df[col], color="#2196F3", edgecolor="black", alpha=0.85)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel(col)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="x")
        for i, (g, v) in enumerate(zip(df["group"], df[col])):
            ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(DOCS_DIR / "04_per_category.png", dpi=150)
    plt.close(fig)
    print("  ✓ 04_per_category.png")


def fig_subgroups():
    """Subgroup analysis: 4 panels for sex / age / deciduous / clinical."""
    base = REPO_ROOT / "identification" / "runs" / "embedding_triplet_v1" / "analysis" / "subgroups"
    files = {
        "Sex": "subgroup_sex.csv",
        "Age bucket": "subgroup_age.csv",
        "Dentition": "subgroup_deciduous.csv",
        "Eruption (subset)": "subgroup_erupted.csv",
        "Root complete (subset)": "subgroup_root.csv",
    }

    available = [(t, base / f) for t, f in files.items() if (base / f).exists()]
    if not available:
        return
    n = len(available)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(13, 4 * rows))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for ax, (title, path) in zip(axes, available):
        df = pd.read_csv(path)
        if df.empty:
            ax.set_visible(False)
            continue
        ax.bar(df["group"], df["rank1_micro"], color="#2196F3", edgecolor="black", alpha=0.85)
        ax.set_title(title)
        ax.set_ylim(0, df["rank1_micro"].max() * 1.4)
        ax.grid(True, alpha=0.3, axis="y")
        for i, (g, v, n_samp) in enumerate(zip(df["group"], df["rank1_micro"], df["n_samples"])):
            ax.text(i, v + 0.005, f"{v:.3f}\nn={n_samp}", ha="center", fontsize=8)
        ax.set_xticklabels(df["group"], rotation=15)
        ax.set_ylabel("Rank-1")

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Subgroup Rank-1 (baseline model)")
    fig.tight_layout()
    fig.savefig(DOCS_DIR / "05_subgroups.png", dpi=150)
    plt.close(fig)
    print("  ✓ 05_subgroups.png")


def main():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing figures to {DOCS_DIR}")
    fig_model_comparison()
    fig_multi_tooth_curve()
    fig_per_fdi_baseline()
    fig_per_category()
    fig_subgroups()
    print("Done.")


if __name__ == "__main__":
    main()
