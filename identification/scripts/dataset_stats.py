"""
Compute and log dataset statistics from the clean manifest.

Outputs: dataset_stats.json, summary printout, matplotlib plots.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_manifest(path):
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def compute_stats(rows):
    stats = {}

    # Basic counts
    stats["total_crops"] = len(rows)
    persons = set(r["person_id"] for r in rows)
    stats["total_persons"] = len(persons)
    stats["crops_per_person_mean"] = round(len(rows) / len(persons), 1)

    # Per split
    split_counts = Counter(r["split"] for r in rows)
    stats["splits"] = dict(split_counts)
    split_persons = {}
    for split in ["train", "val", "test"]:
        split_persons[split] = len(set(r["person_id"] for r in rows if r["split"] == split))
    stats["persons_per_split"] = split_persons

    # Teeth per person distribution
    teeth_per_person = Counter(r["person_id"] for r in rows)
    tpp_values = list(teeth_per_person.values())
    stats["teeth_per_person"] = {
        "min": min(tpp_values), "max": max(tpp_values),
        "mean": round(np.mean(tpp_values), 1),
        "median": round(np.median(tpp_values), 1),
    }

    # FDI distribution
    fdi_counts = Counter(r["tooth_fdi"] for r in rows)
    stats["fdi_distribution"] = dict(sorted(fdi_counts.items(), key=lambda x: int(x[0])))
    stats["unique_fdi_values"] = len(fdi_counts)

    # Quadrant distribution
    stats["quadrant_distribution"] = dict(Counter(r["quadrant"] for r in rows))

    # Jaw distribution
    stats["jaw_distribution"] = dict(Counter(r["jaw"] for r in rows))

    # Deciduous vs permanent
    deciduous_count = sum(1 for r in rows if r["is_deciduous"] == "True")
    stats["deciduous_count"] = deciduous_count
    stats["permanent_count"] = len(rows) - deciduous_count

    # Demographics
    ages = [float(r["age"]) for r in rows if r["age"]]
    if ages:
        stats["age"] = {
            "min": round(min(ages), 1), "max": round(max(ages), 1),
            "mean": round(np.mean(ages), 1),
        }
    stats["sex_distribution"] = dict(Counter(r["sex"] for r in rows if r["sex"]))
    stats["age_group_distribution"] = dict(Counter(r["age_group"] for r in rows if r["age_group"]))

    # Eruption/root data availability
    with_eruption = sum(1 for r in rows if r["erupted"] != "")
    stats["with_eruption_data"] = with_eruption
    if with_eruption > 0:
        erupted_rows = [r for r in rows if r["erupted"] != ""]
        stats["eruption_distribution"] = dict(Counter(r["erupted"] for r in erupted_rows))
        stats["root_complete_distribution"] = dict(Counter(r["root_complete"] for r in erupted_rows))

    # Rare teeth (fewer than 50 samples)
    rare = {fdi: count for fdi, count in fdi_counts.items() if count < 50}
    stats["rare_teeth_under_50"] = dict(sorted(rare.items(), key=lambda x: x[1]))

    return stats, teeth_per_person, fdi_counts


def plot_stats(stats, teeth_per_person, fdi_counts, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Teeth per person histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    values = list(teeth_per_person.values())
    ax.hist(values, bins=range(min(values), max(values) + 2), edgecolor="black", alpha=0.7)
    ax.set_xlabel("Teeth per person")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of teeth per person")
    fig.tight_layout()
    fig.savefig(output_dir / "teeth_per_person.png", dpi=150)
    plt.close(fig)

    # 2. FDI distribution bar chart
    fig, ax = plt.subplots(figsize=(16, 5))
    fdis_sorted = sorted(fdi_counts.keys(), key=int)
    counts = [fdi_counts[f] for f in fdis_sorted]
    colors = ["#2196F3" if int(f) < 50 else "#FF9800" for f in fdis_sorted]
    ax.bar(fdis_sorted, counts, color=colors, edgecolor="black", alpha=0.8)
    ax.set_xlabel("FDI tooth number")
    ax.set_ylabel("Count")
    ax.set_title("Crops per FDI tooth number (blue=permanent, orange=deciduous)")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(output_dir / "fdi_distribution.png", dpi=150)
    plt.close(fig)

    # 3. Age distribution
    age_groups = stats.get("age_group_distribution", {})
    if age_groups:
        fig, ax = plt.subplots(figsize=(10, 5))
        groups = sorted(age_groups.keys())
        ax.bar(groups, [age_groups[g] for g in groups], edgecolor="black", alpha=0.7)
        ax.set_xlabel("Age group")
        ax.set_ylabel("Count (crops)")
        ax.set_title("Age group distribution")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(output_dir / "age_distribution.png", dpi=150)
        plt.close(fig)

    print(f"Plots saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Compute dataset statistics")
    parser.add_argument("--manifest", default="identification/data/manifest_clean.csv")
    parser.add_argument("--output-json", default="identification/data/dataset_stats.json")
    parser.add_argument("--output-plots", default="identification/data/plots")
    args = parser.parse_args()

    rows = load_manifest(args.manifest)
    stats, teeth_per_person, fdi_counts = compute_stats(rows)

    # Save JSON
    with open(args.output_json, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved: {args.output_json}")

    # Print summary
    print(f"\n{'='*50}")
    print(f"Dataset Summary")
    print(f"{'='*50}")
    print(f"Total crops: {stats['total_crops']}")
    print(f"Total persons: {stats['total_persons']}")
    print(f"Crops/person: {stats['teeth_per_person']}")
    print(f"Unique FDI values: {stats['unique_fdi_values']}")
    print(f"Permanent: {stats['permanent_count']}, Deciduous: {stats['deciduous_count']}")
    print(f"Splits: {stats['splits']}")
    print(f"Persons/split: {stats['persons_per_split']}")
    if "age" in stats:
        print(f"Age range: {stats['age']['min']} - {stats['age']['max']} (mean {stats['age']['mean']})")
    print(f"Sex: {stats.get('sex_distribution', {})}")
    print(f"With eruption data: {stats['with_eruption_data']}")
    if stats.get("rare_teeth_under_50"):
        print(f"Rare teeth (<50 samples): {stats['rare_teeth_under_50']}")

    # Generate plots
    plot_stats(stats, teeth_per_person, fdi_counts, args.output_plots)


if __name__ == "__main__":
    main()
