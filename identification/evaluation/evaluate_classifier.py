"""
Evaluate a trained tooth classifier.

Computes overall accuracy, per-class accuracy, confusion matrix,
per-quadrant/jaw accuracy, and generates plots.

Usage:
    python -m identification.evaluation.evaluate_classifier --checkpoint path/to/best.pt
    python -m identification.evaluation.evaluate_classifier --checkpoint path/to/best.pt --split test
    python -m identification.evaluation.evaluate_classifier --compare path/raw/best.pt path/masked/best.pt
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from identification.data.tooth_dataset import ToothDataset, get_val_transforms
from identification.models.classifier import ToothClassifier


def load_checkpoint(checkpoint_path: str, device: str):
    """Load model, config, and label_map from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    label_map = ckpt["label_map"]
    num_classes = len(label_map)

    model = ToothClassifier(
        num_classes=num_classes,
        pretrained=False,
        dropout=cfg["model"].get("dropout", 0.2),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model, cfg, label_map, ckpt


def build_filter_fn(cfg: dict):
    target_col = cfg["data"]["target_col"]
    if cfg["data"].get("filter_nonempty", False):
        return lambda df: df[df[target_col].notna() & (df[target_col] != "")]
    return None


@torch.no_grad()
def predict(model, loader, device):
    """Run inference, return all predictions and labels."""
    all_preds = []
    all_labels = []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def compute_metrics(y_true, y_pred, label_map, manifest_df):
    """Compute comprehensive metrics."""
    label_to_name = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)
    class_names = [label_to_name[i] for i in range(num_classes)]

    metrics = {}

    # Overall accuracy
    metrics["overall_accuracy"] = float((y_true == y_pred).mean())

    # Per-class accuracy and counts
    per_class = {}
    for idx in range(num_classes):
        name = label_to_name[idx]
        mask = y_true == idx
        count = int(mask.sum())
        acc = float((y_pred[mask] == idx).mean()) if count > 0 else 0.0
        per_class[name] = {"accuracy": acc, "count": count}
    metrics["per_class"] = per_class

    # Accuracy on classes with >= 50 samples
    well_represented = [
        per_class[name]["accuracy"]
        for name in per_class
        if per_class[name]["count"] >= 50
    ]
    if well_represented:
        metrics["accuracy_classes_ge50"] = float(np.mean(well_represented))

    # Per-quadrant accuracy (derive from class names if FDI)
    if all(name.isdigit() and len(name) == 2 for name in class_names):
        quadrant_correct = defaultdict(int)
        quadrant_total = defaultdict(int)
        jaw_correct = defaultdict(int)
        jaw_total = defaultdict(int)

        for idx in range(num_classes):
            name = label_to_name[idx]
            q = int(name[0])
            jaw = "upper" if q in (1, 2, 5, 6) else "lower"
            mask = y_true == idx
            count = int(mask.sum())
            correct = int((y_pred[mask] == idx).sum())
            quadrant_correct[q] += correct
            quadrant_total[q] += count
            jaw_correct[jaw] += correct
            jaw_total[jaw] += count

        metrics["per_quadrant"] = {
            str(q): float(quadrant_correct[q] / quadrant_total[q]) if quadrant_total[q] > 0 else 0.0
            for q in sorted(quadrant_total.keys())
        }
        metrics["per_jaw"] = {
            jaw: float(jaw_correct[jaw] / jaw_total[jaw]) if jaw_total[jaw] > 0 else 0.0
            for jaw in ["upper", "lower"]
        }

    # Binary metrics (precision, recall, F1)
    if num_classes <= 2:
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1)
        metrics["precision"] = float(p)
        metrics["recall"] = float(r)
        metrics["f1"] = float(f1)

    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names, output_path):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    n = len(class_names)

    if n <= 10:
        fig, ax = plt.subplots(figsize=(8, 7))
        fontsize = 10
    else:
        fig, ax = plt.subplots(figsize=(20, 18))
        fontsize = 5

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=90, fontsize=fontsize)
    ax.set_yticklabels(class_names, fontsize=fontsize)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_per_class_accuracy(per_class, output_path):
    """Bar chart of per-class accuracy, color-coded permanent vs deciduous."""
    names = sorted(per_class.keys(), key=lambda x: int(x) if x.isdigit() else x)
    accs = [per_class[n]["accuracy"] for n in names]
    counts = [per_class[n]["count"] for n in names]

    colors = []
    for n in names:
        if n.isdigit() and int(n) >= 50:
            colors.append("#FF9800")  # deciduous = orange
        else:
            colors.append("#2196F3")  # permanent = blue

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    ax1.bar(range(len(names)), accs, color=colors, edgecolor="black", alpha=0.8)
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Per-class accuracy (blue=permanent, orange=deciduous)")
    ax1.set_ylim(0, 1.05)

    ax2.bar(range(len(names)), counts, color=colors, edgecolor="black", alpha=0.8)
    ax2.set_ylabel("Sample count")
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=90, fontsize=7)
    ax2.set_xlabel("FDI tooth number")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_quadrant_accuracy(per_quadrant, output_path):
    """Bar chart of per-quadrant accuracy."""
    quads = sorted(per_quadrant.keys(), key=int)
    accs = [per_quadrant[q] for q in quads]
    labels = [f"Q{q}" for q in quads]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2196F3" if int(q) <= 4 else "#FF9800" for q in quads]
    ax.bar(labels, accs, color=colors, edgecolor="black", alpha=0.8)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-quadrant accuracy (blue=permanent, orange=deciduous)")
    ax.set_ylim(0, 1.05)
    for i, (label, acc) in enumerate(zip(labels, accs)):
        ax.text(i, acc + 0.02, f"{acc:.1%}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def evaluate_single(checkpoint_path: str, split: str, crop_mode: str = None, output_dir: str = None):
    """Evaluate a single checkpoint."""
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, label_map, ckpt = load_checkpoint(checkpoint_path, device)

    if crop_mode is None:
        crop_mode = cfg["data"]["crop_mode"]

    filter_fn = build_filter_fn(cfg)
    manifest_path = cfg["data"]["manifest"]
    target_col = cfg["data"]["target_col"]

    dataset = ToothDataset(
        manifest_path=manifest_path,
        split=split,
        crop_mode=crop_mode,
        target_col=target_col,
        filter_fn=filter_fn,
        transform=get_val_transforms(),
        label_map=label_map,
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    print(f"Evaluating on {split} split: {len(dataset)} samples, {len(label_map)} classes")

    # Load manifest for metadata
    manifest_df = pd.read_csv(manifest_path, dtype=str)

    y_pred, y_true = predict(model, loader, device)
    metrics = compute_metrics(y_true, y_pred, label_map, manifest_df)

    # Output
    if output_dir is None:
        output_dir = Path(checkpoint_path).parent / f"eval_{split}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Classification report
    label_to_name = {v: k for k, v in label_map.items()}
    class_names = [label_to_name[i] for i in range(len(label_map))]
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    with open(output_dir / "classification_report.txt", "w") as f:
        f.write(report)

    # Plots
    plot_confusion_matrix(y_true, y_pred, class_names, output_dir / "confusion_matrix.png")

    if "per_class" in metrics:
        plot_per_class_accuracy(metrics["per_class"], output_dir / "per_class_accuracy.png")

    if "per_quadrant" in metrics:
        plot_quadrant_accuracy(metrics["per_quadrant"], output_dir / "quadrant_accuracy.png")

    # Print summary
    print(f"\n{'='*50}")
    print(f"Results ({split} split, crop_mode={crop_mode})")
    print(f"{'='*50}")
    print(f"Overall accuracy: {metrics['overall_accuracy']:.4f}")
    if "accuracy_classes_ge50" in metrics:
        print(f"Accuracy (classes ≥50 samples): {metrics['accuracy_classes_ge50']:.4f}")
    if "per_quadrant" in metrics:
        for q, acc in metrics["per_quadrant"].items():
            print(f"  Quadrant {q}: {acc:.4f}")
    if "per_jaw" in metrics:
        for jaw, acc in metrics["per_jaw"].items():
            print(f"  {jaw}: {acc:.4f}")
    if "precision" in metrics:
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1: {metrics['f1']:.4f}")
    print(f"\nSaved to: {output_dir}")

    return metrics


def compare_checkpoints(paths: list, split: str):
    """Compare two checkpoints side by side."""
    results = []
    for path in paths:
        print(f"\n--- Evaluating {path} ---")
        m = evaluate_single(path, split)
        results.append({"checkpoint": path, **m})

    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    for r in results:
        name = Path(r["checkpoint"]).parent.name
        print(f"  {name}: accuracy={r['overall_accuracy']:.4f}", end="")
        if "accuracy_classes_ge50" in r:
            print(f", acc_ge50={r['accuracy_classes_ge50']:.4f}", end="")
        print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate tooth classifier")
    parser.add_argument("--checkpoint", help="Path to checkpoint")
    parser.add_argument("--split", default="test", help="Split to evaluate on")
    parser.add_argument("--crop-mode", default=None, help="Override crop mode from config")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--compare", nargs="+", help="Compare multiple checkpoints")
    args = parser.parse_args()

    if args.compare:
        compare_checkpoints(args.compare, args.split)
    elif args.checkpoint:
        evaluate_single(args.checkpoint, args.split, args.crop_mode, args.output_dir)
    else:
        parser.error("Provide --checkpoint or --compare")


if __name__ == "__main__":
    main()
