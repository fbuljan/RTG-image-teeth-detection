"""
Evaluate a trained embedding model.

Verification: all-pairs cosine similarity → ROC AUC, EER
Retrieval: nearest-neighbor search → Rank-1, Rank-5, Rank-10, mAP, CMC curve

Usage:
    python -m identification.evaluation.evaluate_embedding --checkpoint path/to/best.pt
    python -m identification.evaluation.evaluate_embedding --checkpoint path/to/best.pt --split test
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from identification.data.tooth_dataset import ToothDataset, get_val_transforms
from identification.models.embedding_model import ToothEmbeddingModel


def load_checkpoint(checkpoint_path: str, device: str):
    """Load embedding model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    label_map = ckpt["label_map"]

    model = ToothEmbeddingModel(
        embedding_dim=cfg["model"].get("embedding_dim", 128),
        pretrained=False,
        dropout=cfg["model"].get("dropout", 0.2),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model, cfg, label_map, ckpt


@torch.no_grad()
def extract_embeddings(model, loader, device):
    """Extract embeddings and labels for all samples."""
    all_embeddings = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        emb = model(images)
        all_embeddings.append(emb.cpu())
        all_labels.append(labels)

    if device == "mps":
        torch.mps.empty_cache()

    return torch.cat(all_embeddings, dim=0).numpy(), torch.cat(all_labels, dim=0).numpy()


def evaluate_verification(embeddings: np.ndarray, labels: np.ndarray):
    """
    All-pairs verification evaluation.

    For each pair, compute cosine similarity and check if same person.
    Returns ROC AUC, EER, and data for ROC curve plotting.
    """
    # Cosine similarity matrix (embeddings are L2-normalized)
    sim_matrix = embeddings @ embeddings.T
    n = len(labels)

    # Extract upper triangle (exclude diagonal)
    triu_idx = np.triu_indices(n, k=1)
    pair_scores = sim_matrix[triu_idx]
    pair_labels = (labels[triu_idx[0]] == labels[triu_idx[1]]).astype(int)

    # ROC curve
    fpr, tpr, thresholds = roc_curve(pair_labels, pair_scores)
    auc = roc_auc_score(pair_labels, pair_scores)

    # EER: where FPR == 1 - TPR (FAR == FRR)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2)
    eer_threshold = float(thresholds[eer_idx])

    # Stats
    num_positive = int(pair_labels.sum())
    num_negative = int(len(pair_labels) - num_positive)

    return {
        "auc": float(auc),
        "eer": eer,
        "eer_threshold": eer_threshold,
        "num_pairs": int(len(pair_labels)),
        "num_positive_pairs": num_positive,
        "num_negative_pairs": num_negative,
        "_roc_fpr": fpr,
        "_roc_tpr": tpr,
    }


def evaluate_retrieval(embeddings: np.ndarray, labels: np.ndarray):
    """
    Closed-set retrieval evaluation.

    For each sample, find nearest neighbors and check if they share the same person.
    Returns Rank-1, Rank-5, Rank-10, mAP, and CMC curve.
    """
    sim_matrix = embeddings @ embeddings.T
    n = len(labels)

    # Exclude self-matches
    np.fill_diagonal(sim_matrix, -float("inf"))

    # Sort by similarity (descending)
    ranked_indices = np.argsort(-sim_matrix, axis=1)
    ranked_labels = labels[ranked_indices]

    # Expand labels for comparison
    query_labels = labels[:, np.newaxis]  # (N, 1)
    matches = (ranked_labels == query_labels)  # (N, N-1)

    # Rank-K accuracy
    rank1 = float(matches[:, 0].mean())
    rank5 = float(matches[:, :5].any(axis=1).mean())
    rank10 = float(matches[:, :10].any(axis=1).mean())

    # CMC curve (cumulative match characteristic)
    max_rank = min(50, n - 1)
    cmc = np.zeros(max_rank)
    for r in range(max_rank):
        cmc[r] = float(matches[:, :r + 1].any(axis=1).mean())

    # Mean average precision
    aps = []
    for i in range(n):
        match_positions = np.where(matches[i])[0]
        if len(match_positions) == 0:
            aps.append(0.0)
            continue
        # Precision at each relevant position
        precisions = np.arange(1, len(match_positions) + 1) / (match_positions + 1)
        aps.append(float(precisions.mean()))
    mAP = float(np.mean(aps))

    # Per-person Rank-1 (macro average)
    unique_labels = np.unique(labels)
    per_person_rank1 = []
    for lbl in unique_labels:
        mask = labels == lbl
        person_matches = matches[mask, 0]
        per_person_rank1.append(float(person_matches.mean()))
    macro_rank1 = float(np.mean(per_person_rank1))

    return {
        "rank1_micro": rank1,
        "rank1_macro": macro_rank1,
        "rank5": rank5,
        "rank10": rank10,
        "mAP": mAP,
        "_cmc": cmc,
    }


def plot_roc_curve(fpr, tpr, auc, eer, output_path):
    """Plot ROC curve with AUC and EER annotated."""
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)

    # EER point
    eer_x = eer
    eer_y = 1 - eer
    ax.plot(eer_x, eer_y, "ro", markersize=8, label=f"EER={eer:.4f}")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Verification ROC Curve")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_cmc_curve(cmc, output_path):
    """Plot cumulative match characteristic curve."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ranks = np.arange(1, len(cmc) + 1)
    ax.plot(ranks, cmc, linewidth=2, marker="o", markersize=3)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Recognition Rate")
    ax.set_title("Cumulative Match Characteristic (CMC)")
    ax.set_xlim([1, len(cmc)])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)

    # Annotate key ranks
    for r in [1, 5, 10]:
        if r <= len(cmc):
            ax.annotate(f"R{r}={cmc[r-1]:.3f}", xy=(r, cmc[r-1]),
                        xytext=(r + 2, cmc[r-1] - 0.05), fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate tooth embedding model")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--split", default="test", help="Split to evaluate on")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, label_map, ckpt = load_checkpoint(args.checkpoint, device)

    manifest_path = cfg["data"]["manifest"]
    target_col = cfg["data"]["target_col"]

    dataset = ToothDataset(
        manifest_path=manifest_path,
        split=args.split,
        crop_mode=cfg["data"]["crop_mode"],
        target_col=target_col,
        transform=get_val_transforms(),
        label_map=label_map,
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    print(f"Evaluating on {args.split}: {len(dataset)} samples, {len(label_map)} persons")

    # Extract embeddings
    print("Extracting embeddings...")
    embeddings, labels = extract_embeddings(model, loader, device)
    print(f"Embeddings shape: {embeddings.shape}")

    # Verification
    print("Computing verification metrics...")
    ver_metrics = evaluate_verification(embeddings, labels)

    # Retrieval
    print("Computing retrieval metrics...")
    ret_metrics = evaluate_retrieval(embeddings, labels)

    # Output directory
    if args.output_dir is None:
        output_dir = Path(args.checkpoint).parent / f"eval_{args.split}"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics = {
        "verification": {k: v for k, v in ver_metrics.items() if not k.startswith("_")},
        "retrieval": {k: v for k, v in ret_metrics.items() if not k.startswith("_")},
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Plots
    plot_roc_curve(
        ver_metrics["_roc_fpr"], ver_metrics["_roc_tpr"],
        ver_metrics["auc"], ver_metrics["eer"],
        output_dir / "roc_curve.png",
    )
    plot_cmc_curve(ret_metrics["_cmc"], output_dir / "cmc_curve.png")

    # Print summary
    print(f"\n{'='*50}")
    print(f"Results ({args.split} split)")
    print(f"{'='*50}")
    print(f"Verification:")
    print(f"  AUC:          {ver_metrics['auc']:.4f}")
    print(f"  EER:          {ver_metrics['eer']:.4f}")
    print(f"  Pairs:        {ver_metrics['num_positive_pairs']:,} pos / {ver_metrics['num_negative_pairs']:,} neg")
    print(f"Retrieval:")
    print(f"  Rank-1 micro: {ret_metrics['rank1_micro']:.4f}")
    print(f"  Rank-1 macro: {ret_metrics['rank1_macro']:.4f}")
    print(f"  Rank-5:       {ret_metrics['rank5']:.4f}")
    print(f"  Rank-10:      {ret_metrics['rank10']:.4f}")
    print(f"  mAP:          {ret_metrics['mAP']:.4f}")
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
