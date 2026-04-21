"""
Visualize embeddings using t-SNE and UMAP.

Generates scatter plots colored by person, tooth FDI, quadrant, and jaw.

Usage:
    python -m identification.evaluation.visualize_embeddings --checkpoint path/to/best.pt
    python -m identification.evaluation.visualize_embeddings --checkpoint path/to/best.pt --split test --n-persons 20
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from identification.data.tooth_dataset import ToothDataset, get_val_transforms
from identification.models.embedding_model import ToothEmbeddingModel


def load_checkpoint(checkpoint_path: str, device: str):
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

    return model, cfg, label_map


@torch.no_grad()
def extract_embeddings(model, loader, device):
    all_embeddings = []
    all_labels = []
    for images, labels in loader:
        images = images.to(device)
        emb = model(images)
        all_embeddings.append(emb.cpu())
        all_labels.append(labels)
    return torch.cat(all_embeddings).numpy(), torch.cat(all_labels).numpy()


def reduce_dimensions(embeddings, method="tsne", perplexity=30):
    """Reduce to 2D using t-SNE or UMAP."""
    if method == "tsne":
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
        return reducer.fit_transform(embeddings)
    elif method == "umap":
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        return reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown method: {method}")


def plot_embeddings(coords_2d, color_labels, title, output_path, max_legend=20):
    """Scatter plot of 2D embeddings colored by labels."""
    fig, ax = plt.subplots(figsize=(12, 10))

    unique_labels = sorted(set(color_labels))
    n_colors = len(unique_labels)

    if n_colors <= max_legend:
        cmap = plt.cm.get_cmap("tab20", n_colors)
        for i, lbl in enumerate(unique_labels):
            mask = np.array(color_labels) == lbl
            ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                       c=[cmap(i)], s=15, alpha=0.6, label=str(lbl))
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7, ncol=1)
    else:
        # Too many labels for legend — just use colormap
        label_to_int = {l: i for i, l in enumerate(unique_labels)}
        colors = [label_to_int[l] for l in color_labels]
        scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1],
                             c=colors, cmap="tab20", s=10, alpha=0.5)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize tooth embeddings")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--split", default="test", help="Split to visualize")
    parser.add_argument("--n-persons", type=int, default=20, help="Number of persons for person-colored plot")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--method", default="tsne", choices=["tsne", "umap"], help="Dimensionality reduction method")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, label_map = load_checkpoint(args.checkpoint, device)

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

    # Extract embeddings
    print(f"Extracting embeddings for {len(dataset)} samples...")
    embeddings, labels = extract_embeddings(model, loader, device)

    # Load metadata from manifest (same order as dataset since shuffle=False)
    manifest_df = pd.read_csv(manifest_path, dtype=str)
    split_df = manifest_df[manifest_df["split"] == args.split].reset_index(drop=True)
    person_ids = split_df["person_id"].tolist()
    tooth_fdis = split_df["tooth_fdi"].tolist()
    quadrants = split_df["quadrant"].tolist()
    jaws = split_df["jaw"].tolist()

    # Output directory
    if args.output_dir is None:
        output_dir = Path(args.checkpoint).parent / f"viz_{args.split}"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Subset for person-colored plot (too many persons to show all)
    unique_persons = sorted(set(person_ids))
    selected_persons = set(unique_persons[:args.n_persons])
    person_mask = np.array([p in selected_persons for p in person_ids])

    # 1. Person-colored plot (subset)
    print(f"Computing {args.method.upper()} for {person_mask.sum()} samples ({args.n_persons} persons)...")
    subset_embeddings = embeddings[person_mask]
    subset_persons = [p[:8] for p, m in zip(person_ids, person_mask) if m]  # truncate IDs for readability

    coords_person = reduce_dimensions(subset_embeddings, method=args.method)
    plot_embeddings(
        coords_person, subset_persons,
        f"Embeddings by person ({args.n_persons} persons, {args.method.upper()})",
        output_dir / f"by_person_{args.method}.png",
    )
    print(f"  Saved: by_person_{args.method}.png")

    # 2. Full dataset plots (tooth FDI, quadrant, jaw)
    print(f"Computing {args.method.upper()} for all {len(embeddings)} samples...")
    coords_all = reduce_dimensions(embeddings, method=args.method)

    plot_embeddings(
        coords_all, tooth_fdis,
        f"Embeddings by tooth FDI ({args.method.upper()})",
        output_dir / f"by_tooth_fdi_{args.method}.png",
    )
    print(f"  Saved: by_tooth_fdi_{args.method}.png")

    plot_embeddings(
        coords_all, quadrants,
        f"Embeddings by quadrant ({args.method.upper()})",
        output_dir / f"by_quadrant_{args.method}.png",
    )
    print(f"  Saved: by_quadrant_{args.method}.png")

    plot_embeddings(
        coords_all, jaws,
        f"Embeddings by jaw ({args.method.upper()})",
        output_dir / f"by_jaw_{args.method}.png",
    )
    print(f"  Saved: by_jaw_{args.method}.png")

    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
