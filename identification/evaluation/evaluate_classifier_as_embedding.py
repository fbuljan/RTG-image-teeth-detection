"""
Evaluate a classifier checkpoint as a feature extractor for person retrieval.

Loads the ResNet-18 backbone from a classifier (FDI / eruption / root), extracts
512-dim features for the test split, L2-normalizes, then runs the same
verification + retrieval + multi-tooth aggregation evaluation as the metric
learning models.

This is the "direct classification baseline" comparison: does a model trained
for classification transfer to person retrieval better or worse than one trained
with metric learning?

Usage:
    python -m identification.evaluation.evaluate_classifier_as_embedding \
        --checkpoint identification/runs/tooth_fdi_raw/best.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from identification.data.tooth_dataset import ToothDataset, get_val_transforms
from identification.models.classifier import ToothClassifier
from identification.evaluation.evaluate_embedding import (
    evaluate_retrieval,
    evaluate_verification,
)
from identification.evaluation.evaluate_person_retrieval import (
    evaluate_multi_tooth,
    evaluate_single_tooth_vs_aggregated_gallery,
)


def load_classifier_checkpoint(checkpoint_path, device):
    """Load a classifier checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    label_map = ckpt["label_map"]

    model = ToothClassifier(
        num_classes=len(label_map),
        pretrained=False,
        dropout=cfg["model"].get("dropout", 0.2),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model, cfg, label_map


@torch.no_grad()
def extract_classifier_features(model, loader, device):
    """
    Extract penultimate (backbone) features from a classifier.

    The model's backbone outputs 512-dim features (since classifier.fc was replaced).
    We L2-normalize them so we can use cosine similarity directly.
    """
    all_features = []
    all_labels_str = []

    for images, _ in loader:
        images = images.to(device)
        feats = model.get_features(images)  # (B, 512)
        feats = F.normalize(feats, p=2, dim=1)
        all_features.append(feats.cpu())

    if device == "mps":
        torch.mps.empty_cache()

    return torch.cat(all_features, dim=0).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to a classifier checkpoint (e.g. tooth_fdi_raw/best.pt)")
    parser.add_argument("--manifest", default="identification/data/manifest_clean.csv")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, _ = load_classifier_checkpoint(args.checkpoint, device)
    print(f"Loaded classifier: target={cfg['data']['target_col']}, classes={len(_)}")

    # Build a person-id label map for the retrieval task
    person_label_map = ToothDataset.build_label_map(args.manifest, "person_id")

    dataset = ToothDataset(
        manifest_path=args.manifest,
        split=args.split,
        crop_mode=cfg["data"]["crop_mode"],
        target_col="person_id",
        transform=get_val_transforms(),
        label_map=person_label_map,
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    print(f"Evaluating {args.split}: {len(dataset)} samples, {len(person_label_map)} persons")

    # Extract features and matching person labels
    print("Extracting classifier features...")
    features = extract_classifier_features(model, loader, device)
    df = pd.read_csv(args.manifest, dtype=str)
    df_split = df[df["split"] == args.split].reset_index(drop=True)
    assert len(df_split) == len(features)
    labels = np.array([person_label_map[p] for p in df_split["person_id"]])

    # Drop any NaN/Inf rows (rare MPS edge cases)
    finite_mask = np.all(np.isfinite(features), axis=1)
    if not finite_mask.all():
        n_bad = (~finite_mask).sum()
        print(f"Dropping {n_bad} non-finite feature rows")
        features = features[finite_mask]
        labels = labels[finite_mask]

    print(f"Feature shape: {features.shape}")

    # Single-tooth verification + retrieval
    print("\n=== Single-tooth verification + retrieval ===")
    ver = evaluate_verification(features, labels)
    ret = evaluate_retrieval(features, labels)
    print(f"AUC: {ver['auc']:.4f}, EER: {ver['eer']:.4f}")
    print(f"Rank-1: {ret['rank1_micro']:.4f}, Rank-5: {ret['rank5']:.4f}, mAP: {ret['mAP']:.4f}")

    # Multi-tooth aggregation
    rng = np.random.RandomState(args.seed)
    print("\n=== Multi-tooth aggregation (mean) ===")
    sweep = []
    for n_q in [1, 2, 4, 8, 16]:
        res = evaluate_multi_tooth(features, labels, n_q, args.n_trials, "mean", rng)
        if res is None:
            continue
        print(f"  n_query={n_q}: R1={res['rank1_mean']:.4f}±{res['rank1_std']:.4f}, "
              f"R5={res['rank5_mean']:.4f}, mAP={res['mAP_mean']:.4f}")
        sweep.append(res)

    print("\n=== Forensic: 1 query tooth vs mean-aggregated gallery ===")
    forensic = evaluate_single_tooth_vs_aggregated_gallery(features, labels, args.n_trials, "mean", rng)
    if forensic is not None:
        print(f"  R1={forensic['rank1_mean']:.4f}±{forensic['rank1_std']:.4f}, "
              f"R5={forensic['rank5_mean']:.4f}, mAP={forensic['mAP_mean']:.4f}")

    # Output
    if args.output_dir is None:
        output_dir = Path(args.checkpoint).parent / "analysis" / "as_embedding"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "checkpoint": args.checkpoint,
        "feature_dim": int(features.shape[1]),
        "verification": {k: v for k, v in ver.items() if not k.startswith("_")},
        "retrieval_single_tooth": {k: v for k, v in ret.items() if not k.startswith("_")},
        "multi_tooth_sweep": sweep,
        "forensic_1tooth": forensic,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
