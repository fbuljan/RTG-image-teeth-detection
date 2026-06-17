"""Evaluate an FDI classifier checkpoint on the val split of a given manifest.

The rotnorm pass criterion requires the rotnorm FDI classifier val acc to be
within 1pp of the deployed FDI classifier (baseline ~95%). Run this once for
the rotnorm classifier on the rotnorm manifest, and emit a JSON the rotnorm
evaluator can cross-check before declaring the rotnorm run a pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from identification.data.tooth_dataset import ToothDataset, get_val_transforms
from identification.models.classifier import ToothClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def evaluate(checkpoint_path: Path, manifest_path: Path, split: str = "val") -> dict:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    label_map = ckpt["label_map"]
    cfg = ckpt["config"]
    model = ToothClassifier(
        num_classes=len(label_map),
        pretrained=False,
        dropout=cfg["model"].get("dropout", 0.2),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    inv = {v: k for k, v in label_map.items()}
    ds = ToothDataset(
        manifest_path=str(manifest_path),
        split=split,
        crop_mode=cfg["data"].get("crop_mode", "raw"),
        target_col=cfg["data"].get("target_col", "tooth_fdi"),
        label_map=label_map,
        transform=get_val_transforms(),
    )
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
    n_correct = 0
    n_total = 0
    per_class_correct: dict[int, int] = {}
    per_class_total: dict[int, int] = {}
    with torch.no_grad():
        for batch in loader:
            imgs, labels = batch[0], batch[1]
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1)
            for p, l in zip(preds.cpu().numpy(), labels.cpu().numpy()):
                per_class_total[int(l)] = per_class_total.get(int(l), 0) + 1
                if int(p) == int(l):
                    per_class_correct[int(l)] = per_class_correct.get(int(l), 0) + 1
                    n_correct += 1
                n_total += 1
    acc = n_correct / max(1, n_total)
    per_class = {
        inv[k]: {"n": per_class_total[k], "acc": per_class_correct.get(k, 0) / per_class_total[k]}
        for k in sorted(per_class_total.keys())
    }
    return {
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "split": split,
        "n_total": n_total,
        "n_correct": n_correct,
        "val_accuracy": acc,
        "per_class": per_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--baseline-checkpoint", default=None,
                        help="Optional reference checkpoint to compare against (uses ITS manifest).")
    parser.add_argument("--baseline-manifest", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    ckpt = (PROJECT_ROOT / args.checkpoint).resolve()
    manifest = (PROJECT_ROOT / args.manifest).resolve()
    result = evaluate(ckpt, manifest, split=args.split)
    print(f"Val accuracy: {result['val_accuracy']:.4f} on {result['n_total']} crops")

    baseline_result = None
    if args.baseline_checkpoint:
        bp = (PROJECT_ROOT / args.baseline_checkpoint).resolve()
        bm = (PROJECT_ROOT / (args.baseline_manifest or args.manifest)).resolve()
        print(f"Baseline: evaluating {bp} on {bm}...")
        baseline_result = evaluate(bp, bm, split=args.split)
        delta = result["val_accuracy"] - baseline_result["val_accuracy"]
        print(f"Baseline val accuracy: {baseline_result['val_accuracy']:.4f}")
        print(f"Delta: {delta:+.4f}  (pre-registered threshold: within ±0.01)")
        result["baseline"] = baseline_result
        result["delta_vs_baseline"] = delta
        result["passes_rotnorm_criterion"] = bool(abs(delta) <= 0.01)

    if args.output:
        out = (PROJECT_ROOT / args.output).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved → {out}")


if __name__ == "__main__":
    main()
