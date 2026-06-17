"""
Build identification/data/pair_table_train.csv for the GT/YOLO blend training run.

For each (image_id, tooth_fdi) train pair that exists in both manifest_yolo.csv
and manifest_clean.csv, compute the YOLO predicted-polygon vs GT red-mask IoU
at panoramic resolution. A row is `accept=True` when both:
  - mask_iou >= 0.5   (geometric agreement; same physical tooth)
  - fdi_confidence >= 0.5 (classifier above the cliff in the disagreement tail)

This eliminates the adversarial-review "adjacent-tooth FDI swap" failure mode
that pure FDI-agreement gating cannot rule out.

Usage:
    python -m identification.scripts.build_pair_table \\
        --output identification/data/pair_table_train.csv
"""

from __future__ import annotations

import argparse
import functools
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

from backend.pipeline import _expand_bbox
from identification.models.classifier import ToothClassifier
from identification.scripts.extract_crops_gt import detect_red_mask

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_YOLO_SEG = PROJECT_ROOT / "runs-segmentation/default-seg/weights/best.pt"
DEFAULT_FDI_CLASSIFIER = PROJECT_ROOT / "identification/runs/tooth_fdi_raw/best.pt"
DEFAULT_GT_MANIFEST = PROJECT_ROOT / "identification/data/manifest_clean.csv"
DEFAULT_YOLO_MANIFEST = PROJECT_ROOT / "identification/data/manifest_yolo.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "identification/data/pair_table_train.csv"
DEFAULT_RAW_ROOT = PROJECT_ROOT / "dataset_raw"

print = functools.partial(print, flush=True)


def load_fdi_classifier(path: Path, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    label_map = ckpt["label_map"]
    cfg = ckpt["config"]
    model = ToothClassifier(
        num_classes=len(label_map),
        pretrained=False,
        dropout=cfg["model"].get("dropout", 0.2),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    label_inv = {v: k for k, v in label_map.items()}
    return model, label_inv


def polygon_to_binary(polygon: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Rasterise a polygon (Nx2 float) into a binary HxW mask."""
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], color=1)
    return mask


def iou_binary(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU of two binary masks. Auto-aligns shape mismatches by cropping
    to the common (min) shape — handles ±1px CVAT mask vs panoramic rounding."""
    if a.shape != b.shape:
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        a = a[:h, :w]
        b = b[:h, :w]
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def read_gt_mask_for_fdi(raw_root: Path, image_id: str, fdi: str) -> np.ndarray | None:
    """Load the GT red-overlay mask for one (image_id, fdi). Returns binary HxW or None."""
    candidates = [
        raw_root / image_id / f"{image_id}+{fdi}.png",
    ]
    # Allow CVAT hash-variant fallback like "+11 #1.png"
    fallback = sorted((raw_root / image_id).glob(f"{image_id}+{fdi}*.png"))
    candidates.extend(p for p in fallback if p not in candidates)
    for path in candidates:
        if path.exists():
            bgr = cv2.imread(str(path))
            if bgr is None:
                continue
            return detect_red_mask(bgr).astype(bool)
    return None


def process_image(
    image_id: str,
    pano_path: Path,
    gt_train_pairs: list[tuple[str, str, str]],
    yolo_model: YOLO,
    fdi_classifier: ToothClassifier,
    fdi_label_inv: dict[int, str],
    raw_root: Path,
    device: str,
    yolo_conf: float = 0.25,
    yolo_iou: float = 0.45,
    yolo_imgsz: int = 640,
    crop_size: int = 224,
) -> list[dict]:
    """For one panoramic, return rows: one per GT (image_id, fdi) train pair."""
    bgr = cv2.imread(str(pano_path))
    if bgr is None:
        return []
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    results = yolo_model.predict(
        source=str(pano_path),
        conf=yolo_conf,
        iou=yolo_iou,
        imgsz=yolo_imgsz,
        verbose=False,
        device=device,
    )
    if not results or results[0].masks is None or results[0].masks.xy is None:
        return _missing_rows(image_id, gt_train_pairs, reason="no_yolo_masks")

    polygons = [np.asarray(p) for p in results[0].masks.xy]
    if not polygons:
        return _missing_rows(image_id, gt_train_pairs, reason="no_yolo_polygons")

    # FDI classify each YOLO polygon (matches extract_crops_yolo.py)
    pil_image = Image.fromarray(rgb)
    raw_crops = []
    bboxes = []
    for poly in polygons:
        xs, ys = poly[:, 0], poly[:, 1]
        tight = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
        bbox_padded = _expand_bbox(tight, (w, h), padding_ratio=0.1)
        bboxes.append(bbox_padded)
        raw_crops.append(pil_image.crop(bbox_padded))

    # Inline a minimal _to_tensor — extract_crops_yolo's version pads + normalises
    from backend.pipeline import _to_tensor
    with torch.no_grad():
        logits_list = []
        for crop in raw_crops:
            t = _to_tensor(crop, crop_size, device)
            logits_list.append(fdi_classifier(t))
        logits = torch.cat(logits_list, dim=0)
        probs = F.softmax(logits, dim=1).cpu().numpy()
    fdi_class_idx = probs.argmax(axis=1)
    fdi_conf = probs.max(axis=1)
    fdi_labels = [fdi_label_inv[idx] for idx in fdi_class_idx]

    # De-duplicate same-FDI YOLO predictions, keep highest-confidence — matches
    # extract_crops_yolo.py logic so YOLO crops on disk correspond.
    seen: dict[str, int] = {}
    keep_indices: list[int] = []
    for i, fdi in enumerate(fdi_labels):
        if fdi in seen and fdi_conf[i] <= fdi_conf[seen[fdi]]:
            continue
        if fdi in seen:
            keep_indices.remove(seen[fdi])
        seen[fdi] = i
        keep_indices.append(i)

    # Build YOLO-FDI -> polygon-index map (after dedup)
    yolo_by_fdi: dict[str, int] = {fdi_labels[i]: i for i in keep_indices}

    # Rasterise YOLO polygons we'll need (only those matching a GT pair)
    needed_fdis = {fdi for _, fdi, _ in gt_train_pairs}
    yolo_masks: dict[str, np.ndarray] = {}
    for fdi in needed_fdis:
        if fdi in yolo_by_fdi:
            yolo_masks[fdi] = polygon_to_binary(polygons[yolo_by_fdi[fdi]], (h, w)).astype(bool)

    rows = []
    for gt_crop_path, fdi, yolo_crop_path in gt_train_pairs:
        if fdi not in yolo_by_fdi:
            rows.append({
                "image_id": image_id, "tooth_fdi": fdi,
                "gt_crop_path": gt_crop_path, "yolo_crop_path": yolo_crop_path,
                "mask_iou": 0.0, "fdi_confidence": 0.0, "accept": False,
                "reason": "yolo_missing_fdi",
            })
            continue
        idx_y = yolo_by_fdi[fdi]
        gt_mask = read_gt_mask_for_fdi(raw_root, image_id, fdi)
        if gt_mask is None:
            rows.append({
                "image_id": image_id, "tooth_fdi": fdi,
                "gt_crop_path": gt_crop_path, "yolo_crop_path": yolo_crop_path,
                "mask_iou": 0.0, "fdi_confidence": float(fdi_conf[idx_y]), "accept": False,
                "reason": "gt_mask_missing",
            })
            continue
        iou = iou_binary(yolo_masks[fdi], gt_mask)
        conf = float(fdi_conf[idx_y])
        accept = (iou >= 0.5) and (conf >= 0.5)
        rows.append({
            "image_id": image_id, "tooth_fdi": fdi,
            "gt_crop_path": gt_crop_path, "yolo_crop_path": yolo_crop_path,
            "mask_iou": float(iou), "fdi_confidence": conf, "accept": accept,
            "reason": "ok" if accept else f"iou<0.5({iou:.2f})" if iou < 0.5 else f"conf<0.5({conf:.2f})",
        })
    return rows


def _missing_rows(image_id, gt_train_pairs, reason):
    return [{
        "image_id": image_id, "tooth_fdi": fdi,
        "gt_crop_path": gt, "yolo_crop_path": yolo,
        "mask_iou": 0.0, "fdi_confidence": 0.0, "accept": False, "reason": reason,
    } for gt, fdi, yolo in gt_train_pairs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo-weights", default=str(DEFAULT_YOLO_SEG))
    parser.add_argument("--fdi-classifier", default=str(DEFAULT_FDI_CLASSIFIER))
    parser.add_argument("--gt-manifest", default=str(DEFAULT_GT_MANIFEST))
    parser.add_argument("--yolo-manifest", default=str(DEFAULT_YOLO_MANIFEST))
    parser.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--limit", type=int, default=None, help="Process first N images (smoke).")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading models...")
    yolo_model = YOLO(args.yolo_weights)
    fdi_classifier, fdi_label_inv = load_fdi_classifier(Path(args.fdi_classifier), device)

    gt_df = pd.read_csv(args.gt_manifest, dtype=str)
    yolo_df = pd.read_csv(args.yolo_manifest, dtype=str)
    gt_train = gt_df[gt_df["split"] == "train"]
    yolo_train = yolo_df[yolo_df["split"] == "train"]

    # Inner-join train rows on (image_id, tooth_fdi)
    paired = gt_train.merge(
        yolo_train[["image_id", "tooth_fdi", "crop_path"]].rename(columns={"crop_path": "yolo_crop_path"}),
        on=["image_id", "tooth_fdi"], how="inner",
    )
    print(f"GT train rows: {len(gt_train)}")
    print(f"YOLO train rows: {len(yolo_train)}")
    print(f"FDI-agree paired rows: {len(paired)}")

    # Group by image
    grouped = paired.groupby("image_id")
    image_ids = sorted(grouped.groups.keys())
    if args.limit:
        image_ids = image_ids[: args.limit]
    print(f"Images to process: {len(image_ids)}")

    raw_root = Path(args.raw_root)
    all_rows = []
    t0 = time.perf_counter()
    for i, image_id in enumerate(image_ids):
        pano = raw_root / image_id / f"{image_id}.png"
        if not pano.exists():
            continue
        sub = grouped.get_group(image_id)
        gt_pairs = list(zip(sub["crop_path"], sub["tooth_fdi"], sub["yolo_crop_path"]))
        rows = process_image(
            image_id, pano, gt_pairs,
            yolo_model, fdi_classifier, fdi_label_inv, raw_root, device,
        )
        all_rows.extend(rows)
        if (i + 1) % 25 == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            eta = (len(image_ids) - i - 1) / rate
            print(f"  [{i+1}/{len(image_ids)}] {rate:.1f} img/s, eta {eta/60:.1f} min")
        if device == "mps" and (i + 1) % 25 == 0:
            torch.mps.empty_cache()

    df = pd.DataFrame(all_rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(df)} rows)")

    # Summary
    print("\n=== Pair-table summary ===")
    print(f"  total rows         : {len(df)}")
    print(f"  accept=True        : {df['accept'].sum()} ({df['accept'].mean()*100:.1f}%)")
    print(f"  accept=False       : {(~df['accept']).sum()} ({(~df['accept']).mean()*100:.1f}%)")
    print(f"\n  rejection reasons (top 5):")
    for reason, n in df[~df["accept"]]["reason"].value_counts().head(5).items():
        print(f"    {reason}: {n}")
    print(f"\n  IoU distribution (accept=True only):")
    accepted = df[df["accept"]]
    if len(accepted):
        print(f"    mean   : {accepted['mask_iou'].mean():.3f}")
        print(f"    median : {accepted['mask_iou'].median():.3f}")
        print(f"    p25    : {accepted['mask_iou'].quantile(0.25):.3f}")
        print(f"    p10    : {accepted['mask_iou'].quantile(0.10):.3f}")


if __name__ == "__main__":
    main()
