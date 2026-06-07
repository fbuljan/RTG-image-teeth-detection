"""Extract tooth crops by running the deployed YOLO segmentation + FDI classifier.

Mirrors the runtime crop pipeline in backend/pipeline.py so the resulting
crops (and the registries built from them) match the distribution the demo
sees at inference. Used to fix the Phase 7.1 deployment distribution-shift
issue: rebuilding the ensemble registry from YOLO crops eliminates the
gap between Phase 7.1 eval results and live demo behaviour.

Output layout (mirrors crops_gt/):

    identification/data/crops_yolo/{image_id}/{FDI}_raw.png
    identification/data/crops_yolo/{image_id}/{FDI}_masked.png   # zeroed outside the mask polygon
    identification/data/manifest_yolo.csv                         # 1:1 schema with manifest_clean.csv

Usage:
    python -m identification.scripts.extract_crops_yolo
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

from backend.pipeline import _expand_bbox, _resize_with_padding, _to_tensor
from identification.models.classifier import ToothClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_YOLO_SEG = PROJECT_ROOT / "runs-segmentation/default-seg/weights/best.pt"
DEFAULT_FDI_CLASSIFIER = PROJECT_ROOT / "identification/runs/tooth_fdi_raw/best.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "identification/data/crops_yolo"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "identification/data/manifest_yolo.csv"
DEFAULT_MANIFEST_REF = PROJECT_ROOT / "identification/data/manifest_clean.csv"


def load_fdi_classifier(path: Path, device: str) -> tuple[ToothClassifier, dict[int, str]]:
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


def _mask_crop(panoramic: np.ndarray, polygon: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Crop the panoramic to bbox with everything outside the polygon zeroed."""
    x1, y1, x2, y2 = bbox
    mask_full = np.zeros(panoramic.shape[:2], dtype=np.uint8)
    poly_int = polygon.astype(np.int32)
    cv2.fillPoly(mask_full, [poly_int], color=1)
    region = panoramic[y1:y2, x1:x2].copy()
    region_mask = mask_full[y1:y2, x1:x2]
    region[region_mask == 0] = 0
    return region


def process_panoramic(
    image_id: str,
    panoramic_path: Path,
    yolo_model: YOLO,
    fdi_classifier: ToothClassifier,
    fdi_label_inv: dict[int, str],
    output_root: Path,
    device: str,
    yolo_conf: float = 0.25,
    yolo_iou: float = 0.45,
    yolo_imgsz: int = 640,
    crop_size: int = 224,
) -> list[dict]:
    """Run the YOLO seg + FDI classifier on one panoramic, save crops, return rows."""
    panoramic_bgr = cv2.imread(str(panoramic_path))
    if panoramic_bgr is None:
        return []
    panoramic_rgb = cv2.cvtColor(panoramic_bgr, cv2.COLOR_BGR2RGB)
    h, w = panoramic_rgb.shape[:2]

    results = yolo_model.predict(
        source=str(panoramic_path),
        conf=yolo_conf,
        iou=yolo_iou,
        imgsz=yolo_imgsz,
        verbose=False,
        device=device if device != "mps" else "mps",
    )
    if not results:
        return []
    res0 = results[0]
    if res0.masks is None or res0.masks.xy is None or len(res0.masks.xy) == 0:
        return []

    polygons = [np.asarray(p) for p in res0.masks.xy]

    # Predict FDI for each detected tooth
    pil_image = Image.fromarray(panoramic_rgb)
    raw_crops_pil: list[Image.Image] = []
    masked_crops_arr: list[np.ndarray] = []
    bboxes: list[tuple[int, int, int, int]] = []
    for poly in polygons:
        xs, ys = poly[:, 0], poly[:, 1]
        tight = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
        bbox_padded = _expand_bbox(tight, (w, h), padding_ratio=0.1)
        bboxes.append(bbox_padded)
        x1, y1, x2, y2 = bbox_padded
        raw_crops_pil.append(pil_image.crop((x1, y1, x2, y2)))
        masked_region = _mask_crop(panoramic_rgb, poly, bbox_padded)
        masked_crops_arr.append(masked_region)

    # FDI classification (vectorised across all crops)
    if not raw_crops_pil:
        return []
    with torch.no_grad():
        logits_list = []
        for crop in raw_crops_pil:
            t = _to_tensor(crop, crop_size, device)
            logits_list.append(fdi_classifier(t))
        logits = torch.cat(logits_list, dim=0)
        probs = F.softmax(logits, dim=1).cpu().numpy()
    fdi_class_idx = probs.argmax(axis=1)
    fdi_conf = probs.max(axis=1)
    fdi_labels = [fdi_label_inv[idx] for idx in fdi_class_idx]

    # De-duplicate same-FDI predictions, keep the higher-confidence one
    seen: dict[str, int] = {}
    keep_indices: list[int] = []
    for i, fdi in enumerate(fdi_labels):
        if fdi in seen and fdi_conf[i] <= fdi_conf[seen[fdi]]:
            continue
        if fdi in seen:
            keep_indices.remove(seen[fdi])
        seen[fdi] = i
        keep_indices.append(i)
    keep_indices.sort()

    # Save crops and build rows
    out_dir = output_root / image_id
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for i in keep_indices:
        fdi = fdi_labels[i]
        # Raw crop with aspect-preserving padding to 224x224
        raw_resized = _resize_with_padding(raw_crops_pil[i], crop_size)
        raw_path = out_dir / f"{fdi}_raw.png"
        raw_resized.save(raw_path)

        # Masked crop — go via PIL using the same _resize_with_padding util
        masked_pil = Image.fromarray(masked_crops_arr[i])
        masked_resized = _resize_with_padding(masked_pil, crop_size)
        masked_path = out_dir / f"{fdi}_masked.png"
        masked_resized.save(masked_path)

        rows.append({
            "image_id": image_id,
            "tooth_fdi": fdi,
            "fdi_confidence": float(fdi_conf[i]),
            "crop_path": str(raw_path.relative_to(PROJECT_ROOT)),
            "masked_crop_path": str(masked_path.relative_to(PROJECT_ROOT)),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo-weights", default=str(DEFAULT_YOLO_SEG))
    parser.add_argument("--fdi-classifier", default=str(DEFAULT_FDI_CLASSIFIER))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--manifest-out", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--manifest-ref", default=str(DEFAULT_MANIFEST_REF))
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N persons (smoke test).")
    args = parser.parse_args()

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading YOLO segmenter: {args.yolo_weights}")
    yolo_model = YOLO(args.yolo_weights)
    print(f"Loading FDI classifier: {args.fdi_classifier}")
    fdi_classifier, fdi_label_inv = load_fdi_classifier(Path(args.fdi_classifier), device)

    # Reference manifest: tells us which persons exist + their per-person metadata
    # (age, sex, split). FDI-level columns (erupted, root_complete) are dropped
    # because the FDI assignment may differ.
    ref_df = pd.read_csv(args.manifest_ref, dtype=str)
    person_meta_cols = ["person_id", "image_id", "age", "sex", "age_group", "split"]
    person_meta = (
        ref_df[person_meta_cols]
        .drop_duplicates("image_id")
        .set_index("image_id")
        .to_dict(orient="index")
    )
    image_ids = sorted(person_meta.keys())
    if args.limit:
        image_ids = image_ids[: args.limit]
    print(f"Persons to process: {len(image_ids)}")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    skipped = 0
    for image_id in tqdm(image_ids, desc="Extracting YOLO crops"):
        panoramic_path = PROJECT_ROOT / "dataset_raw" / image_id / f"{image_id}.png"
        if not panoramic_path.exists():
            skipped += 1
            continue
        per_tooth_rows = process_panoramic(
            image_id, panoramic_path, yolo_model, fdi_classifier, fdi_label_inv,
            output_root, device,
        )
        meta = person_meta[image_id]
        for r in per_tooth_rows:
            quadrant = int(r["tooth_fdi"][0]) if r["tooth_fdi"].isdigit() and len(r["tooth_fdi"]) >= 1 else None
            tooth_num = int(r["tooth_fdi"][1]) if r["tooth_fdi"].isdigit() and len(r["tooth_fdi"]) >= 2 else None
            jaw = "upper" if quadrant in (1, 2, 5, 6) else ("lower" if quadrant in (3, 4, 7, 8) else None)
            is_deciduous = quadrant in (5, 6, 7, 8)
            rows.append({
                **meta,
                **r,
                "quadrant": quadrant,
                "tooth_num": tooth_num,
                "jaw": jaw,
                "is_deciduous": is_deciduous,
                # erupted / root_complete come from the 600-image XML subset;
                # not joinable with predicted FDIs because the join key changed.
                "erupted": None,
                "root_complete": None,
            })

    manifest_df = pd.DataFrame(rows)
    # Reorder to roughly match manifest_clean.csv schema
    column_order = [
        "crop_path", "masked_crop_path",
        "person_id", "image_id", "tooth_fdi", "quadrant", "jaw", "tooth_num",
        "is_deciduous", "age", "sex", "age_group", "erupted", "root_complete",
        "split", "fdi_confidence",
    ]
    manifest_df = manifest_df[[c for c in column_order if c in manifest_df.columns]]
    manifest_df.to_csv(args.manifest_out, index=False)
    print(f"\nWrote {len(manifest_df)} crops across {manifest_df['image_id'].nunique()} persons")
    print(f"Manifest: {args.manifest_out}")
    print(f"Crops:    {args.output_dir}/")
    if skipped:
        print(f"Skipped (missing panoramic): {skipped}")


if __name__ == "__main__":
    main()
