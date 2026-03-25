"""
Extract individual tooth crops from dataset_raw/ using mask files.

For each mask file ({image_id}+{FDI}.png), extracts the red region,
computes its bounding box, and saves:
  - raw crop (rectangular bbox with padding from original image)
  - masked crop (background zeroed outside the tooth contour)

Output structure:
  identification/data/crops_gt/{image_id}/{FDI}_raw.png
  identification/data/crops_gt/{image_id}/{FDI}_masked.png
"""

import argparse
import os
import sys
import re
import cv2
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

# Pre-compute HSV bounds as constants
_LOWER_RED1 = np.array([0, 70, 50])
_UPPER_RED1 = np.array([10, 255, 255])
_LOWER_RED2 = np.array([170, 70, 50])
_UPPER_RED2 = np.array([180, 255, 255])


def detect_red_mask(mask_bgr):
    """Extract binary mask of red regions using HSV thresholding."""
    hsv = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, _LOWER_RED1, _UPPER_RED1)
    mask2 = cv2.inRange(hsv, _LOWER_RED2, _UPPER_RED2)
    return cv2.bitwise_or(mask1, mask2)


def resize_with_padding(image, target_size=224):
    """Resize image preserving aspect ratio, pad with black to target_size x target_size."""
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    y_off = (target_size - new_h) // 2
    x_off = (target_size - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def process_folder(folder_path, output_dir, target_size=224, padding_ratio=0.1):
    """Process one dataset_raw folder. Returns (n_ok, n_err, error_messages)."""
    folder_path = Path(folder_path)
    folder_name = folder_path.name

    main_image_path = folder_path / f"{folder_name}.png"
    if not main_image_path.exists():
        return 0, 0, [f"No main image: {main_image_path}"]

    mask_files = sorted(folder_path.glob(f"{folder_name}+*.png"))
    if not mask_files:
        return 0, 0, [f"No mask files in {folder_path}"]

    main_img = cv2.imread(str(main_image_path))
    if main_img is None:
        return 0, 0, [f"Failed to read: {main_image_path}"]

    img_h, img_w = main_img.shape[:2]
    out_dir = Path(output_dir) / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    n_ok, n_err = 0, 0
    errors = []
    png_params = [cv2.IMWRITE_PNG_COMPRESSION, 1]

    # Track which FDI values already have a clean +FDI.png file
    # so we skip # duplicates when the normal version exists
    clean_fdis = set()
    for mp in mask_files:
        m = re.search(r'\+(\d+)\.png$', mp.name)
        if m:
            clean_fdis.add(m.group(1))

    for mask_path in mask_files:
        # Match clean +FDI.png, or +FDI #N.png / +FDI .png (CVAT artifacts)
        match = re.search(r'\+(\d+)(?:\s*(?:#\d+)?\s*)\.png$', mask_path.name)
        if not match:
            n_err += 1
            errors.append(f"{folder_name}: Bad mask filename: {mask_path.name}")
            continue
        fdi = match.group(1)

        # Skip # duplicate if we already have (or will process) the clean version
        is_hash_variant = '#' in mask_path.name or re.search(r'\+\d+\s+\.png$', mask_path.name)
        if is_hash_variant and fdi in clean_fdis:
            continue  # clean version takes priority
        # If output already exists from a previous # variant of same FDI, skip
        if (out_dir / f"{fdi}_raw.png").exists():
            continue

        mask_bgr = cv2.imread(str(mask_path))
        if mask_bgr is None:
            n_err += 1
            errors.append(f"{folder_name} FDI={fdi}: Failed to read mask")
            continue

        binary_mask = detect_red_mask(mask_bgr)
        coords = cv2.findNonZero(binary_mask)
        if coords is None:
            n_err += 1
            errors.append(f"{folder_name} FDI={fdi}: No red region found")
            continue

        x, y, w, h = cv2.boundingRect(coords)
        if w < 5 or h < 5:
            n_err += 1
            errors.append(f"{folder_name} FDI={fdi}: Crop too small: {w}x{h}")
            continue

        # Add padding
        pad_x = int(w * padding_ratio)
        pad_y = int(h * padding_ratio)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img_w, x + w + pad_x)
        y2 = min(img_h, y + h + pad_y)

        raw_crop = main_img[y1:y2, x1:x2]
        mask_region = binary_mask[y1:y2, x1:x2]
        masked_crop = raw_crop.copy()
        masked_crop[mask_region == 0] = 0

        raw_resized = resize_with_padding(raw_crop, target_size)
        masked_resized = resize_with_padding(masked_crop, target_size)

        cv2.imwrite(str(out_dir / f"{fdi}_raw.png"), raw_resized, png_params)
        cv2.imwrite(str(out_dir / f"{fdi}_masked.png"), masked_resized, png_params)
        n_ok += 1

    return n_ok, n_err, errors


def _process_wrapper(args):
    """Wrapper for multiprocessing."""
    folder_path, output_dir, target_size, padding_ratio = args
    return process_folder(folder_path, output_dir, target_size, padding_ratio)


def main():
    parser = argparse.ArgumentParser(description="Extract tooth crops from dataset_raw")
    parser.add_argument("--input-dir", default="dataset_raw", help="Path to dataset_raw/")
    parser.add_argument("--output-dir", default="identification/data/crops_gt", help="Output directory")
    parser.add_argument("--target-size", type=int, default=224, help="Output crop size (square)")
    parser.add_argument("--padding-ratio", type=float, default=0.1, help="Bbox padding ratio")
    parser.add_argument("--workers", type=int, default=0, help="Number of workers (0=auto)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    folders = sorted([f for f in input_dir.iterdir() if f.is_dir()])
    print(f"Found {len(folders)} folders in {input_dir}")

    total_crops = 0
    total_errors = 0
    error_log = []

    # Skip already-processed folders
    output_dir = Path(args.output_dir)
    existing = set(d.name for d in output_dir.iterdir() if d.is_dir()) if output_dir.exists() else set()
    to_process = [f for f in folders if f.name not in existing]
    print(f"Already processed: {len(existing)}, remaining: {len(to_process)}")

    for i, folder in enumerate(to_process):
        n_ok, n_err, errs = process_folder(folder, args.output_dir, args.target_size, args.padding_ratio)
        total_crops += n_ok
        total_errors += n_err
        error_log.extend(errs)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(to_process)} folders, {total_crops} crops", flush=True)

    print(f"\nDone. Extracted {total_crops} crops, {total_errors} errors.")
    if error_log:
        log_path = Path(args.output_dir) / "extraction_errors.log"
        with open(log_path, "w") as f:
            f.write("\n".join(error_log))
        print(f"Error log: {log_path}")


if __name__ == "__main__":
    main()
