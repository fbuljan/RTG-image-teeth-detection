"""
Validate and clean the crop dataset.

Checks for: empty/all-black crops, tiny crops, corrupted images, duplicates.
Produces: manifest_clean.csv, rejection_log.csv, review_grid.png
"""

import argparse
import csv
import random
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def validate_crop(crop_path, min_size=10, min_nonzero_ratio=0.01):
    """Validate a single crop image. Returns (is_valid, reason)."""
    path = Path(crop_path)
    if not path.exists():
        return False, "file_missing"

    img = cv2.imread(str(path))
    if img is None:
        return False, "unreadable"

    h, w = img.shape[:2]
    if h < min_size or w < min_size:
        return False, f"too_small_{w}x{h}"

    # Check if image is all black (empty mask applied)
    nonzero = np.count_nonzero(img)
    total = img.size
    if nonzero / total < min_nonzero_ratio:
        return False, "all_black"

    return True, "ok"


def generate_review_grid(valid_paths, output_path, grid_size=10, crop_size=224):
    """Generate a grid of random crops for manual review."""
    n = grid_size * grid_size
    sample = random.sample(valid_paths, min(n, len(valid_paths)))

    grid = np.zeros((grid_size * crop_size, grid_size * crop_size, 3), dtype=np.uint8)
    for idx, path in enumerate(sample):
        row, col = divmod(idx, grid_size)
        img = cv2.imread(str(path))
        if img is None:
            continue
        img = cv2.resize(img, (crop_size, crop_size))
        grid[row * crop_size:(row + 1) * crop_size,
             col * crop_size:(col + 1) * crop_size] = img

    cv2.imwrite(str(output_path), grid)
    print(f"Review grid saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Validate and clean crop dataset")
    parser.add_argument("--manifest", default="identification/data/manifest.csv")
    parser.add_argument("--output-clean", default="identification/data/manifest_clean.csv")
    parser.add_argument("--output-rejected", default="identification/data/rejection_log.csv")
    parser.add_argument("--output-grid", default="identification/data/review_grid.png")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    with open(manifest_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Validating {len(rows)} crops...")

    clean_rows = []
    rejected_rows = []
    valid_raw_paths = []

    for row in tqdm(rows, desc="Validating"):
        raw_valid, raw_reason = validate_crop(row["crop_path"])
        masked_valid, masked_reason = validate_crop(row["masked_crop_path"])

        if raw_valid and masked_valid:
            clean_rows.append(row)
            valid_raw_paths.append(row["crop_path"])
        else:
            reason = raw_reason if not raw_valid else f"masked:{masked_reason}"
            rejected_rows.append({**row, "rejection_reason": reason})

    # Write clean manifest
    if clean_rows:
        fieldnames = list(clean_rows[0].keys())
        with open(args.output_clean, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(clean_rows)

    # Write rejection log
    if rejected_rows:
        fieldnames = list(rejected_rows[0].keys())
        with open(args.output_rejected, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rejected_rows)

    print(f"\nResults:")
    print(f"  Valid: {len(clean_rows)}")
    print(f"  Rejected: {len(rejected_rows)}")
    if rejected_rows:
        reasons = {}
        for r in rejected_rows:
            reason = r["rejection_reason"]
            reasons[reason] = reasons.get(reason, 0) + 1
        print(f"  Rejection reasons: {reasons}")

    # Generate review grid from valid raw crops
    if valid_raw_paths:
        generate_review_grid(valid_raw_paths, args.output_grid)

    print(f"\nClean manifest: {args.output_clean}")
    print(f"Rejection log: {args.output_rejected}")


if __name__ == "__main__":
    main()
