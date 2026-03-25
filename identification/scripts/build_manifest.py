"""
Build the dataset manifest CSV from extracted crops and metadata sources.

Walks identification/data/crops_gt/ and creates a CSV combining:
- Crop file paths
- Person/image identity (from folder names)
- Tooth FDI info (from filenames)
- Demographics (from ground_truth.csv)
- Eruption/root status (from annotations.xml, available for ~600 images)
- Train/val/test split (from splits/*.txt)
"""

import argparse
import csv
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from identification.utils.metadata_parser import (
    parse_ground_truth,
    parse_annotations_xml,
    parse_splits,
    fdi_to_info,
)


def build_manifest(crops_dir, gt_csv, xml_path, splits_dir, output_path):
    """Build manifest CSV from crops directory and metadata sources."""
    crops_dir = Path(crops_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load metadata
    print("Loading metadata...")
    gt_meta = parse_ground_truth(gt_csv)
    xml_meta = parse_annotations_xml(xml_path)
    split_map = parse_splits(splits_dir)

    print(f"  ground_truth.csv: {len(gt_meta)} images")
    print(f"  annotations.xml: {len(xml_meta)} images")
    print(f"  splits: {len(split_map)} images")

    # Walk crops directory
    rows = []
    image_dirs = sorted([d for d in crops_dir.iterdir() if d.is_dir()])

    for image_dir in image_dirs:
        image_id = image_dir.name

        # Find all raw crop files
        raw_crops = sorted(image_dir.glob("*_raw.png"))
        for raw_path in raw_crops:
            fdi_str = raw_path.stem.replace("_raw", "")
            masked_path = image_dir / f"{fdi_str}_masked.png"

            # FDI-derived info
            fdi_info = fdi_to_info(fdi_str)

            # Ground truth metadata
            gt = gt_meta.get(image_id, {})

            # XML annotation metadata for this specific tooth
            xml_teeth = xml_meta.get(image_id, {})
            xml_tooth = xml_teeth.get(fdi_str, {})

            # Split assignment
            split = split_map.get(image_id, "unknown")

            rows.append({
                "crop_path": str(raw_path.relative_to(crops_dir.parent.parent.parent)),
                "masked_crop_path": str(masked_path.relative_to(crops_dir.parent.parent.parent)) if masked_path.exists() else "",
                "person_id": image_id,
                "image_id": image_id,
                "tooth_fdi": fdi_str,
                "quadrant": fdi_info["quadrant"],
                "jaw": fdi_info["jaw"],
                "tooth_num": fdi_info["tooth_num"],
                "is_deciduous": fdi_info["is_deciduous"],
                "age": gt.get("age", ""),
                "sex": gt.get("sex", ""),
                "age_group": gt.get("age_group", ""),
                "erupted": xml_tooth.get("erupted", ""),
                "root_complete": xml_tooth.get("root_complete", ""),
                "split": split,
            })

    # Write CSV
    fieldnames = [
        "crop_path", "masked_crop_path", "person_id", "image_id",
        "tooth_fdi", "quadrant", "jaw", "tooth_num", "is_deciduous",
        "age", "sex", "age_group", "erupted", "root_complete", "split",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nManifest written: {output_path}")
    print(f"  Total rows: {len(rows)}")
    print(f"  Unique persons: {len(set(r['person_id'] for r in rows))}")
    print(f"  Splits: { {s: sum(1 for r in rows if r['split'] == s) for s in ['train', 'val', 'test', 'unknown']} }")
    print(f"  With eruption data: {sum(1 for r in rows if r['erupted'] != '')}")


def main():
    parser = argparse.ArgumentParser(description="Build dataset manifest CSV")
    parser.add_argument("--crops-dir", default="identification/data/crops_gt")
    parser.add_argument("--gt-csv", default="ground_truth.csv")
    parser.add_argument("--xml", default="annotations.xml")
    parser.add_argument("--splits-dir", default="splits")
    parser.add_argument("--output", default="identification/data/manifest.csv")
    args = parser.parse_args()

    build_manifest(args.crops_dir, args.gt_csv, args.xml, args.splits_dir, args.output)


if __name__ == "__main__":
    main()
