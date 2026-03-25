"""
Parse and merge metadata from ground_truth.csv and annotations.xml.

- ground_truth.csv: image_name, age, sex, age_group (tab-delimited, 1200 rows)
- annotations.xml: per-tooth eruption/root status, quad, index (600 images)
"""

import csv
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_ground_truth(csv_path="ground_truth.csv"):
    """Parse ground_truth.csv. Returns dict keyed by image_name (without .png)."""
    metadata = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            image_name = row["image_name"].replace(".png", "")
            metadata[image_name] = {
                "age": float(row["age"]),
                "sex": row["sex"],
                "age_group": row["age_group"],
            }
    return metadata


def parse_annotations_xml(xml_path="annotations.xml"):
    """
    Parse annotations.xml. Returns dict keyed by image_name (without .png),
    where each value is a dict keyed by FDI string → tooth metadata.

    FDI is computed as quad*10 + index (standard FDI notation).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    annotations = {}
    for image_elem in root.findall("image"):
        image_name = image_elem.get("name").replace(".png", "")
        teeth = {}

        for box in image_elem.findall("box"):
            attrs = {}
            for attr in box.findall("attribute"):
                attrs[attr.get("name")] = attr.text

            quad = int(attrs.get("quad", 0))
            index = int(attrs.get("index", 0))
            if quad == 0 or index == 0:
                continue

            fdi = str(quad * 10 + index)

            erupted = attrs.get("iznikli zub", "false").lower() == "true"
            unerupted = attrs.get("ne iznikli zub", "false").lower() == "true"
            root_complete = attrs.get("zavrsen rast korijena", "false").lower() == "true"
            root_incomplete = attrs.get("nezavrsen rast korijena", "false").lower() == "true"

            teeth[fdi] = {
                "erupted": erupted,
                "unerupted": unerupted,
                "root_complete": root_complete,
                "root_incomplete": root_incomplete,
                "quad": quad,
                "index": index,
            }

        annotations[image_name] = teeth

    return annotations


def parse_splits(splits_dir="splits"):
    """Parse splits/*.txt. Returns dict: image_name (without .png) → split name."""
    split_map = {}
    for split_name in ["train", "val", "test"]:
        split_file = Path(splits_dir) / f"{split_name}.txt"
        if not split_file.exists():
            continue
        with open(split_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Format: dataset_raw\{image_id}\{image_id}.png (Windows backslashes)
                # Extract image_id from the path
                parts = line.replace("\\", "/").split("/")
                # parts: ['dataset_raw', '{image_id}', '{image_id}.png']
                if len(parts) >= 2:
                    image_name = parts[1]  # folder name = image_id
                    split_map[image_name] = split_name
    return split_map


def fdi_to_info(fdi_str):
    """Derive quadrant, jaw, and deciduous flag from FDI tooth number."""
    fdi = int(fdi_str)
    quad = fdi // 10
    tooth_num = fdi % 10
    is_deciduous = quad >= 5
    jaw = "upper" if quad in (1, 2, 5, 6) else "lower"
    return {
        "quadrant": quad,
        "tooth_num": tooth_num,
        "is_deciduous": is_deciduous,
        "jaw": jaw,
    }
