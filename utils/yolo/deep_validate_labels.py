import os
import cv2
import numpy as np
import csv
from tqdm import tqdm

LABELS_DIR = "datasets/labels/train"
IMAGES_DIR = "datasets/images/train"
OUTPUT_CSV = "label_validation_report.csv"
MAX_FILES = 100

def bbox_from_polygon(coords):
    """Izračunaj bounding box iz polygon koordinata (normalizirane)"""
    pts = np.array(coords).reshape(-1, 2)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    return (x_center, y_center, width, height)

def iou(box1, box2):
    """Izračun Intersection-over-Union između dva bounding boxa (center format)"""
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2

    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2

    xi1 = max(x1_min, x2_min)
    yi1 = max(y1_min, y2_min)
    xi2 = min(x1_max, x2_max)
    yi2 = min(y1_max, y2_max)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def validate_label(label_path, img_path):
    results = {
        "image": os.path.basename(img_path),
        "num_instances": 0,
        "num_duplicates": 0,
        "invalid_lines": 0
    }

    if not os.path.exists(img_path):
        results["error"] = "image not found"
        return results

    try:
        with open(label_path, "r") as f:
            lines = f.read().strip().splitlines()
    except:
        results["error"] = "could not read label"
        return results

    boxes = []
    for line in lines:
        parts = line.strip().split()
        # YOLO segmentation format: class_id x1 y1 x2 y2 ... xn yn
        # Minimum: class_id + 3 points (6 coords) = 7 values
        if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
            results["invalid_lines"] += 1
            continue

        try:
            class_id = int(parts[0])
            poly_coords = list(map(float, parts[1:]))
            if not all(0 <= c <= 1 for c in poly_coords):
                results["invalid_lines"] += 1
                continue
            # Calculate bbox from polygon for duplicate detection
            bbox = bbox_from_polygon(poly_coords)
            boxes.append(bbox)
        except:
            results["invalid_lines"] += 1
            continue

    results["num_instances"] = len(boxes)

    # Detekcija duplikata (IOU > 0.8)
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if iou(boxes[i], boxes[j]) > 0.8:
                results["num_duplicates"] += 1
                break

    return results

def main():
    output_dir = os.path.dirname(OUTPUT_CSV)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    label_files = sorted([f for f in os.listdir(LABELS_DIR) if f.endswith(".txt")])[:MAX_FILES]

    report = []
    for label_file in tqdm(label_files, desc="Validating"):
        image_file = os.path.splitext(label_file)[0] + ".png"
        label_path = os.path.join(LABELS_DIR, label_file)
        img_path = os.path.join(IMAGES_DIR, image_file)
        result = validate_label(label_path, img_path)
        report.append(result)

    # Spremi CSV
    with open(OUTPUT_CSV, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=report[0].keys())
        writer.writeheader()
        writer.writerows(report)

    # Sažetak
    total = len(report)
    invalid = sum(1 for r in report if r["invalid_lines"] > 0 or "error" in r)
    dupes = sum(1 for r in report if r["num_duplicates"] > 0)

    print(f"\n✅ Validirano {total} labela")
    print(f"⚠️  Labela s greškama u linijama: {invalid}")
    print(f"⚠️  Slika s potencijalnim duplikatima: {dupes}")
    print(f"📄 Izvještaj spremljen u: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
