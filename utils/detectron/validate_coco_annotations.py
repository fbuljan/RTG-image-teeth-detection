import os
import random
import json
import cv2
import numpy as np
from tqdm import tqdm

# Putanje
COCO_ANNOTATION_FILE = "coco_annotations.json"
IMAGE_ROOT = "dataset_raw"
OUTPUT_DIR = "visual_validation"
NUM_SAMPLES = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Učitavanje COCO anotacija
with open(COCO_ANNOTATION_FILE, "r") as f:
    coco_data = json.load(f)

# Mapiranja ID <-> info
id_to_filename = {img["id"]: img["file_name"] for img in coco_data["images"]}
id_to_annotations = {}
for ann in coco_data["annotations"]:
    id_to_annotations.setdefault(ann["image_id"], []).append(ann)

# Odaberi slučajnih 10 slika
sampled_ids = random.sample(list(id_to_filename.keys()), NUM_SAMPLES)

# Statistika
total_instances = 0
total_points = 0
image_stats = []

for img_id in tqdm(sampled_ids, desc="Validating"):
    filename = id_to_filename[img_id]                    # npr. "image_001.png"
    name_wo_ext = os.path.splitext(filename)[0]          # npr. "image_001"
    image_path = os.path.join(IMAGE_ROOT, name_wo_ext, filename)
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Nije pronađena slika: {image_path}")
        continue

    anns = id_to_annotations.get(img_id, [])
    inst_count = len(anns)
    total_instances += inst_count

    for ann in anns:
        segm = ann["segmentation"]
        if not segm:
            continue
        pts = np.array(segm[0], dtype=np.float32).reshape(-1, 2)
        total_points += len(pts)

        pts_int = pts.astype(int)
        cv2.polylines(image, [pts_int], isClosed=True, color=(0, 255, 0), thickness=2)

        # Bounding box (opcionalno)
        x, y, w, h = map(int, ann["bbox"])
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 1)

    # Spremi rezultat
    save_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(save_path, image)

    image_stats.append({
        "file": filename,
        "instances": inst_count,
    })

# Summary
avg_instances = total_instances / NUM_SAMPLES
avg_points = total_points / max(total_instances, 1)

print("\n📊 VALIDACIJA:")
print(f"- Prosječan broj instanci po slici: {avg_instances:.2f}")
print(f"- Prosječan broj točaka po maski: {avg_points:.2f}")
print(f"- Rezultati spremljeni u: {OUTPUT_DIR}/")

print("\n📋 Statistika po slici:")
for stat in image_stats:
    print(f"  {stat['file']}: {stat['instances']} maski")
