import os
import json
from tqdm import tqdm

# === CONFIG ===
COCO_JSON_PATH = "coco_annotations.json"
IMAGES_DIR = "dataset_raw"  # adjust if needed

def main():
    with open(COCO_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check required fields
    required_keys = ["images", "annotations", "categories"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required field: {key}")

    image_id_map = {}
    missing_images = []
    bad_annotations = 0

    # Validate images
    print("🔍 Checking image files...")
    for img in tqdm(data["images"]):
        image_id = img.get("id")
        file_name = img.get("file_name")
        if not file_name or not image_id:
            raise ValueError(f"Invalid image entry: {img}")
        image_id_map[image_id] = file_name

        # --- FIXED PATH ---
        name_without_ext = os.path.splitext(file_name)[0]
        image_path = os.path.join(IMAGES_DIR, name_without_ext, file_name)
        if not os.path.exists(image_path):
            missing_images.append(file_name)

    # Validate annotations
    print("🔍 Checking annotations...")
    for ann in tqdm(data["annotations"]):
        if ann["image_id"] not in image_id_map:
            print(f"⚠️ Annotation references unknown image_id: {ann['image_id']}")
            bad_annotations += 1

        seg = ann.get("segmentation")
        if seg is None or (isinstance(seg, list) and len(seg) == 0):
            print(f"⚠️ Invalid segmentation in annotation: {ann.get('id')}")
            bad_annotations += 1

    # Summary
    print("\n✅ Integrity check complete.\n")
    print(f"Missing images: {len(missing_images)}")
    if missing_images:
        print("First few missing:", missing_images[:5])
    print(f"Annotations with issues: {bad_annotations}")

if __name__ == "__main__":
    main()
