import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from datetime import datetime

# Postavke
input_root = "dataset_raw"
output_json = "coco_annotations.json"

# Klasifikacijska kategorija (samo zubi)
categories = [{"id": 1, "name": "tooth"}]

images = []
annotations = []
ann_id = 1
img_id = 1

for folder_name in tqdm(os.listdir(input_root)):
    folder_path = os.path.join(input_root, folder_name)
    if not os.path.isdir(folder_path):
        continue

    image_name = folder_name + ".png"
    image_path = os.path.join(folder_path, image_name)

    if not os.path.exists(image_path):
        continue

    image = cv2.imread(image_path)
    if image is None:
        continue

    height, width = image.shape[:2]

    images.append({
        "id": img_id,
        "file_name": image_name,
        "width": width,
        "height": height
    })

    for file in os.listdir(folder_path):
        if file.startswith(folder_name + "+") and file.endswith(".png"):
            mask_path = os.path.join(folder_path, file)

            mask_bgr = cv2.imread(mask_path)
            if mask_bgr is None:
                continue

            hsv = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2HSV)

            lower_red1 = np.array([0, 70, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 70, 50])
            upper_red2 = np.array([180, 255, 255])

            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(mask1, mask2)

            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 10:
                    continue

                segmentation = cnt.flatten().tolist()
                if len(segmentation) < 6:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "segmentation": [segmentation],
                    "bbox": [x, y, w, h],
                    "area": float(area),
                    "iscrowd": 0
                })
                ann_id += 1

    img_id += 1

coco_format = {
    "info": {
        "description": "Tooth segmentation dataset for Detectron2",
        "version": "1.0",
        "date_created": datetime.now().isoformat()
    },
    "licenses": [],
    "images": images,
    "annotations": annotations,
    "categories": categories
}

with open(output_json, "w") as f:
    json.dump(coco_format, f, indent=2)

print("✅ COCO anotacije uspješno generirane:", output_json)
