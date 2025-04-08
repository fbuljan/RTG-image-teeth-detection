import os
import cv2
import glob
import numpy as np

IMG_DIR = "datasets/images/train"
LBL_DIR = "datasets/labels/train"
OUT_DIR = "label_previews"

os.makedirs(OUT_DIR, exist_ok=True)

# Koliko slika želiš pregledati
MAX_IMAGES = 10

# Učitaj do MAX_IMAGES labeliranih slika
image_paths = glob.glob(os.path.join(IMG_DIR, "*.png"))[:MAX_IMAGES]

for img_path in image_paths:
    filename = os.path.basename(img_path)
    name, _ = os.path.splitext(filename)
    label_path = os.path.join(LBL_DIR, name + ".txt")

    if not os.path.exists(label_path):
        print(f"⚠️ Label file not found: {label_path}")
        continue

    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:5])
            segmentation = list(map(float, parts[5:]))

            # Denormaliziraj bounding box
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Nacrtaj segmentaciju ako postoji
            if len(segmentation) >= 6:
                points = np.array(segmentation, dtype=np.float32).reshape(-1, 2)
                points[:, 0] *= w
                points[:, 1] *= h
                points = points.astype(np.int32)
                cv2.polylines(img, [points], isClosed=True, color=(0, 0, 255), thickness=2)

    out_path = os.path.join(OUT_DIR, filename)
    cv2.imwrite(out_path, img)
    print(f"✅ Spremljena anotacija: {out_path}")

print("\n✅ Gotovo! Otvori 'label_previews/' i pregledaj rezultate.")
