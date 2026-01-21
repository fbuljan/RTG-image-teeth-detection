import os, cv2, glob, numpy as np

IMG_DIR = "datasets/images/train"
LBL_DIR = "datasets/labels/train"
OUT_DIR = "label_previews"
os.makedirs(OUT_DIR, exist_ok=True)

EXTS = ("*.png", "*.jpg", "*.jpeg")
image_paths = []
for ext in EXTS:
    image_paths += glob.glob(os.path.join(IMG_DIR, ext))
image_paths = sorted(image_paths)[:10]  # MAX_IMAGES = 10

def warn(msg): print(f"⚠️ {msg}")

for img_path in image_paths:
    filename = os.path.basename(img_path)
    stem, _ = os.path.splitext(filename)
    label_path = os.path.join(LBL_DIR, stem + ".txt")

    if not os.path.exists(label_path):
        warn(f"Label not found: {label_path}")
        continue

    img = cv2.imread(img_path)
    if img is None:
        warn(f"Failed to read image: {img_path}")
        continue

    h, w = img.shape[:2]
    overlay = img.copy()

    with open(label_path, "r") as f:
        for li, line in enumerate(f, start=1):
            parts = line.strip().split()
            # YOLO segmentation format: class_id x1 y1 x2 y2 ... xn yn
            # Minimum: class_id + 3 points (6 coords) = 7 values
            if len(parts) < 7:
                warn(f"{stem}.txt line {li}: too few values (need class + min 3 points)")
                continue

            try:
                class_id = int(parts[0])
            except Exception:
                warn(f"{stem}.txt line {li}: parse error for class_id")
                continue

            # Parse polygon coordinates (all values after class_id)
            coords = parts[1:]
            if len(coords) % 2 != 0:
                warn(f"{stem}.txt line {li}: odd number of coordinates")
                continue

            try:
                pts = np.array(list(map(float, coords)), dtype=np.float32).reshape(-1, 2)
            except Exception:
                warn(f"{stem}.txt line {li}: parse error for polygon")
                continue

            # range check before denorm
            if (pts < 0).any() or (pts > 1).any():
                warn(f"{stem}.txt line {li}: poly coords outside [0,1]")

            # denorm + clip
            pts[:, 0] = np.clip(pts[:, 0] * w, 0, w-1)
            pts[:, 1] = np.clip(pts[:, 1] * h, 0, h-1)
            pts = pts.astype(np.int32)

            # Calculate bounding box from polygon for visualization
            x_min, y_min = pts.min(axis=0)
            x_max, y_max = pts.max(axis=0)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
            cv2.putText(img, str(class_id), (x_min, max(0, y_min-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

            # draw outline + filled (alpha)
            cv2.polylines(img, [pts], isClosed=True, color=(0,0,255), thickness=2)
            cv2.fillPoly(overlay, [pts], color=(0,0,255))
    
    # alpha blend for masks
    out = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)
    out_path = os.path.join(OUT_DIR, filename)
    cv2.imwrite(out_path, out)
    print(f"✅ Spremljeno: {out_path}")

print("\n✅ Gotovo! Pogledaj 'label_previews/'")
