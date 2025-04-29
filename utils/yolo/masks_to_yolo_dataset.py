import os
import cv2
import numpy as np
import shutil
import random

input_root = "dataset_raw"
output_root = "datasets"
temp_images = os.path.join(output_root, "temp_images")
temp_labels = os.path.join(output_root, "temp_labels")

os.makedirs(temp_images, exist_ok=True)
os.makedirs(temp_labels, exist_ok=True)

counter = 0
total = 1197

# 1. Prolazak kroz sve foldere
for folder_name in os.listdir(input_root):
    folder_path = os.path.join(input_root, folder_name)
    if not os.path.isdir(folder_path):
        continue

    image_name = folder_name + ".png"
    image_path = os.path.join(folder_path, image_name)

    if not os.path.exists(image_path):
        print(f"\u26a0\ufe0f Slika nije pronađena: {image_path}")
        continue

    image = cv2.imread(image_path)
    if image is None:
        print(f"\u26a0\ufe0f Ne mogu učitati sliku: {image_path}")
        continue

    h, w = image.shape[:2]
    img_out_path = os.path.join(temp_images, image_name)
    if not os.path.exists(img_out_path):
        cv2.imwrite(img_out_path, image)

    yolo_lines = []
    mask_count = 0

    # 2. Pronađi sve maske koje počinju s imenom + nesto
    for file in os.listdir(folder_path):
        if file.startswith(folder_name + "+") and file.endswith(".png"):
            mask_count += 1
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
                x, y, bw, bh = cv2.boundingRect(cnt)
                if bw < 3 or bh < 3:
                    continue

                x_center = (x + bw / 2) / w
                y_center = (y + bh / 2) / h
                norm_bw = bw / w
                norm_bh = bh / h

                # segmentacija (maksimalno 100 točaka)
                cnt = cnt.squeeze()
                if len(cnt.shape) != 2:
                    continue
                if len(cnt) > 100:
                    idx = np.linspace(0, len(cnt)-1, 100, dtype=np.int32)
                    cnt = cnt[idx]

                segmentation = []
                for px, py in cnt:
                    segmentation.append(f"{px/w:.6f} {py/h:.6f}")

                seg_str = " ".join(segmentation)
                line = f"0 {x_center:.6f} {y_center:.6f} {norm_bw:.6f} {norm_bh:.6f} {seg_str}"
                yolo_lines.append(line)

    # 3. Spremi anotaciju ako ne postoji ili je prazna
    label_name = os.path.splitext(image_name)[0] + ".txt"
    label_path = os.path.join(temp_labels, label_name)
    if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines))

    # 4. Ispis
    counter += 1
    print(f"[{counter}/{total}] Slika: {folder_name} → pronađeno maski: {mask_count}")

# 5. Podjela na train/val/test ako images/train ne postoji
img_train_dir = os.path.join(output_root, "images", "train")
if not os.path.exists(img_train_dir):
    all_files = [f for f in os.listdir(temp_images) if f.endswith(".png")]
    random.shuffle(all_files)

    n = len(all_files)
    train_end = int(0.7 * n)
    val_end = train_end + int(0.15 * n)

    splits = {
        "train": all_files[:train_end],
        "val": all_files[train_end:val_end],
        "test": all_files[val_end:]
    }

    for split, files in splits.items():
        img_dir = os.path.join(output_root, "images", split)
        lbl_dir = os.path.join(output_root, "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for fname in files:
            shutil.move(os.path.join(temp_images, fname), os.path.join(img_dir, fname))
            txt_name = os.path.splitext(fname)[0] + ".txt"
            label_src = os.path.join(temp_labels, txt_name)
            label_dst = os.path.join(lbl_dir, txt_name)

            if os.path.exists(label_src):
                shutil.move(label_src, label_dst)
            else:
                open(label_dst, "w").close()

    shutil.rmtree(temp_images)
    shutil.rmtree(temp_labels)

print("\u2705 Gotovo! Dataset za segmentaciju i detekciju je spreman.")