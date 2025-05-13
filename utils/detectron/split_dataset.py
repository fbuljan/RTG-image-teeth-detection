import os
import random

DATASET_ROOT = "dataset_raw"
COCO_JSON_PATH = "coco_annotations.json"
SPLIT_DIR = "splits"
SPLIT_NAMES = ["train", "val", "test"]

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def create_split():
    os.makedirs(SPLIT_DIR, exist_ok=True)

    all_folders = [d for d in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, d))]
    image_paths = [os.path.join(DATASET_ROOT, name, f"{name}.png") for name in all_folders]
    random.shuffle(image_paths)

    total = len(image_paths)
    train_end = int(TRAIN_RATIO * total)
    val_end = train_end + int(VAL_RATIO * total)

    splits = {
        "train": image_paths[:train_end],
        "val": image_paths[train_end:val_end],
        "test": image_paths[val_end:],
    }

    for split_name in SPLIT_NAMES:
        with open(os.path.join(SPLIT_DIR, f"{split_name}.txt"), "w") as f:
            for path in splits[split_name]:
                f.write(path + "\n")

    print("✅ Dataset split written to", SPLIT_DIR)

if __name__ == "__main__":
    create_split()
