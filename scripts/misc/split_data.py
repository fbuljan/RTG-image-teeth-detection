import os
import shutil
import pandas as pd
import random

data_dir = "data/images"
train_dir = "data/images/train"
val_dir = "data/images/val"
test_dir = "data/images/test"

for d in [train_dir, val_dir, test_dir]:
    os.makedirs(d, exist_ok=True)

df = pd.read_csv("data/images_data.csv", delimiter=";")
image_files = df["image_name"].tolist()

random.shuffle(image_files)
split_train = int(len(image_files) * 0.8)
split_val = int(len(image_files) * 0.9)

train_images = image_files[:split_train]
val_images = image_files[split_train:split_val]
test_images = image_files[split_val:]

for img in train_images:
    shutil.move(os.path.join(data_dir, img), train_dir)

for img in val_images:
    shutil.move(os.path.join(data_dir, img), val_dir)

for img in test_images:
    shutil.move(os.path.join(data_dir, img), test_dir)

print("Data divided to train/val/test!")
