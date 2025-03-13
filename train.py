import torch
from ultralytics import YOLO
import datetime

# === CONFIGURATION ===
MODEL_NAME = "yolov8s.pt"
DATA_CONFIG = "data.yaml"
EPOCHS = 50
BATCH_SIZE = 4
IMAGE_SIZE = 640
NAME = "yolov8_tooth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === TRAINING ===
print(f"Starting YOLOv8 training on device: {DEVICE}")
print(f"Dataset: {DATA_CONFIG}")
print(f"Hyperparameters: epochs={EPOCHS}, batch={BATCH_SIZE}, imgsz={IMAGE_SIZE}")
print("========================================")

model = YOLO(MODEL_NAME)

results = model.train(
    data=DATA_CONFIG,
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    imgsz=IMAGE_SIZE,
    name=NAME,
    device=DEVICE,
)

# === DONE ===
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("========================================")
print(f"Training complete at {timestamp}")
print(f"Results saved to: runs/detect/{NAME}")
