import torch
from ultralytics import YOLO

def main():
    MODEL_NAME = "yolov8s.pt"
    DATA_CONFIG = "data.yaml"
    EPOCHS = 50
    BATCH_SIZE = 4
    IMAGE_SIZE = 640
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PROJECT_DIR = "runs-detection"

    model = YOLO(MODEL_NAME)
    model.train(
        data=DATA_CONFIG,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        project=PROJECT_DIR,
        device=DEVICE
    )

    model.val(
        data=DATA_CONFIG,
        split="test",
        save=True
    )

if __name__ == "__main__":
    main()
