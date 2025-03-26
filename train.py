import torch
import yaml
from ultralytics import YOLO

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model = YOLO(config["model"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.train(
        data=config["data"],
        epochs=config["epochs"],
        batch=config["batch"],
        imgsz=config["imgsz"],
        project=config["project"],
        device=device
    )

    model.val(
        data=config["data"],
        split="test",
        save=True
    )

if __name__ == "__main__":
    main()
