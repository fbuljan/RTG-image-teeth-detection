import torch
import os
import glob
import cv2
import numpy as np
from data_loader import test_loader
from yolo_wrapper import YOLOWrapper
from ultralytics.utils.plotting import Annotator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLOWrapper("yolov8s.pt", num_classes=1).model.to(device)
model.eval()

checkpoint_dir = "logs/checkpoints"
checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "weights-*.pth")), reverse=True)

if not checkpoints:
    raise FileNotFoundError("No checkpoint files found.")

latest_checkpoint = checkpoints[0]
weights_name = os.path.basename(latest_checkpoint).replace(".pth", "")
output_dir = f"logs/{weights_name}"
os.makedirs(output_dir, exist_ok=True)

print(f"Loading weights from {latest_checkpoint}")
model.load_state_dict(torch.load(latest_checkpoint, map_location=device))

with torch.no_grad():
    for batch_idx, (images, _, _) in enumerate(test_loader):
        images = images.to(device)
        preds = model(images)

        for i, pred in enumerate(preds):
            img_np = (images[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            annotator = Annotator(img_np)

            for box in pred.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0].item())
                label = f"Tooth {conf:.2f}"
                annotator.box_label((x1, y1, x2, y2), label, color=(0, 255, 0))

            output_path = os.path.join(output_dir, f"test_{batch_idx}_{i}.jpg")
            cv2.imwrite(output_path, img_np)

print(f"Test images saved in {output_dir}")
