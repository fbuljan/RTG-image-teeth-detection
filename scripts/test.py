import torch
import os
import glob
from data_loader import test_loader
from customYOLO import CustomYOLO
from utils import calculate_metrics, calculate_iou

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomYOLO().to(device)
model.eval()

# Load the latest weights
checkpoint_dir = "logs/checkpoints"
checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "weights-*.pth")), reverse=True)
if not checkpoints:
    raise FileNotFoundError("No checkpoint files found in logs/checkpoints/")
latest_checkpoint = checkpoints[0]
print(f"Loading weights from {latest_checkpoint}")
model.load_state_dict(torch.load(latest_checkpoint, map_location=device))

total_metrics = {"loss": 0, "accuracy": 0, "precision": 0, "recall": 0, "iou": 0}
num_batches = len(test_loader)

with torch.no_grad():
    for images, bboxes, attr_labels, index_values, quad_values, ages, sexes in test_loader:
        images = images.to(device)
        attr_labels = attr_labels.to(device)
        index_values = index_values.to(device)
        quad_values = quad_values.to(device)
        bboxes = bboxes.to(device)

        yolo_out, attributes_pred, index_pred, quad_pred = model(images)

        loss = model.compute_loss(
            yolo_loss=yolo_out.loss,
            attributes_pred=attributes_pred,
            attributes_true=attr_labels,
            index_pred=index_pred,
            index_true=index_values,
            quad_pred=quad_pred,
            quad_true=quad_values
        )
        total_metrics["loss"] += loss.item()

        batch_metrics = calculate_metrics(index_pred, index_values, quad_pred, quad_values)
        for key in ["accuracy", "precision", "recall"]:
            total_metrics[key] += batch_metrics[key]
        
        # Compute IoU for bounding boxes
        iou_score = calculate_iou(yolo_out.pred_boxes, bboxes)
        total_metrics["iou"] += iou_score

# Compute average metrics
for key in total_metrics:
    total_metrics[key] /= num_batches

print(f"Test Results | Loss: {total_metrics['loss']:.4f} | Accuracy: {total_metrics['accuracy']:.4f} | Precision: {total_metrics['precision']:.4f} | Recall: {total_metrics['recall']:.4f} | IoU: {total_metrics['iou']:.4f}")
