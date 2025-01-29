import torch
import os
import glob
from data_loader import test_loader
from scripts.misc.customYOLO import CustomYOLO
from scripts.misc.utils import calculate_metrics, calculate_iou

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomYOLO().to(device)
model.eval()

checkpoint_dir = "logs/checkpoints"
checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "weights-*.pth")), reverse=True)
if not checkpoints:
    raise FileNotFoundError("No checkpoint files found.")
latest_checkpoint = checkpoints[0]
print(f"Loading weights from {latest_checkpoint}")
model.load_state_dict(torch.load(latest_checkpoint, map_location=device))

total_metrics = {"loss": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "iou": 0.0}
num_batches = len(test_loader)

with torch.no_grad():
    for images, bboxes, attr_labels, index_values, quad_values, ages, sexes in test_loader:
        images = images.to(device)
        bboxes = [b.to(device) for b in bboxes]
        attr_labels = [a.to(device) for a in attr_labels]
        index_values = [i.to(device) for i in index_values]
        quad_values = [q.to(device) for q in quad_values]

        yolo_out, attributes_pred, index_pred, quad_pred = model(images)
        yolo_loss = 0.0
        loss = model.compute_loss(
            yolo_loss=yolo_loss,
            attributes_pred=attributes_pred,
            attributes_true=attr_labels,
            index_pred=index_pred,
            index_true=index_values,
            quad_pred=quad_pred,
            quad_true=quad_values
        )
        total_metrics["loss"] += loss.item()

        batch_metrics = calculate_metrics(index_pred, index_values, quad_pred, quad_values)
        total_metrics["accuracy"] += batch_metrics["accuracy"]
        total_metrics["precision"] += batch_metrics["precision"]
        total_metrics["recall"] += batch_metrics["recall"]

        pred_boxes_list = [res.boxes.xyxy for res in yolo_out]
        iou_accum = 0.0
        pairs_count = 0
        for i in range(len(pred_boxes_list)):
            if len(pred_boxes_list[i]) > 0 and len(bboxes[i]) > 0:
                p_xyxy = pred_boxes_list[i]
                t_xywh = bboxes[i][:, 1:]
                p_xywh = torch.zeros_like(p_xyxy)
                p_xywh[:, 0] = (p_xyxy[:, 0] + p_xyxy[:, 2]) / 2
                p_xywh[:, 1] = (p_xyxy[:, 1] + p_xyxy[:, 3]) / 2
                p_xywh[:, 2] = p_xyxy[:, 2] - p_xyxy[:, 0]
                p_xywh[:, 3] = p_xyxy[:, 3] - p_xyxy[:, 1]
                m = min(len(t_xywh), len(p_xywh))
                if m > 0:
                    iou_accum += calculate_iou(p_xywh[:m], t_xywh[:m])
                    pairs_count += 1
        if pairs_count > 0:
            iou_accum /= pairs_count
        total_metrics["iou"] += iou_accum

for k in total_metrics:
    total_metrics[k] /= num_batches

print(
    f"Test Results | Loss: {total_metrics['loss']:.4f} "
    f"| Accuracy: {total_metrics['accuracy']:.4f} "
    f"| Precision: {total_metrics['precision']:.4f} "
    f"| Recall: {total_metrics['recall']:.4f} "
    f"| IoU: {total_metrics['iou']:.4f}"
)
