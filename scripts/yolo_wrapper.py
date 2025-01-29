import torch
from ultralytics import YOLO
import torchvision.ops as ops
import numpy as np

class YOLOWrapper:
    def __init__(self, model_path="yolov8s.pt", num_classes=1):
        self.model = YOLO(model_path)
        self.model.model.names = ["tooth"]
        self.model.model.nc = num_classes

    def validate(self, val_loader, device="cuda"):
        self.model.model.eval()
        
        total_iou = 0
        total_detections = 0
        all_pred_boxes = []
        all_true_boxes = []

        with torch.no_grad():
            for images, bboxes, *_ in val_loader:
                images = images.to(device)
                targets = self._convert_to_yolo_format(bboxes)

                preds = self.model.model(images)

                for i, pred in enumerate(preds):
                    pred_boxes = pred.boxes.xyxy
                    pred_scores = pred.boxes.conf
                    pred_labels = pred.boxes.cls

                    true_boxes = targets[i]["boxes"]
                    true_labels = targets[i]["labels"]

                    if true_boxes.numel() == 0 or pred_boxes.numel() == 0:
                        continue

                    ious = ops.box_iou(pred_boxes, true_boxes)
                    max_ious, _ = ious.max(dim=1)

                    total_iou += max_ious.sum().item()
                    total_detections += max_ious.numel()

                    for j in range(len(pred_boxes)):
                        all_pred_boxes.append([i] + pred_boxes[j].tolist() + [pred_scores[j].item(), pred_labels[j].item()])
                    for j in range(len(true_boxes)):
                        all_true_boxes.append([i] + true_boxes[j].tolist() + [1.0, true_labels[j].item()])

        avg_iou = total_iou / total_detections if total_detections > 0 else 0
        map50 = self._calculate_map(all_pred_boxes, all_true_boxes, iou_threshold=0.5)

        print(f"Validation mAP@50: {map50:.4f}, Average IoU: {avg_iou:.4f}")

    def _calculate_map(self, pred_boxes, true_boxes, iou_threshold=0.5):
        pred_boxes = sorted(pred_boxes, key=lambda x: x[5], reverse=True)
        true_boxes_per_image = {i: [] for i in range(len(true_boxes))}
        
        for tb in true_boxes:
            true_boxes_per_image[tb[0]].append(tb[1:])

        tp = np.zeros(len(pred_boxes))
        fp = np.zeros(len(pred_boxes))
        total_gt_boxes = len(true_boxes)

        for i, pred in enumerate(pred_boxes):
            image_idx = pred[0]
            pred_bbox = np.array(pred[1:5])
            best_iou = 0
            best_gt_idx = -1

            for j, gt in enumerate(true_boxes_per_image[image_idx]):
                gt_bbox = np.array(gt[:4])
                iou = self._iou(pred_bbox, gt_bbox)

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou > iou_threshold:
                tp[i] = 1
                del true_boxes_per_image[image_idx][best_gt_idx]
            else:
                fp[i] = 1

        precision = np.cumsum(tp) / (np.cumsum(tp) + np.cumsum(fp))
        recall = np.cumsum(tp) / total_gt_boxes if total_gt_boxes > 0 else np.zeros(len(tp))

        return self._calculate_ap(precision, recall)

    def _iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return inter_area / (box1_area + box2_area - inter_area + 1e-6)

    def _calculate_ap(precision, recall):
        recall = np.concatenate(([0], recall, [1]))
        precision = np.concatenate(([0], precision, [0]))

        for i in range(len(precision) - 1, 0, -1):
            precision[i - 1] = max(precision[i - 1], precision[i])

        indices = np.where(recall[1:] != recall[:-1])[0]
        return np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])

    def _convert_to_yolo_format(self, bboxes):
        targets = []
        for bbox in bboxes:
            targets.append({"boxes": bbox[:, 1:], "labels": bbox[:, 0].long()})
        return targets

    def predict(self, images, device="cuda"):
        self.model.to(device)
        results = self.model(images)
        return results
