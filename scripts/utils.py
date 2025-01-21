import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score

def calculate_metrics(index_pred, index_true_list, quad_pred, quad_true_list):
    # index_pred -> [B, 10]
    # index_true_list -> list of length B, each is a tensor [N_i]
    # quad_pred -> [B, 10]
    # quad_true_list -> list of length B, each is a tensor [N_i]
    B = index_pred.shape[0]

    index_pred_labels = torch.argmax(index_pred, dim=1)  # [B]
    quad_pred_labels = torch.argmax(quad_pred, dim=1)    # [B]

    # Collect the first ground-truth from each image's list of boxes
    gt_index = []
    gt_quad = []
    for i in range(B):
        idx_tensor = index_true_list[i]
        quad_tensor = quad_true_list[i]
        
        if idx_tensor.shape[0] > 0:
            gt_index.append(idx_tensor[0])
        else:
            gt_index.append(torch.tensor(0, device=idx_tensor.device))

        if quad_tensor.shape[0] > 0:
            gt_quad.append(quad_tensor[0])
        else:
            gt_quad.append(torch.tensor(0, device=quad_tensor.device))

    gt_index = torch.stack(gt_index, dim=0)  # [B]
    gt_quad = torch.stack(gt_quad, dim=0)    # [B]

    accuracy = (
        accuracy_score(gt_index.cpu(), index_pred_labels.cpu())
        + accuracy_score(gt_quad.cpu(), quad_pred_labels.cpu())
    ) / 2

    precision = (
        precision_score(gt_index.cpu(), index_pred_labels.cpu(), average='macro', zero_division=0)
        + precision_score(gt_quad.cpu(), quad_pred_labels.cpu(), average='macro', zero_division=0)
    ) / 2

    recall = (
        recall_score(gt_index.cpu(), index_pred_labels.cpu(), average='macro', zero_division=0)
        + recall_score(gt_quad.cpu(), quad_pred_labels.cpu(), average='macro', zero_division=0)
    ) / 2

    return {"accuracy": accuracy, "precision": precision, "recall": recall}

def calculate_iou(pred_boxes, true_boxes):
    pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

    true_x1 = true_boxes[:, 0] - true_boxes[:, 2] / 2
    true_y1 = true_boxes[:, 1] - true_boxes[:, 3] / 2
    true_x2 = true_boxes[:, 0] + true_boxes[:, 2] / 2
    true_y2 = true_boxes[:, 1] + true_boxes[:, 3] / 2

    inter_x1 = torch.max(pred_x1, true_x1)
    inter_y1 = torch.max(pred_y1, true_y1)
    inter_x2 = torch.min(pred_x2, true_x2)
    inter_y2 = torch.min(pred_y2, true_y2)

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    true_area = (true_x2 - true_x1) * (true_y2 - true_y1)

    union_area = pred_area + true_area - inter_area
    iou = inter_area / torch.clamp(union_area, min=1e-6)
    
    return iou.mean().item()
