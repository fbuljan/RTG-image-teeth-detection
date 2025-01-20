import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score

def calculate_metrics(index_pred, index_true, quad_pred, quad_true):
    index_pred_labels = torch.argmax(index_pred, dim=1)
    quad_pred_labels = torch.argmax(quad_pred, dim=1)

    accuracy = (accuracy_score(index_true.cpu(), index_pred_labels.cpu()) + 
                accuracy_score(quad_true.cpu(), quad_pred_labels.cpu())) / 2

    precision = (precision_score(index_true.cpu(), index_pred_labels.cpu(), average='macro', zero_division=0) + 
                 precision_score(quad_true.cpu(), quad_pred_labels.cpu(), average='macro', zero_division=0)) / 2

    recall = (recall_score(index_true.cpu(), index_pred_labels.cpu(), average='macro', zero_division=0) + 
              recall_score(quad_true.cpu(), quad_pred_labels.cpu(), average='macro', zero_division=0)) / 2

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }
