import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO

class CustomYOLO(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = YOLO("yolov8s.pt")
        self.yolo_model = base_model.model

        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, 128)

        self.attribute_head = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
            nn.Sigmoid()
        )
        self.index_head = nn.Linear(128, 10)
        self.quad_head = nn.Linear(128, 10)

    def forward(self, x):
        x = x / 255.0 if x.max() > 1 else x
        yolo_out = self.yolo_model(x)
        feats = self.image_encoder(x).view(x.size(0), -1)
        feats = self.fc(feats)
        attributes = self.attribute_head(feats)    # shape [B, 4]
        index_pred = self.index_head(feats)        # shape [B, 10]
        quad_pred = self.quad_head(feats)          # shape [B, 10]
        return yolo_out, attributes, index_pred, quad_pred

    def compute_loss(
        self,
        yolo_loss,
        attributes_pred,  # [B, 4]
        attributes_true,  # list of [N_i, 4]
        index_pred,       # [B, 10]
        index_true,       # list of [N_i]
        quad_pred,        # [B, 10]
        quad_true         # list of [N_i]
    ):
        # Convert list-of-tensors into [B, 4] by taking the first box
        attr_batch = []
        for a in attributes_true:
            if a.shape[0] > 0:
                attr_batch.append(a[0])  # first box
            else:
                attr_batch.append(torch.zeros(4, device=a.device))
        attr_labels = torch.stack(attr_batch, dim=0)  # [B, 4]

        # Convert index_true and quad_true to [B] by taking the first
        index_batch = []
        for idx in index_true:
            if idx.shape[0] > 0:
                index_batch.append(idx[0])
            else:
                index_batch.append(torch.tensor(0, device=idx.device))
        index_labels = torch.stack(index_batch, dim=0)  # [B]

        quad_batch = []
        for qd in quad_true:
            if qd.shape[0] > 0:
                quad_batch.append(qd[0])
            else:
                quad_batch.append(torch.tensor(0, device=qd.device))
        quad_labels = torch.stack(quad_batch, dim=0)   # [B]

        attr_loss = F.binary_cross_entropy(attributes_pred, attr_labels)
        index_loss = F.cross_entropy(index_pred, index_labels)
        quad_loss = F.cross_entropy(quad_pred, quad_labels)

        total_loss = yolo_loss + attr_loss + index_loss + quad_loss
        return total_loss
