from ultralytics import YOLO
import torch.nn as nn

class CustomYOLO(nn.Module):
    def __init__(self):
        super(CustomYOLO, self).__init__()
        self.yolo = YOLO("yolov8s.pt")
        self.attribute_head = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
            nn.Sigmoid()
        )
        self.index_head = nn.Linear(1280, 10)
        self.quad_head = nn.Linear(1280, 10)

    def forward(self, x):
        yolo_out = self.yolo(x)
        features = yolo_out.features
        attributes = self.attribute_head(features)
        index_pred = self.index_head(features)
        quad_pred = self.quad_head(features)
        return yolo_out, attributes, index_pred, quad_pred
