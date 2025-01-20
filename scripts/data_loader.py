import torch
from torch.utils.data import DataLoader
from dataset import ToothDetectionDataset, collate_fn

# Create dataset instances
train_dataset = ToothDetectionDataset("data/annotations.xml", "data/images_data.csv", "data/images/train")
val_dataset = ToothDetectionDataset("data/annotations.xml", "data/images_data.csv", "data/images/val")
test_dataset = ToothDetectionDataset("data/annotations.xml", "data/images_data.csv", "data/images/test")

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
