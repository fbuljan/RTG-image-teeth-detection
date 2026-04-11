"""
Training script for tooth classification (FDI index, eruption, root status).

Config-driven via YAML. Supports MPS/CUDA/CPU with automatic device selection.

Usage:
    python -m identification.training.train_classifier --config identification/configs/tooth_classifier.yaml
    python -m identification.training.train_classifier --config identification/configs/tooth_classifier.yaml --resume path/to/last.pt
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
import yaml

from identification.data.tooth_dataset import (
    ToothDataset,
    get_train_transforms,
    get_val_transforms,
)
from identification.models.classifier import ToothClassifier


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_filter_fn(cfg: dict):
    """Build a row filter function based on config."""
    target_col = cfg["data"]["target_col"]
    if cfg["data"].get("filter_nonempty", False):
        return lambda df: df[df[target_col].notna() & (df[target_col] != "")]
    return None


def compute_class_weights(dataset: ToothDataset, strategy: str, device: str) -> torch.Tensor:
    """Compute class weights for imbalanced datasets."""
    if strategy == "none":
        return None
    counts = dataset.get_class_counts()
    num_classes = dataset.num_classes
    weight = torch.zeros(num_classes)
    for name, idx in dataset.label_map.items():
        c = counts.get(name, 1)
        if strategy == "inverse_freq":
            weight[idx] = 1.0 / c
        elif strategy == "sqrt_inverse_freq":
            weight[idx] = 1.0 / np.sqrt(c)
    weight = weight / weight.sum() * num_classes
    return weight.to(device)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

        # MPS memory management
        if device == "mps" and (i + 1) % 50 == 0:
            torch.mps.empty_cache()

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

    if device == "mps":
        torch.mps.empty_cache()

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }


def main():
    parser = argparse.ArgumentParser(description="Train tooth classifier")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    # Output directory
    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config copy
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # Build label map and filter function
    filter_fn = build_filter_fn(cfg)
    manifest_path = cfg["data"]["manifest"]
    target_col = cfg["data"]["target_col"]
    label_map = ToothDataset.build_label_map(manifest_path, target_col, filter_fn)
    num_classes = len(label_map)
    print(f"Task: {target_col}, Classes: {num_classes}, Labels: {list(label_map.keys())[:10]}...")

    # Build datasets
    train_dataset = ToothDataset(
        manifest_path=manifest_path,
        split="train",
        crop_mode=cfg["data"]["crop_mode"],
        target_col=target_col,
        filter_fn=filter_fn,
        transform=get_train_transforms(cfg.get("augmentation")),
        label_map=label_map,
    )
    val_dataset = ToothDataset(
        manifest_path=manifest_path,
        split="val",
        crop_mode=cfg["data"]["crop_mode"],
        target_col=target_col,
        filter_fn=filter_fn,
        transform=get_val_transforms(),
        label_map=label_map,
    )
    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    # DataLoaders
    batch_size = cfg["train"]["batch_size"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg["data"].get("num_workers", 0),
        pin_memory=cfg["data"].get("pin_memory", False),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 0),
        pin_memory=cfg["data"].get("pin_memory", False),
    )

    # Model
    model_cfg = cfg["model"]
    model = ToothClassifier(
        num_classes=num_classes,
        pretrained=model_cfg.get("pretrained", True),
        dropout=model_cfg.get("dropout", 0.2),
    ).to(device)
    print(f"Model: ResNet-18, {sum(p.numel() for p in model.parameters()):,} params")

    # Loss
    train_cfg = cfg["train"]
    class_weights = compute_class_weights(
        train_dataset, train_cfg.get("class_weight_strategy", "none"), device
    )
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=train_cfg.get("label_smoothing", 0.0),
    )

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    # Scheduler: linear warmup + cosine decay
    epochs = train_cfg["epochs"]
    warmup_epochs = train_cfg.get("warmup_epochs", 2)
    steps_per_epoch = len(train_loader)
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = epochs * steps_per_epoch

    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
    )

    # Resume
    start_epoch = 0
    best_val_acc = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_acc = checkpoint.get("best_val_accuracy", 0.0)
        print(f"Resumed from epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")

    # Training log
    log_path = output_dir / "training_log.csv"
    log_fields = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "time_s"]
    if not args.resume:
        with open(log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=log_fields).writeheader()

    # Training loop
    patience = cfg["eval"].get("patience", 8)
    patience_counter = 0

    print(f"\nTraining for {epochs} epochs (patience={patience})...\n")

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch
        )
        val_metrics = validate(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} | "
            f"lr={lr:.6f} | {elapsed:.1f}s"
        )

        # Log
        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_fields)
            writer.writerow({
                "epoch": epoch + 1,
                "train_loss": f"{train_metrics['loss']:.6f}",
                "train_acc": f"{train_metrics['accuracy']:.6f}",
                "val_loss": f"{val_metrics['loss']:.6f}",
                "val_acc": f"{val_metrics['accuracy']:.6f}",
                "lr": f"{lr:.8f}",
                "time_s": f"{elapsed:.1f}",
            })

        # Checkpoint
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_accuracy": max(best_val_acc, val_metrics["accuracy"]),
            "label_map": label_map,
            "config": cfg,
        }

        if cfg["output"].get("save_last", True):
            torch.save(checkpoint_data, output_dir / "last.pt")

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            patience_counter = 0
            if cfg["output"].get("save_best", True):
                torch.save(checkpoint_data, output_dir / "best.pt")
            print(f"  -> New best val_acc={best_val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1} (patience={patience})")
                break

    print(f"\nTraining complete. Best val_acc={best_val_acc:.4f}")
    print(f"Checkpoints: {output_dir}")


if __name__ == "__main__":
    main()
