"""
Training script for tooth embedding model (metric learning).

Uses triplet loss with online hard negative mining via pytorch-metric-learning.
Person ID is the label — teeth from the same person should be close in embedding space.

Usage:
    python -m identification.training.train_embedding --config identification/configs/embedding_triplet.yaml
    python -m identification.training.train_embedding --config identification/configs/embedding_triplet.yaml --resume path/to/last.pt
"""

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
import yaml

from pytorch_metric_learning import losses, miners

from identification.data.tooth_dataset import (
    ToothDataset,
    get_train_transforms,
    get_val_transforms,
)
from identification.data.pk_sampler import PKSampler
from identification.models.embedding_model import ToothEmbeddingModel


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


def build_loss_and_miner(cfg: dict, device: str):
    """Build loss function and miner from config."""
    loss_cfg = cfg.get("loss", {})
    miner_cfg = cfg.get("miner", {})

    loss_type = loss_cfg.get("type", "triplet_margin")
    if loss_type == "triplet_margin":
        loss_fn = losses.TripletMarginLoss(margin=loss_cfg.get("margin", 0.2))
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    miner_type = miner_cfg.get("type", "multi_similarity")
    if miner_type == "multi_similarity":
        miner_fn = miners.MultiSimilarityMiner(epsilon=miner_cfg.get("epsilon", 0.1))
    elif miner_type == "triplet_margin":
        miner_fn = miners.TripletMarginMiner(
            margin=loss_cfg.get("margin", 0.2),
            type_of_triplets=miner_cfg.get("triplet_type", "semihard"),
        )
    else:
        raise ValueError(f"Unknown miner type: {miner_type}")

    return loss_fn, miner_fn


def train_one_epoch(model, loader, loss_fn, miner_fn, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0.0
    num_batches = 0
    num_triplets = 0

    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        embeddings = model(images)

        # Mine hard pairs and compute loss
        # Fall back to CPU if MPS has issues with PML operations
        try:
            hard_pairs = miner_fn(embeddings, labels)
            loss = loss_fn(embeddings, labels, hard_pairs)
        except RuntimeError:
            emb_cpu = embeddings.detach().cpu().requires_grad_(True)
            lab_cpu = labels.cpu()
            hard_pairs = miner_fn(emb_cpu, lab_cpu)
            loss = loss_fn(emb_cpu, lab_cpu, hard_pairs)

        if loss.item() > 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1
        if hasattr(hard_pairs, '__len__') and len(hard_pairs) > 0:
            num_triplets += len(hard_pairs[0]) if isinstance(hard_pairs, tuple) else 0

        # MPS memory management
        if device == "mps" and (i + 1) % 50 == 0:
            torch.mps.empty_cache()

    return {
        "loss": total_loss / max(num_batches, 1),
        "num_triplets": num_triplets,
    }


@torch.no_grad()
def compute_val_metrics(model, val_loader, device):
    """Embed all val samples and compute Rank-1 accuracy."""
    model.eval()
    all_embeddings = []
    all_labels = []

    for images, labels in val_loader:
        images = images.to(device)
        emb = model(images)
        all_embeddings.append(emb.cpu())
        all_labels.append(labels)

    embeddings = torch.cat(all_embeddings, dim=0)  # (N, dim)
    labels = torch.cat(all_labels, dim=0)           # (N,)

    # Cosine similarity (embeddings are L2-normalized → dot product = cosine)
    sim_matrix = embeddings @ embeddings.T
    sim_matrix.fill_diagonal_(-float("inf"))  # exclude self

    # Rank-1: is the nearest neighbor from the same person?
    nn_indices = sim_matrix.argmax(dim=1)
    nn_labels = labels[nn_indices]
    rank1 = (nn_labels == labels).float().mean().item()

    # Rank-5
    _, topk_indices = sim_matrix.topk(5, dim=1)
    topk_labels = labels[topk_indices]
    rank5 = (topk_labels == labels.unsqueeze(1)).any(dim=1).float().mean().item()

    if device == "mps":
        torch.mps.empty_cache()

    return {"rank1": rank1, "rank5": rank5}


def main():
    parser = argparse.ArgumentParser(description="Train tooth embedding model")
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

    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # Build label map (person_id → int)
    manifest_path = cfg["data"]["manifest"]
    target_col = cfg["data"]["target_col"]
    label_map = ToothDataset.build_label_map(manifest_path, target_col)
    num_persons = len(label_map)
    print(f"Task: person identification, Persons: {num_persons}")

    # Build datasets
    train_dataset = ToothDataset(
        manifest_path=manifest_path,
        split="train",
        crop_mode=cfg["data"]["crop_mode"],
        target_col=target_col,
        transform=get_train_transforms(cfg.get("augmentation")),
        label_map=label_map,
    )
    val_dataset = ToothDataset(
        manifest_path=manifest_path,
        split="val",
        crop_mode=cfg["data"]["crop_mode"],
        target_col=target_col,
        transform=get_val_transforms(),
        label_map=label_map,
    )
    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    # PK sampler for training
    sampler_cfg = cfg["sampler"]
    p, k = sampler_cfg["p"], sampler_cfg["k"]
    train_labels = train_dataset.get_labels()
    pk_sampler = PKSampler(train_labels, p=p, k=k)
    print(f"PK sampler: P={p}, K={k}, batch_size={p*k}, batches/epoch={len(pk_sampler)}")

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=pk_sampler,
        num_workers=cfg["data"].get("num_workers", 0),
        pin_memory=cfg["data"].get("pin_memory", False),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 0),
        pin_memory=cfg["data"].get("pin_memory", False),
    )

    # Model
    model_cfg = cfg["model"]
    model = ToothEmbeddingModel(
        embedding_dim=model_cfg.get("embedding_dim", 128),
        pretrained=model_cfg.get("pretrained", True),
        dropout=model_cfg.get("dropout", 0.2),
    ).to(device)
    print(f"Model: ResNet-18 → {model_cfg.get('embedding_dim', 128)}-dim embedding, "
          f"{sum(p.numel() for p in model.parameters()):,} params")

    # Loss and miner
    loss_fn, miner_fn = build_loss_and_miner(cfg, device)
    print(f"Loss: {cfg['loss']['type']}, Miner: {cfg['miner']['type']}")

    # Optimizer
    train_cfg = cfg["train"]
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    # Scheduler: linear warmup + cosine decay
    epochs = train_cfg["epochs"]
    warmup_epochs = train_cfg.get("warmup_epochs", 2)
    steps_per_epoch = len(pk_sampler)
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
    best_rank1 = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_rank1 = checkpoint.get("best_rank1", 0.0)
        print(f"Resumed from epoch {start_epoch}, best_rank1={best_rank1:.4f}")

    # Training log
    log_path = output_dir / "training_log.csv"
    log_fields = ["epoch", "train_loss", "val_rank1", "val_rank5", "lr", "time_s"]
    if not args.resume:
        with open(log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=log_fields).writeheader()

    # Training loop
    eval_cfg = cfg.get("eval", {})
    val_every = eval_cfg.get("val_every_n_epochs", 3)
    patience = eval_cfg.get("patience", 10)
    patience_counter = 0

    print(f"\nTraining for {epochs} epochs (validate every {val_every}, patience={patience})...\n")

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, loss_fn, miner_fn, optimizer, scheduler, device, epoch
        )

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        # Validate periodically
        val_rank1, val_rank5 = 0.0, 0.0
        if (epoch + 1) % val_every == 0 or epoch == 0:
            val_metrics = compute_val_metrics(model, val_loader, device)
            val_rank1 = val_metrics["rank1"]
            val_rank5 = val_metrics["rank5"]

            print(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"loss={train_metrics['loss']:.4f} | "
                f"val_rank1={val_rank1:.4f} val_rank5={val_rank5:.4f} | "
                f"lr={lr:.6f} | {elapsed:.1f}s"
            )
        else:
            print(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"loss={train_metrics['loss']:.4f} | "
                f"lr={lr:.6f} | {elapsed:.1f}s"
            )

        # Log
        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_fields)
            writer.writerow({
                "epoch": epoch + 1,
                "train_loss": f"{train_metrics['loss']:.6f}",
                "val_rank1": f"{val_rank1:.6f}" if val_rank1 > 0 else "",
                "val_rank5": f"{val_rank5:.6f}" if val_rank5 > 0 else "",
                "lr": f"{lr:.8f}",
                "time_s": f"{elapsed:.1f}",
            })

        # Checkpoint
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_rank1": max(best_rank1, val_rank1),
            "label_map": label_map,
            "config": cfg,
        }

        if cfg["output"].get("save_last", True):
            torch.save(checkpoint_data, output_dir / "last.pt")

        # Early stopping on validation epochs only
        if val_rank1 > 0:
            if val_rank1 > best_rank1:
                best_rank1 = val_rank1
                patience_counter = 0
                if cfg["output"].get("save_best", True):
                    torch.save(checkpoint_data, output_dir / "best.pt")
                print(f"  -> New best rank1={best_rank1:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1} (patience={patience})")
                    break

    print(f"\nTraining complete. Best rank1={best_rank1:.4f}")
    print(f"Checkpoints: {output_dir}")


if __name__ == "__main__":
    main()
