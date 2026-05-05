"""
Training script for tooth embedding model WITH FDI metadata fusion.

Extends train_embedding.py: each sample now provides (image, person_label, fdi_idx).
The model takes both image and FDI index, concatenating a learned FDI embedding
with the visual features before the projection head.

Usage:
    python -m identification.training.train_embedding_metadata --config identification/configs/embedding_metadata.yaml
"""

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
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
from identification.models.embedding_model import ToothEmbeddingModelWithMetadata


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


def build_loss_and_miner(cfg: dict):
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


def train_one_epoch(model, loader, loss_fn, miner_fn, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for i, (images, labels, fdi_idx) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        fdi_idx = fdi_idx.to(device)

        embeddings = model(images, fdi_idx)

        try:
            hard_pairs = miner_fn(embeddings, labels)
            loss = loss_fn(embeddings, labels, hard_pairs)
        except RuntimeError:
            emb_cpu = embeddings.cpu()
            lab_cpu = labels.cpu()
            hard_pairs = miner_fn(emb_cpu, lab_cpu)
            loss = loss_fn(emb_cpu, lab_cpu, hard_pairs).to(device)

        if loss.item() > 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        if device == "mps" and (i + 1) % 50 == 0:
            torch.mps.empty_cache()

    return {"loss": total_loss / max(num_batches, 1)}


@torch.no_grad()
def compute_val_metrics(model, val_loader, device):
    model.eval()
    all_embeddings = []
    all_labels = []

    for images, labels, fdi_idx in val_loader:
        images = images.to(device)
        fdi_idx = fdi_idx.to(device)
        emb = model(images, fdi_idx)
        all_embeddings.append(emb.cpu())
        all_labels.append(labels)

    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)

    sim_matrix = embeddings @ embeddings.T
    sim_matrix.fill_diagonal_(-float("inf"))

    nn_indices = sim_matrix.argmax(dim=1)
    rank1 = (labels[nn_indices] == labels).float().mean().item()

    _, topk_indices = sim_matrix.topk(5, dim=1)
    rank5 = (labels[topk_indices] == labels.unsqueeze(1)).any(dim=1).float().mean().item()

    if device == "mps":
        torch.mps.empty_cache()

    return {"rank1": rank1, "rank5": rank5}


def main():
    parser = argparse.ArgumentParser(description="Train embedding model with FDI metadata")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # Build both label maps
    manifest_path = cfg["data"]["manifest"]
    target_col = cfg["data"]["target_col"]
    label_map = ToothDataset.build_label_map(manifest_path, target_col)
    fdi_label_map = ToothDataset.build_label_map(manifest_path, "tooth_fdi")
    num_persons = len(label_map)
    num_fdi = len(fdi_label_map)
    print(f"Persons: {num_persons}, FDI classes: {num_fdi}")

    train_dataset = ToothDataset(
        manifest_path=manifest_path, split="train",
        crop_mode=cfg["data"]["crop_mode"], target_col=target_col,
        transform=get_train_transforms(cfg.get("augmentation")),
        label_map=label_map, return_metadata=True, fdi_label_map=fdi_label_map,
    )
    val_dataset = ToothDataset(
        manifest_path=manifest_path, split="val",
        crop_mode=cfg["data"]["crop_mode"], target_col=target_col,
        transform=get_val_transforms(),
        label_map=label_map, return_metadata=True, fdi_label_map=fdi_label_map,
    )
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    sampler_cfg = cfg["sampler"]
    p, k = sampler_cfg["p"], sampler_cfg["k"]
    pk_sampler = PKSampler(train_dataset.get_labels(), p=p, k=k)
    print(f"PK sampler: P={p}, K={k}, batches/epoch={len(pk_sampler)}")

    train_loader = DataLoader(
        train_dataset, batch_sampler=pk_sampler,
        num_workers=cfg["data"].get("num_workers", 0),
        pin_memory=cfg["data"].get("pin_memory", False),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=False,
        num_workers=cfg["data"].get("num_workers", 0),
    )

    # Model
    model_cfg = cfg["model"]
    model = ToothEmbeddingModelWithMetadata(
        num_fdi=num_fdi,
        fdi_embedding_dim=model_cfg.get("fdi_embedding_dim", 16),
        embedding_dim=model_cfg.get("embedding_dim", 128),
        pretrained=model_cfg.get("pretrained", True),
        dropout=model_cfg.get("dropout", 0.2),
    ).to(device)
    print(f"Model: ResNet-18 + FDI({num_fdi}, {model_cfg.get('fdi_embedding_dim', 16)}d), "
          f"{sum(p.numel() for p in model.parameters()):,} params")

    loss_fn, miner_fn = build_loss_and_miner(cfg)

    train_cfg = cfg["train"]
    optimizer = AdamW(model.parameters(), lr=train_cfg["lr"],
                      weight_decay=train_cfg.get("weight_decay", 0.01))

    epochs = train_cfg["epochs"]
    warmup_epochs = train_cfg.get("warmup_epochs", 2)
    steps_per_epoch = len(pk_sampler)
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = epochs * steps_per_epoch

    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps),
            CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps),
        ],
        milestones=[warmup_steps],
    )

    start_epoch = 0
    best_rank1 = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_rank1 = ckpt.get("best_rank1", 0.0)
        print(f"Resumed from epoch {start_epoch}, best_rank1={best_rank1:.4f}")

    log_path = output_dir / "training_log.csv"
    log_fields = ["epoch", "train_loss", "val_rank1", "val_rank5", "lr", "time_s"]
    if not args.resume:
        with open(log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=log_fields).writeheader()

    eval_cfg = cfg.get("eval", {})
    val_every = eval_cfg.get("val_every_n_epochs", 3)
    patience = eval_cfg.get("patience", 10)
    patience_counter = 0

    print(f"\nTraining {epochs} epochs (val every {val_every}, patience={patience})...\n")

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        train_metrics = train_one_epoch(model, train_loader, loss_fn, miner_fn,
                                         optimizer, scheduler, device)

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        val_rank1, val_rank5 = 0.0, 0.0
        if (epoch + 1) % val_every == 0 or epoch == 0:
            val_metrics = compute_val_metrics(model, val_loader, device)
            val_rank1 = val_metrics["rank1"]
            val_rank5 = val_metrics["rank5"]
            print(f"Epoch {epoch+1:3d}/{epochs} | loss={train_metrics['loss']:.4f} | "
                  f"val_rank1={val_rank1:.4f} val_rank5={val_rank5:.4f} | lr={lr:.6f} | {elapsed:.1f}s")
        else:
            print(f"Epoch {epoch+1:3d}/{epochs} | loss={train_metrics['loss']:.4f} | "
                  f"lr={lr:.6f} | {elapsed:.1f}s")

        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=log_fields).writerow({
                "epoch": epoch + 1,
                "train_loss": f"{train_metrics['loss']:.6f}",
                "val_rank1": f"{val_rank1:.6f}" if val_rank1 > 0 else "",
                "val_rank5": f"{val_rank5:.6f}" if val_rank5 > 0 else "",
                "lr": f"{lr:.8f}",
                "time_s": f"{elapsed:.1f}",
            })

        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_rank1": max(best_rank1, val_rank1),
            "label_map": label_map,
            "fdi_label_map": fdi_label_map,
            "config": cfg,
        }

        if cfg["output"].get("save_last", True):
            torch.save(ckpt_data, output_dir / "last.pt")

        if val_rank1 > 0:
            if val_rank1 > best_rank1:
                best_rank1 = val_rank1
                patience_counter = 0
                if cfg["output"].get("save_best", True):
                    torch.save(ckpt_data, output_dir / "best.pt")
                print(f"  -> New best rank1={best_rank1:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

    print(f"\nTraining complete. Best rank1={best_rank1:.4f}")
    print(f"Checkpoints: {output_dir}")


if __name__ == "__main__":
    main()
