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


def _mine_safe(miner_fn, embeddings, labels):
    """Run the PML miner; on MPS RuntimeError fall back to CPU INDICES only.

    Critically returns the mined indices for use against the on-device embeddings,
    so the triplet loss is computed on-device and gradients flow back through
    the backbone. (The pre-Phase-8.4 code .detach().cpu()'d the embeddings and
    computed loss on CPU, which silently broke gradient flow into the backbone
    on the fallback path — see Phase 8.4 design review.)
    """
    try:
        return miner_fn(embeddings, labels)
    except RuntimeError:
        return miner_fn(embeddings.detach().cpu(), labels.cpu())


def _hard_pairs_nonempty(hard_pairs) -> bool:
    """PML miners return a tuple of index tensors; first element empty == no triplets."""
    if hard_pairs is None:
        return False
    if not hasattr(hard_pairs, "__len__") or len(hard_pairs) == 0:
        return False
    first = hard_pairs[0] if isinstance(hard_pairs, tuple) else hard_pairs
    try:
        return len(first) > 0
    except TypeError:
        return False


def _hard_pairs_to_device(hard_pairs, device):
    """Move mined indices (from possibly-CPU miner) to the embedding device."""
    if isinstance(hard_pairs, tuple):
        return tuple(t.to(device) if hasattr(t, "to") else t for t in hard_pairs)
    return hard_pairs.to(device) if hasattr(hard_pairs, "to") else hard_pairs


def train_one_epoch(model, loader, loss_fn, miner_fn, optimizer, scheduler, device, epoch,
                    aux_lambda: float = 0.0):
    """Training loop; supports Phase 8.4 multi-task FDI aux loss when the model has
    fdi_head and the loader yields (images, person_labels, fdi_labels)."""
    model.train()
    total_loss = 0.0
    total_triplet = 0.0
    total_aux = 0.0
    num_batches = 0
    num_triplets = 0
    aux_enabled = aux_lambda > 0 and getattr(model, "fdi_head", None) is not None

    for i, batch in enumerate(loader):
        if aux_enabled:
            images, labels, fdi_labels = batch
            fdi_labels = fdi_labels.to(device)
        else:
            images, labels = batch
            fdi_labels = None

        images = images.to(device)
        labels = labels.to(device)

        model_out = model(images)
        if aux_enabled:
            embeddings, fdi_logits = model_out
        else:
            embeddings = model_out
            fdi_logits = None

        # Miner (may fall back to CPU on MPS); resulting INDICES move back to device
        # so triplet loss stays on-device and the backbone receives gradient.
        hard_pairs = _mine_safe(miner_fn, embeddings, labels)
        hard_pairs = _hard_pairs_to_device(hard_pairs, embeddings.device)

        # Per-term loss: triplet only when miner found pairs; aux always (when enabled)
        triplet_val = 0.0
        aux_val = 0.0
        if _hard_pairs_nonempty(hard_pairs):
            triplet = loss_fn(embeddings, labels, hard_pairs)
            triplet_val = float(triplet.detach().item())
        else:
            triplet = None

        if aux_enabled:
            aux = F.cross_entropy(fdi_logits, fdi_labels)
            aux_val = float(aux.detach().item())
            if triplet is not None:
                loss = triplet + aux_lambda * aux
            else:
                loss = aux_lambda * aux
        else:
            if triplet is None:
                # No triplet, no aux: skip backward (preserves pre-8.4 behaviour)
                if scheduler is not None:
                    scheduler.step()
                continue
            loss = triplet

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += float(loss.detach().item())
        total_triplet += triplet_val
        total_aux += aux_val
        num_batches += 1
        if _hard_pairs_nonempty(hard_pairs):
            num_triplets += len(hard_pairs[0]) if isinstance(hard_pairs, tuple) else 0

        if device == "mps" and (i + 1) % 50 == 0:
            torch.mps.empty_cache()

    metrics = {
        "loss": total_loss / max(num_batches, 1),
        "num_triplets": num_triplets,
        "triplet_loss": total_triplet / max(num_batches, 1),
        "aux_loss": total_aux / max(num_batches, 1),
    }
    if aux_enabled and metrics["triplet_loss"] > 1e-9:
        metrics["lambda_aux"] = aux_lambda * metrics["aux_loss"]
        metrics["aux_ratio"] = (aux_lambda * metrics["aux_loss"]) / metrics["triplet_loss"]
    return metrics


@torch.no_grad()
def compute_val_metrics(model, val_loader, device, aux_enabled: bool = False):
    """Embed all val samples and compute Rank-1 accuracy. When aux_enabled, the
    val_loader is expected to yield (img, person_label, fdi_label) and the model
    forward returns (emb, fdi_logits); we additionally compute FDI top-1 acc."""
    model.eval()
    all_embeddings = []
    all_labels = []
    fdi_correct = 0
    fdi_total = 0

    for batch in val_loader:
        if aux_enabled:
            images, labels, fdi_labels = batch
            fdi_labels = fdi_labels.to(device)
        else:
            images, labels = batch
            fdi_labels = None

        images = images.to(device)
        out = model(images)
        if aux_enabled:
            emb, fdi_logits = out
            fdi_pred = fdi_logits.argmax(dim=1)
            fdi_correct += int((fdi_pred == fdi_labels).sum().item())
            fdi_total += int(fdi_labels.numel())
        else:
            emb = out
        all_embeddings.append(emb.cpu())
        all_labels.append(labels)

    embeddings = torch.cat(all_embeddings, dim=0)  # (N, dim)
    labels = torch.cat(all_labels, dim=0)           # (N,)

    sim_matrix = embeddings @ embeddings.T
    sim_matrix.fill_diagonal_(-float("inf"))
    nn_indices = sim_matrix.argmax(dim=1)
    nn_labels = labels[nn_indices]
    rank1 = (nn_labels == labels).float().mean().item()

    _, topk_indices = sim_matrix.topk(5, dim=1)
    topk_labels = labels[topk_indices]
    rank5 = (topk_labels == labels.unsqueeze(1)).any(dim=1).float().mean().item()

    if device == "mps":
        torch.mps.empty_cache()

    metrics = {"rank1": rank1, "rank5": rank5}
    if aux_enabled and fdi_total > 0:
        metrics["fdi_acc"] = fdi_correct / fdi_total
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train tooth embedding model")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--init-from-classifier", default=None,
                        help="Path to Phase 2 classifier checkpoint; copies backbone weights")
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

    # Phase 8.4 — optional FDI multi-task aux loss. Build FDI->idx label map
    # consistent with the deployed FDI classifier (52 classes, sorted numerically).
    aux_cfg = cfg.get("aux_loss") or {}
    aux_lambda = float(aux_cfg.get("lambda", 0.0)) if aux_cfg.get("type") == "fdi" else 0.0
    fdi_label_map: dict | None = None
    num_fdi_classes: int | None = None
    if aux_lambda > 0.0:
        fdi_label_map = ToothDataset.build_label_map(manifest_path, "tooth_fdi")
        num_fdi_classes = len(fdi_label_map)
        print(f"Aux loss: FDI classification (num_classes={num_fdi_classes}, lambda={aux_lambda})")

    # Build datasets
    train_dataset = ToothDataset(
        manifest_path=manifest_path,
        split="train",
        crop_mode=cfg["data"]["crop_mode"],
        target_col=target_col,
        transform=get_train_transforms(cfg.get("augmentation")),
        label_map=label_map,
        return_metadata=(fdi_label_map is not None),
        fdi_label_map=fdi_label_map,
    )
    val_dataset = ToothDataset(
        manifest_path=manifest_path,
        split="val",
        crop_mode=cfg["data"]["crop_mode"],
        target_col=target_col,
        transform=get_val_transforms(),
        label_map=label_map,
        return_metadata=(fdi_label_map is not None),
        fdi_label_map=fdi_label_map,
    )
    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    # Phase 8.3 — optional GT->YOLO blend on the train dataset
    blend_cfg = cfg.get("data", {}).get("yolo_blend")
    if blend_cfg:
        n_pairs = train_dataset.enable_yolo_blend(
            pair_table_path=blend_cfg["pair_table"],
            prob=blend_cfg["prob"],
        )
        print(f"YOLO blend: prob={blend_cfg['prob']}, eligible pairs={n_pairs} "
              f"(pair_table={blend_cfg['pair_table']})")

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
        num_fdi_classes=num_fdi_classes,
    ).to(device)
    head_str = f" + FDI head ({num_fdi_classes} cls)" if num_fdi_classes else ""
    print(f"Model: ResNet-18 → {model_cfg.get('embedding_dim', 128)}-dim embedding{head_str}, "
          f"{sum(p.numel() for p in model.parameters()):,} params")

    # Optionally initialize backbone from Phase 2 classifier
    init_from = args.init_from_classifier or cfg.get("init_from_classifier")
    if init_from:
        print(f"Initializing backbone from classifier: {init_from}")
        cls_ckpt = torch.load(init_from, map_location=device, weights_only=False)
        cls_state = cls_ckpt["model_state_dict"]
        backbone_state = {k.replace("backbone.", ""): v for k, v in cls_state.items()
                          if k.startswith("backbone.")}
        missing, unexpected = model.backbone.load_state_dict(backbone_state, strict=False)
        print(f"  Loaded {len(backbone_state)} backbone tensors "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")

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
    log_fields = ["epoch", "train_loss", "triplet_loss", "aux_loss", "aux_ratio",
                  "val_rank1", "val_rank5", "val_fdi_acc", "lr", "time_s"]
    if not args.resume:
        with open(log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=log_fields).writeheader()

    # Training loop
    eval_cfg = cfg.get("eval", {})
    val_every = eval_cfg.get("val_every_n_epochs", 3)
    patience = eval_cfg.get("patience", 10)
    patience_counter = 0

    # Phase 8.4 kill-switches (only active when aux loss is on)
    aux_enabled = aux_lambda > 0.0
    killswitch_cfg = cfg.get("killswitch") or {}
    ks_epoch = int(killswitch_cfg.get("epoch", 20))
    ks_min_val_rank1 = float(killswitch_cfg.get("min_val_rank1", 0.0))
    ks_min_fdi_acc = float(killswitch_cfg.get("min_fdi_acc", 0.0))
    ks_max_aux_ratio = float(killswitch_cfg.get("max_aux_ratio", float("inf")))
    if aux_enabled:
        print(f"Kill-switch at epoch {ks_epoch}: val_rank1 >= {ks_min_val_rank1}, "
              f"val_fdi_acc >= {ks_min_fdi_acc}, max aux_ratio <= {ks_max_aux_ratio}")

    print(f"\nTraining for {epochs} epochs (validate every {val_every}, patience={patience})...\n")

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, loss_fn, miner_fn, optimizer, scheduler, device, epoch,
            aux_lambda=aux_lambda,
        )

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        # Validate periodically
        val_rank1, val_rank5 = 0.0, 0.0
        val_fdi_acc = 0.0
        if (epoch + 1) % val_every == 0 or epoch == 0:
            val_metrics = compute_val_metrics(model, val_loader, device, aux_enabled=aux_enabled)
            val_rank1 = val_metrics["rank1"]
            val_rank5 = val_metrics["rank5"]
            val_fdi_acc = val_metrics.get("fdi_acc", 0.0)

            extras = []
            if aux_enabled:
                extras.append(f"val_fdi={val_fdi_acc:.4f}")
                if "aux_ratio" in train_metrics:
                    extras.append(f"aux/triplet={train_metrics['aux_ratio']:.2f}")
            extras_str = (" | " + " ".join(extras)) if extras else ""

            print(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"loss={train_metrics['loss']:.4f} | "
                f"val_rank1={val_rank1:.4f} val_rank5={val_rank5:.4f}{extras_str} | "
                f"lr={lr:.6f} | {elapsed:.1f}s"
            )
        else:
            ratio_str = (f" aux/triplet={train_metrics['aux_ratio']:.2f}"
                         if aux_enabled and "aux_ratio" in train_metrics else "")
            print(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"loss={train_metrics['loss']:.4f}{ratio_str} | "
                f"lr={lr:.6f} | {elapsed:.1f}s"
            )

        # Log
        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_fields)
            writer.writerow({
                "epoch": epoch + 1,
                "train_loss": f"{train_metrics['loss']:.6f}",
                "triplet_loss": f"{train_metrics.get('triplet_loss', 0.0):.6f}",
                "aux_loss": f"{train_metrics.get('aux_loss', 0.0):.6f}",
                "aux_ratio": (f"{train_metrics['aux_ratio']:.4f}"
                              if "aux_ratio" in train_metrics else ""),
                "val_rank1": f"{val_rank1:.6f}" if val_rank1 > 0 else "",
                "val_rank5": f"{val_rank5:.6f}" if val_rank5 > 0 else "",
                "val_fdi_acc": f"{val_fdi_acc:.6f}" if val_fdi_acc > 0 else "",
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
        if fdi_label_map is not None:
            checkpoint_data["fdi_label_map"] = fdi_label_map

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

        # Phase 8.4 kill-switch at the configured epoch
        if aux_enabled and (epoch + 1) == ks_epoch and val_rank1 > 0:
            reasons = []
            if val_rank1 < ks_min_val_rank1:
                reasons.append(f"val_rank1={val_rank1:.4f} < {ks_min_val_rank1}")
            if val_fdi_acc < ks_min_fdi_acc:
                reasons.append(f"val_fdi_acc={val_fdi_acc:.4f} < {ks_min_fdi_acc}")
            if train_metrics.get("aux_ratio", 0.0) > ks_max_aux_ratio:
                reasons.append(f"aux_ratio={train_metrics['aux_ratio']:.2f} > {ks_max_aux_ratio}")
            if reasons:
                print(f"  KILL-SWITCH triggered at epoch {epoch+1}: " + "; ".join(reasons))
                break

    print(f"\nTraining complete. Best rank1={best_rank1:.4f}")
    print(f"Checkpoints: {output_dir}")


if __name__ == "__main__":
    main()
