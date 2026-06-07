"""Build the full-dataset retrieval registry for the Phase 6 demo.

Embeds every tooth across all splits (train + val + test) using the chosen
embedding checkpoint, mean-pools per person, and saves:

  identification/registry/
    index.faiss              # FAISS IndexFlatIP over (n_persons, embedding_dim)
    index.ids.json           # parallel array of person_ids written by RetrievalIndex
    registry_meta.json       # list of {person_id, fake_name, n_teeth, image_id, panoramic_path, faiss_idx}

Default checkpoint is the FDI-init embedder (best multi-tooth aggregation
performance from Phase 5).

Usage:
    python -m identification.scripts.build_registry
    python -m identification.scripts.build_registry --checkpoint <path> --output-dir <path>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from identification.data.tooth_dataset import ToothDataset, get_val_transforms
from identification.evaluation.evaluate_embedding import (
    extract_embeddings,
    load_checkpoint,
)
from identification.models.embedding_model import ToothEmbeddingModelWithMetadata
from identification.models.person_aggregation import aggregate_by_person
from identification.models.retrieval_index import RetrievalIndex
from identification.utils.fake_names import disambiguate, fake_name_for

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = "identification/runs/embedding_fdi_init_v1/best.pt"
DEFAULT_OUTPUT_DIR = "identification/registry"


def build_full_dataset(cfg, label_map, ckpt, uses_metadata, manifest_override=None):
    """ToothDataset spanning every split — full registry coverage.

    `manifest_override` lets the caller swap in a different manifest (e.g. the
    YOLO-cropped one for the Phase 7.1 ensemble) without touching the embedder's
    training config.
    """
    manifest_path = manifest_override or cfg["data"]["manifest"]
    df = pd.read_csv(manifest_path, dtype=str)
    all_persons = df["person_id"].unique().tolist()
    if label_map is None or set(label_map.keys()) < set(all_persons):
        # Re-build label_map across all splits so train+val+test persons all have labels
        label_map = {pid: i for i, pid in enumerate(sorted(all_persons))}

    datasets = []
    for split in ("train", "val", "test"):
        try:
            ds = ToothDataset(
                manifest_path=manifest_path,
                split=split,
                crop_mode=cfg["data"]["crop_mode"],
                target_col="person_id",
                transform=get_val_transforms(),
                label_map=label_map,
                return_metadata=uses_metadata,
                fdi_label_map=ckpt.get("fdi_label_map") if uses_metadata else None,
            )
            if len(ds) > 0:
                datasets.append(ds)
        except Exception as e:
            print(f"  Skipping split={split}: {e}")
    if not datasets:
        raise RuntimeError("No data found across train/val/test splits")
    combined = torch.utils.data.ConcatDataset(datasets)
    return combined, label_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Path to the embedding checkpoint to use for the registry.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Where to write index files and registry_meta.json.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None,
                        help="Device override; defaults to MPS/CUDA/CPU autoselection.")
    parser.add_argument("--manifest", default=None,
                        help="Override the embedder's training manifest with this one "
                             "(e.g. manifest_yolo.csv for the YOLO-aligned ensemble).")
    args = parser.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available()
                             else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    model, cfg, label_map, ckpt = load_checkpoint(args.checkpoint, device)
    uses_metadata = isinstance(model, ToothEmbeddingModelWithMetadata)
    print(f"Model: {'metadata-fused ' if uses_metadata else ''}embedder, dim={model.projection_head.out_features}")

    print("\n1. Loading dataset (all splits)...")
    if args.manifest:
        print(f"   Manifest override: {args.manifest}")
    dataset, label_map = build_full_dataset(
        cfg, label_map, ckpt, uses_metadata, manifest_override=args.manifest,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"   Total tooth crops: {len(dataset)}")
    print(f"   Distinct persons (label_map): {len(label_map)}")

    print("\n2. Extracting embeddings...")
    embeddings, labels = extract_embeddings(model, loader, device)
    print(f"   Embeddings: {embeddings.shape}")

    print("\n3. Aggregating per person (mean pooling)...")
    person_emb, person_label_idx = aggregate_by_person(embeddings, labels, method="mean")
    label_to_id = {v: k for k, v in label_map.items()}
    person_ids = [label_to_id[lbl] for lbl in person_label_idx]
    print(f"   Persons in registry: {len(person_ids)}")

    print("\n4. Generating fake names...")
    raw_names = {pid: fake_name_for(pid) for pid in person_ids}
    fake_names = disambiguate(raw_names)
    n_collisions = sum(1 for pid in person_ids if fake_names[pid] != raw_names[pid])
    if n_collisions:
        print(f"   {n_collisions} name collisions disambiguated with numeric suffix")

    print("\n5. Verifying panoramic source files...")
    df = pd.read_csv(args.manifest or cfg["data"]["manifest"], dtype=str)
    image_id_by_person = (
        df.drop_duplicates("person_id").set_index("person_id")["image_id"].to_dict()
    )
    tooth_count_by_person = df.groupby("person_id").size().to_dict()
    missing = []
    meta_records = []
    for faiss_idx, pid in enumerate(person_ids):
        image_id = image_id_by_person.get(pid)
        panoramic_path = f"dataset_raw/{image_id}/{image_id}.png" if image_id else None
        absolute = PROJECT_ROOT / panoramic_path if panoramic_path else None
        if absolute is None or not absolute.exists():
            missing.append(pid)
            continue
        meta_records.append(
            {
                "person_id": pid,
                "fake_name": fake_names[pid],
                "n_teeth": int(tooth_count_by_person.get(pid, 0)),
                "image_id": image_id,
                "panoramic_path": panoramic_path,
                "faiss_idx": faiss_idx,
            }
        )
    if missing:
        print(f"   Warning: {len(missing)} persons have no panoramic on disk; excluded.")
    print(f"   Persons with panoramics: {len(meta_records)}")

    # Filter the embedding array to match meta_records (drop missing-panoramic rows)
    keep_indices = [r["faiss_idx"] for r in meta_records]
    person_emb = person_emb[keep_indices]
    person_ids = [meta_records[i]["person_id"] for i in range(len(meta_records))]
    # Renumber faiss_idx to be contiguous after filtering
    for new_idx, record in enumerate(meta_records):
        record["faiss_idx"] = new_idx

    print("\n6. Building FAISS index...")
    embedding_dim = person_emb.shape[1]
    index = RetrievalIndex(dim=embedding_dim)
    index.add(person_emb, person_ids)
    print(f"   FAISS entries: {len(index)}")

    print("\n7. Saving registry...")
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    index.save(str(output_dir / "index"))
    meta_path = output_dir / "registry_meta.json"
    payload = {
        "checkpoint": args.checkpoint,
        "embedding_dim": embedding_dim,
        "aggregation": "mean",
        "n_persons": len(meta_records),
        "uses_metadata": uses_metadata,
        "persons": meta_records,
    }
    with open(meta_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"   Wrote: {output_dir / 'index.faiss'}")
    print(f"   Wrote: {output_dir / 'index.ids.json'}")
    print(f"   Wrote: {meta_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
