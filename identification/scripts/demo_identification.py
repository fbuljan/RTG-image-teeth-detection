"""
End-to-end identification demo.

Loads a trained embedding model, builds a FAISS gallery from one split, and
runs queries from another split. Reports top-K matches with similarity scores
and whether each is the correct person.

Usage:
    # Interactive: pick a random query, show top-10
    python -m identification.scripts.demo_identification \
        --checkpoint identification/runs/embedding_fdi_init_v1/best.pt

    # Full split summary
    python -m identification.scripts.demo_identification \
        --checkpoint identification/runs/embedding_fdi_init_v1/best.pt \
        --query-split test --gallery-split test \
        --n-query 5
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from identification.evaluation.evaluate_embedding import (
    build_eval_dataset,
    extract_embeddings,
    load_checkpoint,
)
from identification.models.embedding_model import ToothEmbeddingModelWithMetadata
from identification.models.person_aggregation import aggregate_by_person
from identification.models.retrieval_index import RetrievalIndex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--query-split", default="test",
                        help="Split for queries (held out at training time)")
    parser.add_argument("--gallery-split", default="test",
                        help="Split for gallery (must contain the correct person for queries to succeed)")
    parser.add_argument("--n-query", type=int, default=5,
                        help="Number of random query persons to demo")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--aggregation", default="mean", choices=["mean", "max", "weighted"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model, cfg, label_map, ckpt = load_checkpoint(args.checkpoint, device)
    uses_metadata = isinstance(model, ToothEmbeddingModelWithMetadata)
    print(f"Model: {'metadata-fused ' if uses_metadata else ''}embedding (dim={model.projection_head.out_features})")

    # --- Build gallery ---
    print(f"\n1. Building gallery from '{args.gallery_split}' split...")
    gallery_dataset = build_eval_dataset(cfg, args.gallery_split, label_map, ckpt, uses_metadata)
    gallery_loader = DataLoader(gallery_dataset, batch_size=64, shuffle=False, num_workers=0)
    gallery_emb, gallery_labels = extract_embeddings(model, gallery_loader, device)

    # Aggregate per person
    person_emb, person_ids = aggregate_by_person(gallery_emb, gallery_labels, method=args.aggregation)
    print(f"   Gallery: {len(person_emb)} persons (aggregated from {len(gallery_emb)} crops)")

    # Build FAISS index
    embedding_dim = person_emb.shape[1]
    index = RetrievalIndex(dim=embedding_dim)
    label_to_id = {v: k for k, v in label_map.items()}
    person_id_strs = [label_to_id[lbl] for lbl in person_ids]
    index.add(person_emb, person_id_strs)
    print(f"   FAISS index: {len(index)} entries")

    # --- Build queries ---
    print(f"\n2. Loading queries from '{args.query_split}' split...")
    query_dataset = build_eval_dataset(cfg, args.query_split, label_map, ckpt, uses_metadata)
    query_loader = DataLoader(query_dataset, batch_size=64, shuffle=False, num_workers=0)
    query_emb, query_labels = extract_embeddings(model, query_loader, device)

    # Pick N random persons that have at least 2 teeth (so we can leave one out)
    unique_query_persons = np.unique(query_labels)
    eligible = [p for p in unique_query_persons if (query_labels == p).sum() >= 2]
    selected = random.sample(eligible, min(args.n_query, len(eligible)))
    print(f"   Selected {len(selected)} query persons")

    # --- Run queries ---
    print(f"\n3. Running queries (top-{args.top_k})...\n")
    successes = 0
    detail_results = []

    for i, person_label in enumerate(selected):
        person_str = label_to_id[person_label]
        person_short = person_str[:50]

        # Pick one tooth as query (forensic scenario: 1 unknown tooth)
        idx = np.where(query_labels == person_label)[0]
        chosen_idx = np.random.choice(idx)
        query_vec = query_emb[chosen_idx]

        sims, neighbor_ids = index.search(query_vec, k=args.top_k)

        is_match = [(nid == person_str) for nid in neighbor_ids]
        rank = next((j for j, m in enumerate(is_match) if m), None)
        success = rank == 0
        successes += int(success)

        print(f"--- Query {i+1}/{len(selected)} ---")
        print(f"True person: {person_short}")
        print(f"Top {args.top_k} matches:")
        for k, (nid, sim, m) in enumerate(zip(neighbor_ids, sims, is_match)):
            marker = "  ← MATCH" if m else ""
            print(f"  {k+1:2d}. sim={sim:.4f}  {nid[:50]}{marker}")

        if rank is None:
            print(f"  Correct person NOT in top-{args.top_k}")
        else:
            print(f"  Correct person at rank {rank+1}")
        print()

        detail_results.append({
            "query_person": person_str,
            "rank": rank if rank is not None else -1,
            "top_match": neighbor_ids[0],
            "top_sim": float(sims[0]),
            "true_match_sim": float(sims[rank]) if rank is not None else None,
        })

    print(f"=== Summary: {successes}/{len(selected)} correct at rank 1 "
          f"({100*successes/len(selected):.1f}%) ===")

    # Save details
    output_dir = Path(args.checkpoint).parent / "analysis" / "demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "demo_results.json", "w") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "query_split": args.query_split,
            "gallery_split": args.gallery_split,
            "aggregation": args.aggregation,
            "n_query": len(selected),
            "rank1_success_rate": successes / len(selected),
            "details": detail_results,
        }, f, indent=2)
    print(f"Saved: {output_dir}/demo_results.json")


if __name__ == "__main__":
    main()
