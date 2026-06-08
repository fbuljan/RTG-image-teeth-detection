"""
GT-crop person-retrieval anchor for Phase 8 comparison.

Re-runs the deployed Phase 5 embedder (or any embedder) through the modern Phase 8
bootstrap protocol on GT crops, so the resulting R1@n=16 + 95% CI lives on the
same codepath as `evaluate_pipeline.py`. This produces the "GT regression anchor"
that Phase 8.x experiments compare against, rather than the stale Phase 5 number
that was measured with only n_trials=5 and no CIs.

Protocol per n_query (matches `_evaluate_sweep_symmetric_paired` in evaluate_pipeline.py):
  - Eligible persons: those with >= n_query + 1 GT crops in the split
  - For each of n_trials random permutations, split each person's teeth into
    query (first n_query) and gallery (rest); aggregate with mean+L2; compute R1
  - Per-person R1 = mean across trials
  - 95% CI: B=1000 bootstrap resamples over persons (paired with seed)

Usage:
  python -m identification.evaluation.evaluate_gt_anchor \\
      --checkpoint identification/runs/embedding_fdi_init_v1/best.pt \\
      --split test \\
      --n-query-list 1 2 4 8 16 \\
      --n-trials 30 \\
      --output-dir identification/runs/phase5_gt_recheck
"""

import argparse
import functools
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from identification.evaluation.evaluate_embedding import (
    build_eval_dataset,
    extract_embeddings,
    load_checkpoint,
)
from identification.models.embedding_model import ToothEmbeddingModelWithMetadata


print = functools.partial(print, flush=True)


def _mean_pool(arr: np.ndarray) -> np.ndarray:
    """Mean + L2 normalise (matches evaluate_pipeline.py)."""
    v = arr.mean(axis=0)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def evaluate_sweep_with_ci(
    per_person_embs: dict,
    n_query_list: list,
    n_trials: int,
    seed: int,
    n_boot: int = 1000,
) -> list:
    rng = np.random.default_rng(seed)
    bootstrap_rng = np.random.default_rng(seed + 100_000)

    results = []
    for n_q in n_query_list:
        eligible = [pid for pid, e in per_person_embs.items() if len(e) >= n_q + 1]
        if len(eligible) < 5:
            results.append({"n_query": n_q, "n_persons": len(eligible), "skipped": True})
            continue

        n_eligible = len(eligible)
        match_r1 = np.zeros((n_trials, n_eligible), dtype=bool)
        match_r5 = np.zeros_like(match_r1)
        match_r10 = np.zeros_like(match_r1)
        ap_arr = np.zeros((n_trials, n_eligible), dtype=np.float64)

        for t in range(n_trials):
            queries, galleries = [], []
            for pid in eligible:
                arr = per_person_embs[pid]
                idx = rng.permutation(len(arr))
                q_idx = idx[:n_q]
                g_idx = idx[n_q:]
                queries.append(_mean_pool(arr[q_idx]))
                galleries.append(_mean_pool(arr[g_idx]))
            Q = np.stack(queries)
            G = np.stack(galleries)
            assert np.isfinite(Q).all() and np.isfinite(G).all(), \
                f"non-finite embeddings at n_q={n_q} trial={t}"
            sim = Q @ G.T
            assert np.isfinite(sim).all(), f"non-finite sim at n_q={n_q} trial={t}"
            ranked = np.argsort(-sim, axis=1)
            pids_arr = np.array(eligible)
            ranked_labels = pids_arr[ranked]
            mat = ranked_labels == pids_arr[:, None]

            match_r1[t] = mat[:, 0]
            match_r5[t] = mat[:, :5].any(axis=1)
            match_r10[t] = mat[:, :10].any(axis=1)
            first_pos = np.argmax(mat, axis=1)
            valid = mat.any(axis=1)
            ap_arr[t] = np.where(valid, 1.0 / (first_pos + 1), 0.0)

        per_person_r1 = match_r1.mean(axis=0)
        per_person_r5 = match_r5.mean(axis=0)
        per_person_ap = ap_arr.mean(axis=0)

        rank1 = float(per_person_r1.mean())
        rank5 = float(per_person_r5.mean())
        rank10 = float(match_r10.mean(axis=0).mean())
        mAP = float(per_person_ap.mean())

        boot_r1 = np.empty(n_boot)
        boot_r5 = np.empty(n_boot)
        for b in range(n_boot):
            sel = bootstrap_rng.integers(0, n_eligible, size=n_eligible)
            boot_r1[b] = per_person_r1[sel].mean()
            boot_r5[b] = per_person_r5[sel].mean()
        ci_low, ci_high = np.percentile(boot_r1, [2.5, 97.5])
        r5_ci_low, r5_ci_high = np.percentile(boot_r5, [2.5, 97.5])

        results.append({
            "n_query": n_q,
            "method": "mean",
            "n_persons": n_eligible,
            "n_trials": n_trials,
            "rank1_mean": rank1,
            "rank1_ci95_low": float(ci_low),
            "rank1_ci95_high": float(ci_high),
            "rank5_mean": rank5,
            "rank5_ci95_low": float(r5_ci_low),
            "rank5_ci95_high": float(r5_ci_high),
            "rank10_mean": rank10,
            "mAP_mean": mAP,
        })
        print(f"  n_q={n_q:3d}: R1={rank1:.4f} [{ci_low:.4f}, {ci_high:.4f}]  "
              f"R5={rank5:.4f}  mAP={mAP:.4f}  n={n_eligible}")
    return results


def main():
    parser = argparse.ArgumentParser(description="GT-crop person-retrieval anchor with bootstrap CIs")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--n-query-list", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--n-trials", type=int, default=30,
                        help="Random query/gallery permutations per n_query")
    parser.add_argument("--n-boot", type=int, default=1000,
                        help="Bootstrap iterations for person-level CI")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, cfg, label_map, ckpt = load_checkpoint(args.checkpoint, device)
    uses_metadata = isinstance(model, ToothEmbeddingModelWithMetadata)

    dataset = build_eval_dataset(cfg, args.split, label_map, ckpt, uses_metadata)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    print(f"Evaluating {args.split}: {len(dataset)} samples, {len(label_map)} persons total")

    print("Extracting embeddings...")
    embeddings, labels = extract_embeddings(model, loader, device)
    embeddings = np.asarray(embeddings)
    labels = np.asarray(labels)
    print(f"Embeddings: {embeddings.shape}, labels: {labels.shape}")

    # Group by person
    per_person_embs = {}
    for pid in np.unique(labels):
        per_person_embs[int(pid)] = embeddings[labels == pid]
    print(f"Persons in {args.split}: {len(per_person_embs)} "
          f"(median teeth/person={np.median([len(e) for e in per_person_embs.values()]):.0f})")

    print(f"\nSweeping with bootstrap CI (n_trials={args.n_trials}, n_boot={args.n_boot}):")
    results = evaluate_sweep_with_ci(
        per_person_embs,
        args.n_query_list,
        n_trials=args.n_trials,
        seed=args.seed,
        n_boot=args.n_boot,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "n_trials": args.n_trials,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "sweep": results,
    }
    out_path = output_dir / "gt_anchor.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: {out_path}")

    # Headline summary
    print("\n=== Headline ===")
    for r in results:
        if r.get("skipped"):
            print(f"  n={r['n_query']}: SKIPPED (only {r['n_persons']} eligible)")
        else:
            print(f"  n={r['n_query']:3d}: R1 = {r['rank1_mean']:.3f} "
                  f"[{r['rank1_ci95_low']:.3f}, {r['rank1_ci95_high']:.3f}]")


if __name__ == "__main__":
    main()
