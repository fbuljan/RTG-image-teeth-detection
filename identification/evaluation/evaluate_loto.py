"""
Leave-One-Tooth-Out (LOTO) per-FDI evaluation.

Pure post-hoc consumer: loads cached per-crop embeddings + stage_ac metadata produced
by `evaluate_pipeline.py` and the deployed FAISS registry. For each FDI with n >= 30
crops, queries each crop alone against the full 1178-person registry, reporting R1/R5
with person-stratified bootstrap CIs.

Observation-only: no Pass/Fail criterion. The pre-registered adversarial
honesty rule:
  If max(per_fdi_R1_mean) - min(per_fdi_R1_mean) <= 0.05, the figure adds nothing
  beyond the pooled single-tooth R1, and the thesis text says so explicitly.

Reuses the open-set calibration caches; no embedder/YOLO inference happens here.

Usage:
  PYTHONPATH=. python identification/evaluation/evaluate_loto.py \\
    --cache-dir identification/runs/phase8_deployed_yolo_reg/cache \\
    --registry identification/registry_ensemble_yolo/embedding_fdi_init_v1 \\
    --out-dir identification/runs/phase8_loto \\
    --rotation upright \\
    --seed 0
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import faiss  # noqa: F401  (RetrievalIndex imports it lazily through faiss-cpu)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from identification.models.retrieval_index import RetrievalIndex  # noqa: E402

MIN_N_PER_FDI = 30
N_BOOTSTRAP = 1000
HONESTY_THRESHOLD_PP = 0.05  # spread <= 5pp → figure adds nothing
K_TOP = 5

# FDI quadrant layout for heatmap (ISO 3950):
#   18 17 16 15 14 13 12 11 | 21 22 23 24 25 26 27 28     ← upper (Q1 right ← left, Q2 left → right)
#   48 47 46 45 44 43 42 41 | 31 32 33 34 35 36 37 38     ← lower (Q4 right ← left, Q3 left → right)
# Heatmap layout (left-to-right anatomical, top=upper jaw, bottom=lower jaw):
UPPER_FDIS = ["18", "17", "16", "15", "14", "13", "12", "11",
              "21", "22", "23", "24", "25", "26", "27", "28"]
LOWER_FDIS = ["48", "47", "46", "45", "44", "43", "42", "41",
              "31", "32", "33", "34", "35", "36", "37", "38"]


# ---------------------------------------------------------------------------
# Cache loading
# ---------------------------------------------------------------------------

_ROT_RE = re.compile(r"__rot([+-]\d+)__")
_EMB_SUFFIX_RE = re.compile(r"__emb[0-9a-f]+\.npy$")


def _is_upright(name: str) -> bool:
    m = _ROT_RE.search(name)
    return bool(m and m.group(1) == "+00000")


def _stage_ac_key(filename: str) -> str:
    """Strip the .json extension to get the cache key."""
    assert filename.endswith(".json"), filename
    return filename[:-5]


def _embedding_key(filename: str) -> str:
    """Strip the __emb<hash>.npy suffix to get the cache key."""
    return _EMB_SUFFIX_RE.sub("", filename)


def load_cache(cache_dir: Path, rotation: str) -> list[dict]:
    """Load (pid, fdi, embedding) records from the cache directory.

    rotation: "upright" → only __rot+00000__; "rotated" → all non-upright; "both" → all.
    """
    stage_ac_dir = cache_dir / "stage_ac"
    emb_dir = cache_dir / "embeddings"
    if not stage_ac_dir.is_dir() or not emb_dir.is_dir():
        raise FileNotFoundError(f"missing stage_ac/ or embeddings/ under {cache_dir}")

    # Map cache_key → emb_path
    emb_by_key: dict[str, Path] = {}
    for p in emb_dir.iterdir():
        if p.suffix != ".npy":
            continue
        emb_by_key[_embedding_key(p.name)] = p

    records: list[dict] = []
    for sj in sorted(stage_ac_dir.iterdir()):
        if sj.suffix != ".json":
            continue
        if rotation == "upright" and not _is_upright(sj.name):
            continue
        if rotation == "rotated" and _is_upright(sj.name):
            continue
        key = _stage_ac_key(sj.name)
        emb_path = emb_by_key.get(key)
        if emb_path is None:
            print(f"  [warn] no embedding for stage_ac key {key}; skipping")
            continue
        meta = json.load(open(sj))
        emb = np.load(emb_path)
        pid = meta["person_id"]
        fdi_labels = meta["fdi_labels"]
        assert len(fdi_labels) == emb.shape[0], (
            f"fdi_labels len ({len(fdi_labels)}) != emb rows ({emb.shape[0]}) for {key}"
        )
        for i, fdi in enumerate(fdi_labels):
            records.append({
                "pid": pid,
                "fdi": str(fdi),
                "emb": emb[i].astype(np.float32),
                "cache_key": key,
            })
    return records


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return v.astype(np.float32)
    return (v / n).astype(np.float32)


def query_against_registry(
    records: list[dict],
    registry: RetrievalIndex,
) -> list[dict]:
    """For each record, query the single L2-normalized crop embedding against the
    full registry and record R1/R5 hit + top1_pid.
    """
    out: list[dict] = []
    for r in records:
        q = _l2_normalize(r["emb"])
        sims, ids = registry.search(q, k=K_TOP)
        out.append({
            "pid": r["pid"],
            "fdi": r["fdi"],
            "top1_pid": ids[0] if ids else None,
            "topk_pids": list(ids),
            "sim_top1": float(sims[0]) if len(sims) else float("nan"),
            "r1": int(ids[0] == r["pid"]) if ids else 0,
            "r5": int(r["pid"] in ids[:K_TOP]) if ids else 0,
        })
    return out


# ---------------------------------------------------------------------------
# Per-FDI aggregation + bootstrap
# ---------------------------------------------------------------------------

def per_fdi_table(
    queried: list[dict],
    rng: np.random.Generator,
) -> dict:
    """Group queried records by FDI, filter n>=MIN_N_PER_FDI, person-bootstrap CI.

    Bootstrap protocol: for each FDI, resample the set of (pid, fdi) records by
    PIDs-with-replacement, taking ALL crops of sampled PIDs at that FDI. This is
    person-stratified bootstrap — the unit of independence is the person, not the
    crop.

    Returns: {
      "global_r1": ..., "global_r5": ..., "global_n": ..., "global_n_persons": ...,
      "per_fdi": [
        {"fdi": "11", "n": 156, "n_persons": 156, "r1_mean": .., "r1_ci_low": .., ...},
        ...
      ],
      "figure_adds_nothing": bool, "spread_pp": float,
    }
    """
    # Global (pooled across all FDIs that survived n>=MIN, but include every
    # qualifying FDI's records so we compare like-for-like).
    by_fdi: dict[str, list[dict]] = defaultdict(list)
    for q in queried:
        by_fdi[q["fdi"]].append(q)

    per_fdi_rows: list[dict] = []
    for fdi in sorted(by_fdi.keys()):
        items = by_fdi[fdi]
        if len(items) < MIN_N_PER_FDI:
            print(f"  [drop] FDI {fdi}: n={len(items)} < {MIN_N_PER_FDI}, suppressed")
            continue
        # Group by pid for bootstrap.
        by_pid: dict[str, list[dict]] = defaultdict(list)
        for it in items:
            by_pid[it["pid"]].append(it)
        pids = list(by_pid.keys())

        # Point estimate (pooled across crops).
        r1_mean = float(np.mean([it["r1"] for it in items]))
        r5_mean = float(np.mean([it["r5"] for it in items]))
        sim_med = float(np.median([it["sim_top1"] for it in items]))

        # Person-stratified bootstrap.
        boot_r1, boot_r5 = [], []
        for _ in range(N_BOOTSTRAP):
            sampled_pids = rng.choice(pids, size=len(pids), replace=True)
            bag = []
            for p in sampled_pids:
                bag.extend(by_pid[p])
            boot_r1.append(np.mean([b["r1"] for b in bag]))
            boot_r5.append(np.mean([b["r5"] for b in bag]))
        ci_r1 = np.percentile(boot_r1, [2.5, 97.5])
        ci_r5 = np.percentile(boot_r5, [2.5, 97.5])

        per_fdi_rows.append({
            "fdi": fdi,
            "n": len(items),
            "n_persons": len(pids),
            "r1_mean": r1_mean,
            "r1_ci_low": float(ci_r1[0]),
            "r1_ci_high": float(ci_r1[1]),
            "r5_mean": r5_mean,
            "r5_ci_low": float(ci_r5[0]),
            "r5_ci_high": float(ci_r5[1]),
            "sim_top1_median": sim_med,
        })

    # Pooled global: all crops from all *qualifying* FDIs (consistent with the
    # per-FDI rows below it).
    qualifying_items: list[dict] = []
    for r in per_fdi_rows:
        qualifying_items.extend(by_fdi[r["fdi"]])
    global_r1 = float(np.mean([q["r1"] for q in qualifying_items])) if qualifying_items else float("nan")
    global_r5 = float(np.mean([q["r5"] for q in qualifying_items])) if qualifying_items else float("nan")

    # Person-stratified global bootstrap.
    global_by_pid: dict[str, list[dict]] = defaultdict(list)
    for it in qualifying_items:
        global_by_pid[it["pid"]].append(it)
    global_pids = list(global_by_pid.keys())
    boot_g_r1, boot_g_r5 = [], []
    for _ in range(N_BOOTSTRAP):
        sampled = rng.choice(global_pids, size=len(global_pids), replace=True)
        bag = []
        for p in sampled:
            bag.extend(global_by_pid[p])
        boot_g_r1.append(np.mean([b["r1"] for b in bag]))
        boot_g_r5.append(np.mean([b["r5"] for b in bag]))
    g_ci_r1 = np.percentile(boot_g_r1, [2.5, 97.5])
    g_ci_r5 = np.percentile(boot_g_r5, [2.5, 97.5])

    # Adversarial honesty rules.
    r1_vals = [r["r1_mean"] for r in per_fdi_rows]
    spread = float(max(r1_vals) - min(r1_vals)) if r1_vals else 0.0
    adds_nothing_by_spread = spread <= HONESTY_THRESHOLD_PP

    # CI-overlap rule: an FDI is a "real outlier" iff its 95% CI does not overlap
    # the global 95% CI. If <=10% of FDIs are real outliers, the per-FDI ranking
    # is noise within the same anatomical class.
    g_lo, g_hi = float(g_ci_r1[0]), float(g_ci_r1[1])
    permanent_fdi_re = re.compile(r"^[1-4][1-8]$")  # FDI 11..48 ⊂ permanent dentition
    n_overlap = 0
    n_real_outliers = 0
    permanent_overlap = 0
    permanent_total = 0
    for r in per_fdi_rows:
        overlaps_global = not (r["r1_ci_high"] < g_lo or r["r1_ci_low"] > g_hi)
        r["ci_overlaps_global"] = overlaps_global
        r["is_deciduous"] = not bool(permanent_fdi_re.match(r["fdi"]))
        if overlaps_global:
            n_overlap += 1
        else:
            n_real_outliers += 1
        if not r["is_deciduous"]:
            permanent_total += 1
            if overlaps_global:
                permanent_overlap += 1

    # The figure adds nothing if: spread ≤ 5pp OR permanent dentition CIs all overlap.
    permanent_fully_overlap = (permanent_overlap == permanent_total) and permanent_total > 0

    return {
        "global_r1_mean": global_r1,
        "global_r1_ci_low": g_lo,
        "global_r1_ci_high": g_hi,
        "global_r5_mean": global_r5,
        "global_r5_ci_low": float(g_ci_r5[0]),
        "global_r5_ci_high": float(g_ci_r5[1]),
        "global_n_crops": len(qualifying_items),
        "global_n_persons": len(global_pids),
        "per_fdi": per_fdi_rows,
        "n_fdi_qualifying": len(per_fdi_rows),
        "n_fdi_dropped_low_n": len(by_fdi) - len(per_fdi_rows),
        "spread_pp": spread,
        "honesty_threshold_pp": HONESTY_THRESHOLD_PP,
        "n_fdi_ci_overlaps_global": n_overlap,
        "n_fdi_real_outliers": n_real_outliers,
        "permanent_overlap_ratio": permanent_overlap / permanent_total if permanent_total else float("nan"),
        "permanent_fully_overlap": permanent_fully_overlap,
        "figure_adds_nothing_by_spread": adds_nothing_by_spread,
        "figure_adds_nothing_by_overlap": permanent_fully_overlap,
        # Combined verdict: either rule firing is enough for the honesty flag.
        "figure_adds_nothing": adds_nothing_by_spread or permanent_fully_overlap,
    }


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

def plot_heatmap(table: dict, out_path: Path, title_suffix: str = "") -> None:
    """2-row × 16-col heatmap arranged anatomically (upper/lower jaw).

    Annotation: R1 (top), n (bottom).
    """
    by_fdi = {r["fdi"]: r for r in table["per_fdi"]}
    rows = [UPPER_FDIS, LOWER_FDIS]
    grid = np.full((2, 16), np.nan)
    annot = [[""] * 16 for _ in range(2)]
    for ri, row in enumerate(rows):
        for ci, fdi in enumerate(row):
            r = by_fdi.get(fdi)
            if r is None:
                continue
            grid[ri, ci] = r["r1_mean"]
            annot[ri][ci] = f"{r['r1_mean']:.2f}\nn={r['n']}"

    fig, ax = plt.subplots(figsize=(14, 4.2))
    im = ax.imshow(grid, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(16))
    ax.set_xticklabels([f"{UPPER_FDIS[i]}\n{LOWER_FDIS[i]}" for i in range(16)], fontsize=8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Upper", "Lower"])
    for ri in range(2):
        for ci in range(16):
            if not np.isnan(grid[ri, ci]):
                colour = "white" if grid[ri, ci] < 0.5 else "black"
                ax.text(ci, ri, annot[ri][ci], ha="center", va="center",
                        fontsize=7, color=colour)
    g_r1 = table["global_r1_mean"]
    g_lo = table["global_r1_ci_low"]
    g_hi = table["global_r1_ci_high"]
    spread = table["spread_pp"]
    if table["figure_adds_nothing"]:
        verdict = "figure adds little (permanent CIs overlap global)" if table["permanent_fully_overlap"] \
                  else "figure adds little (spread ≤ 5pp)"
    else:
        verdict = f"spread {spread*100:.1f}pp; {table['n_fdi_real_outliers']} real outliers"
    ax.set_title(
        f"Per-FDI single-tooth R1 (LOTO){title_suffix}\n"
        f"Global pooled R1 = {g_r1:.3f} [{g_lo:.3f}, {g_hi:.3f}] · "
        f"{table['n_fdi_qualifying']} FDIs · {verdict}",
        fontsize=10,
    )
    fig.colorbar(im, ax=ax, label="single-tooth R1")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", type=Path, required=True,
                   help="Open-set calibration cache root (contains stage_ac/ and embeddings/)")
    p.add_argument("--registry", type=Path, required=True,
                   help="Registry path stem (e.g. identification/registry_ensemble_yolo/embedding_fdi_init_v1) — looks for index.faiss + index.ids.json")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--rotation", choices=["upright", "rotated", "both"], default="upright")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _registry_stem(registry_arg: Path) -> Path:
    """The registry argument can be the dir containing index.faiss, or the dir itself
    used as a stem. RetrievalIndex.load expects a stem (path without .faiss/.ids.json),
    so resolve to {dir}/index if a directory was passed.
    """
    if registry_arg.is_dir():
        return registry_arg / "index"
    return registry_arg


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"[loto] cache:    {args.cache_dir}")
    print(f"[loto] registry: {args.registry}")
    print(f"[loto] rotation: {args.rotation}")
    print(f"[loto] seed:     {args.seed}")

    # 1. Load cache
    records = load_cache(args.cache_dir, args.rotation)
    n_pids = len({r["pid"] for r in records})
    print(f"[loto] loaded {len(records)} crops from {n_pids} persons "
          f"({args.rotation} only)")

    # 2. Load registry
    stem = _registry_stem(args.registry)
    registry = RetrievalIndex.load(str(stem), dim=records[0]["emb"].shape[0])
    print(f"[loto] registry size: {len(registry)} persons")

    # 3. Single-tooth retrieval (full registry, k=5)
    queried = query_against_registry(records, registry)

    # 4. Per-FDI table + bootstrap CI
    table = per_fdi_table(queried, rng)

    # 5. Persist outputs
    out_json = args.out_dir / f"per_fdi_table_{args.rotation}.json"
    with open(out_json, "w") as f:
        json.dump({
            "args": {
                "cache_dir": str(args.cache_dir),
                "registry": str(args.registry),
                "rotation": args.rotation,
                "seed": args.seed,
                "min_n_per_fdi": MIN_N_PER_FDI,
                "n_bootstrap": N_BOOTSTRAP,
                "k_top": K_TOP,
                "honesty_threshold_pp": HONESTY_THRESHOLD_PP,
            },
            "n_records": len(queried),
            "n_persons_queried": n_pids,
            "registry_size": len(registry),
            **table,
        }, f, indent=2)
    print(f"[loto] wrote {out_json}")

    out_png = args.out_dir / f"heatmap_{args.rotation}.png"
    plot_heatmap(table, out_png, title_suffix=f" · {args.rotation}")
    print(f"[loto] wrote {out_png}")

    # 6. Verdict / text summary
    print()
    print(f"[loto] === results ({args.rotation}) ===")
    print(f"  global single-tooth R1 = {table['global_r1_mean']:.3f} "
          f"[{table['global_r1_ci_low']:.3f}, {table['global_r1_ci_high']:.3f}]")
    print(f"  global single-tooth R5 = {table['global_r5_mean']:.3f} "
          f"[{table['global_r5_ci_low']:.3f}, {table['global_r5_ci_high']:.3f}]")
    print(f"  {table['n_fdi_qualifying']} FDIs qualified (n>={MIN_N_PER_FDI}); "
          f"{table['n_fdi_dropped_low_n']} dropped for low support")
    print(f"  spread (max-min R1) = {table['spread_pp']*100:.2f}pp")
    print(f"  FDIs with CI overlapping global:    "
          f"{table['n_fdi_ci_overlaps_global']}/{table['n_fdi_qualifying']}")
    print(f"  Real outliers (CI disjoint from global): {table['n_fdi_real_outliers']}")
    print(f"  Permanent-dentition overlap ratio:  "
          f"{table['permanent_overlap_ratio']*100:.0f}% "
          f"({'all permanent CIs overlap global' if table['permanent_fully_overlap'] else 'some permanent CIs disjoint'})")
    if table["figure_adds_nothing"]:
        print(f"  ADVERSARIAL HONESTY: figure adds little.")
        if table["figure_adds_nothing_by_spread"]:
            print(f"    - spread ≤ {HONESTY_THRESHOLD_PP*100:.0f}pp")
        if table["figure_adds_nothing_by_overlap"]:
            print(f"    - all permanent-dentition (FDI 11-48) CIs overlap the global CI")
        print("  Thesis text recommendation: report the global pooled number and a")
        print("  one-line null finding; if any deciduous FDIs are real outliers,")
        print("  attribute to dataset composition rather than embedder weakness.")
    else:
        print(f"  Spread is {table['spread_pp']*100:.2f}pp > 5pp AND permanent CIs")
        print("  do NOT all overlap. Per-FDI variation may be real; investigate which FDIs lag/lead.")
    print()
    for r in table["per_fdi"]:
        flag = "  " if r["ci_overlaps_global"] else "* "  # * = real outlier
        decid = "d" if r["is_deciduous"] else " "
        print(f"  {flag}{decid} FDI {r['fdi']:>2}: n={r['n']:>3} ({r['n_persons']} persons)  "
              f"R1={r['r1_mean']:.3f} [{r['r1_ci_low']:.3f}, {r['r1_ci_high']:.3f}]  "
              f"R5={r['r5_mean']:.3f}  sim_med={r['sim_top1_median']:.3f}")
    print("  (legend: * = CI disjoint from global; d = deciduous)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
