"""
Adult-deployment-gap subset eval (post-hoc).

Re-aggregates the canonical baseline run (`phase8_deployed_yolo_reg/yolo_eval.json`)
across pre-registered subsets to quantify the pediatric→adult deployment gap:

  1. all-permanent test PIDs (zero deciduous teeth in YOLO detection)
  2. per-age-bucket R1 within the 6-18y range
  3. per-sex R1 (descriptive, for the Discussion chapter)

Pre-registered honesty rule:
  If the all-permanent subset R1 collapses by ≥10pp vs full-test R1, the abstract
  and introduction must say so. (The actual outcome may go either way.)

No model loading, no FAISS work, no YOLO inference — all numbers come from the
existing per-person R1 array in the canonical baseline JSON. The script:

  - Loads `per_person_r1` from `phase8_deployed_yolo_reg/yolo_eval.json` (full-reg n=16).
  - Loads age/sex from `identification/data/manifest.csv` (one row per crop, per-PID
    constant for age and sex).
  - Loads YOLO-detected FDI labels from cached stage_ac JSONs to compute the
    "all-permanent" mask (FDI 51-89 = deciduous).
  - Person-stratified bootstrap CI for each subset (resample PIDs with replacement,
    1000 iters, percentile CI95).

Usage:
  PYTHONPATH=. python identification/evaluation/evaluate_permanent_subset.py \\
    --baseline identification/runs/phase8_deployed_yolo_reg/yolo_eval.json \\
    --stage-ac identification/runs/phase8_deployed_yolo_reg/cache/stage_ac \\
    --manifest identification/data/manifest.csv \\
    --out-dir identification/runs/phase8_permanent \\
    --seed 0
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

N_BOOTSTRAP = 1000
HONESTY_THRESHOLD_PP = 0.10  # 10pp collapse → abstract caveat
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_baseline_per_person_r1(baseline_json: Path, n_query: int = 16) -> tuple[list[str], np.ndarray]:
    """Pull (pids, per_person_r1) from the canonical n=16 full-registry sweep."""
    d = json.load(open(baseline_json))
    sweep = d.get("sweep_full_registry", [])
    target = next((s for s in sweep if s.get("n_query") == n_query and not s.get("skipped")), None)
    if target is None:
        raise ValueError(f"no full-registry sweep at n_query={n_query} in {baseline_json}")
    pids = list(target["pids"])
    r1 = np.array(target["per_person_r1"], dtype=np.float64)
    assert len(pids) == len(r1), f"pids/r1 length mismatch: {len(pids)} vs {len(r1)}"
    return pids, r1


def load_test_metadata(manifest_csv: Path) -> dict[str, dict]:
    """Return {pid: {'age': float, 'sex': str}} for test-split PIDs."""
    meta: dict[str, dict] = {}
    with open(manifest_csv) as f:
        for row in csv.DictReader(f):
            if row.get("split") != "test":
                continue
            pid = row["person_id"]
            if pid not in meta:
                meta[pid] = {"age": float(row["age"]), "sex": row["sex"]}
    return meta


def load_yolo_fdis(stage_ac_dir: Path) -> dict[str, list[str]]:
    """Return {pid: [fdi_label, ...]} from cached upright stage_ac JSONs."""
    out: dict[str, list[str]] = {}
    for f in sorted(stage_ac_dir.iterdir()):
        if f.suffix != ".json":
            continue
        if "__rot+00000__" not in f.name:
            continue
        d = json.load(open(f))
        out[d["person_id"]] = [str(x) for x in d["fdi_labels"]]
    return out


def is_deciduous_fdi(fdi: str) -> bool:
    """Deciduous (primary) dentition has FDI quadrants 5/6/7/8 (51-85)."""
    return bool(fdi) and fdi[0] in "5678"


# ---------------------------------------------------------------------------
# Subset stats
# ---------------------------------------------------------------------------

def subset_r1_with_ci(
    r1: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    label: str = "",
) -> dict:
    sub = r1[mask]
    if len(sub) == 0:
        return {"label": label, "n": 0, "r1_mean": float("nan"),
                "r1_ci_low": float("nan"), "r1_ci_high": float("nan")}
    point = float(np.mean(sub))
    boots = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(sub), size=len(sub))
        boots.append(np.mean(sub[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "label": label,
        "n": int(len(sub)),
        "r1_mean": point,
        "r1_ci_low": float(lo),
        "r1_ci_high": float(hi),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_age_buckets(
    rows: list[dict],
    full_test_r1: float,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = np.arange(len(rows))
    means = [r["r1_mean"] for r in rows]
    errs_low = [r["r1_mean"] - r["r1_ci_low"] for r in rows]
    errs_high = [r["r1_ci_high"] - r["r1_mean"] for r in rows]
    ax.bar(xs, means, yerr=[errs_low, errs_high], capsize=4, color="steelblue", alpha=0.85)
    ax.axhline(full_test_r1, color="red", linestyle="--", linewidth=1,
               label=f"full-test R1 = {full_test_r1:.3f}")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{r['label']}\nn={r['n']}" for r in rows], fontsize=9)
    ax.set_ylabel("Person retrieval R1 @ n=16 (full-reg, deployment-aligned)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-age-bucket R1 with person-bootstrap 95% CI")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", type=Path, required=True,
                   help="Path to phase8_deployed_yolo_reg/yolo_eval.json (canonical n=16 sweep)")
    p.add_argument("--stage-ac", type=Path, required=True,
                   help="Cached stage_ac dir under phase8_deployed_yolo_reg/cache")
    p.add_argument("--manifest", type=Path, required=True,
                   help="identification/data/manifest.csv (provides age + sex per PID)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"[subset] baseline:  {args.baseline}")
    print(f"[subset] stage_ac:  {args.stage_ac}")
    print(f"[subset] manifest:  {args.manifest}")
    print(f"[subset] seed:      {args.seed}")
    print()

    pids, r1 = load_baseline_per_person_r1(args.baseline)
    print(f"[subset] loaded {len(pids)} test PIDs with n>=16 detected teeth from baseline JSON")
    print(f"[subset] full-test R1 (n=16) = {r1.mean():.4f}  (sanity-check against 82.6% headline)")

    meta = load_test_metadata(args.manifest)
    yolo_fdis = load_yolo_fdis(args.stage_ac)
    missing_meta = [p for p in pids if p not in meta]
    missing_fdis = [p for p in pids if p not in yolo_fdis]
    if missing_meta:
        print(f"  [warn] {len(missing_meta)} PIDs missing age/sex metadata")
    if missing_fdis:
        print(f"  [warn] {len(missing_fdis)} PIDs missing YOLO FDI cache")
    print()

    # Subset 1: all-permanent (zero deciduous teeth in YOLO output)
    n_decid = np.array([
        sum(1 for f in yolo_fdis.get(p, []) if is_deciduous_fdi(f))
        for p in pids
    ])
    ap_mask = n_decid == 0
    ad_mask = ~ap_mask

    full_test = subset_r1_with_ci(r1, np.ones_like(r1, dtype=bool), rng, label="full test")
    all_permanent = subset_r1_with_ci(r1, ap_mask, rng, label="all-permanent")
    any_deciduous = subset_r1_with_ci(r1, ad_mask, rng, label="any-deciduous")

    # Subset 2: per-age buckets
    age = np.array([meta.get(p, {}).get("age", -1.0) for p in pids])
    buckets = [("6-9", 6, 10), ("10-12", 10, 13), ("13-15", 13, 16), ("16-18", 16, 19)]
    age_rows: list[dict] = []
    for name, lo, hi in buckets:
        m = (age >= lo) & (age < hi)
        age_rows.append(subset_r1_with_ci(r1, m, rng, label=name))

    # Subset 3: per-sex (descriptive)
    sex = np.array([meta.get(p, {}).get("sex", "?") for p in pids])
    sex_rows = [
        subset_r1_with_ci(r1, sex == "male", rng, label="male"),
        subset_r1_with_ci(r1, sex == "female", rng, label="female"),
    ]

    # Honesty rule: collapse vs full-test by ≥10pp triggers abstract caveat.
    delta = all_permanent["r1_mean"] - full_test["r1_mean"]
    if delta <= -HONESTY_THRESHOLD_PP:
        verdict = "COLLAPSE"
        verdict_text = (f"all-permanent R1 = {all_permanent['r1_mean']:.3f} is "
                        f"{abs(delta)*100:.1f}pp LOWER than full-test "
                        f"{full_test['r1_mean']:.3f} (≥10pp threshold) — "
                        "abstract and introduction MUST report this.")
    elif delta < 0:
        verdict = "MILD-COLLAPSE"
        verdict_text = (f"all-permanent R1 = {all_permanent['r1_mean']:.3f} is "
                        f"{abs(delta)*100:.1f}pp LOWER than full-test "
                        f"{full_test['r1_mean']:.3f} (< 10pp threshold). "
                        "Discussion-chapter caveat sufficient.")
    else:
        verdict = "NO-COLLAPSE-OR-LIFT"
        verdict_text = (f"all-permanent R1 = {all_permanent['r1_mean']:.3f} is "
                        f"{delta*100:.1f}pp HIGHER than full-test "
                        f"{full_test['r1_mean']:.3f}. Adult-proxy subset performs "
                        "BETTER than full test. The deployment-gap concern from the "
                        "pre-registered honesty rule does not fire in this direction; "
                        "however, this is in-distribution adult-proxy testing — true "
                        "out-of-distribution adult data is still untested.")

    # Persist
    result = {
        "args": {
            "baseline": str(args.baseline),
            "stage_ac": str(args.stage_ac),
            "manifest": str(args.manifest),
            "seed": args.seed,
            "n_bootstrap": N_BOOTSTRAP,
            "honesty_threshold_pp": HONESTY_THRESHOLD_PP,
        },
        "full_test": full_test,
        "all_permanent": all_permanent,
        "any_deciduous": any_deciduous,
        "age_buckets": age_rows,
        "per_sex": sex_rows,
        "delta_all_permanent_vs_full_test": delta,
        "honesty_rule_verdict": verdict,
        "honesty_rule_text": verdict_text,
    }
    out_json = args.out_dir / "subset_eval.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[subset] wrote {out_json}")

    out_png = args.out_dir / "age_buckets.png"
    plot_age_buckets(age_rows, full_test["r1_mean"], out_png)
    print(f"[subset] wrote {out_png}")

    print()
    print(f"[subset] === results ===")
    print(f"  full-test      : n={full_test['n']:>3}  R1={full_test['r1_mean']:.4f} "
          f"[{full_test['r1_ci_low']:.3f}, {full_test['r1_ci_high']:.3f}]")
    print(f"  all-permanent  : n={all_permanent['n']:>3}  R1={all_permanent['r1_mean']:.4f} "
          f"[{all_permanent['r1_ci_low']:.3f}, {all_permanent['r1_ci_high']:.3f}]  "
          f"(Δ vs full = {(all_permanent['r1_mean']-full_test['r1_mean'])*100:+.2f}pp)")
    print(f"  any-deciduous  : n={any_deciduous['n']:>3}  R1={any_deciduous['r1_mean']:.4f} "
          f"[{any_deciduous['r1_ci_low']:.3f}, {any_deciduous['r1_ci_high']:.3f}]")
    print()
    print("  Per-age R1:")
    for r in age_rows:
        print(f"    {r['label']:>5}: n={r['n']:>3}  R1={r['r1_mean']:.4f} "
              f"[{r['r1_ci_low']:.3f}, {r['r1_ci_high']:.3f}]")
    print()
    print("  Per-sex R1 (descriptive):")
    for r in sex_rows:
        print(f"    {r['label']:>6}: n={r['n']:>3}  R1={r['r1_mean']:.4f} "
              f"[{r['r1_ci_low']:.3f}, {r['r1_ci_high']:.3f}]")
    print()
    print(f"[subset] HONESTY VERDICT: {verdict}")
    print(f"  {verdict_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
