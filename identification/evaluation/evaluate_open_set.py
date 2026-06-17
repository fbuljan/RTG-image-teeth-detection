"""
Open-set detection (post-hoc consumer of heldout_enrol JSON).

Pre-registered protocol:

1. Read val heldout records (calibration source); read test heldout records (immutable
   evaluation target); optionally read test rotated heldout records (adversarial slice).
2. Compute five open-set features per record: sim_top1, gap_top12, mean_top5_sim,
   gap_top1_vs_mean5, vote_consistency. Records lacking the top-5 features (an older
   schema) are rejected here with a clear error.
3. Z-score features using val in-registry mean/std ONLY (val + in_registry subset).
   Persist (mean, std) per feature.
4. Calibration on val:
     a. Fit L2-regularised logistic regression on z-scored features.
     b. Compute val-CV AUROC via person-stratified bootstrap.
     c. Fallback: if val-CV AUROC(5-feature LR) < val-CV AUROC(sim_top1 only) + 0.03,
        lock score to sim_top1 only (no weights). Mirrors the weighted-aggregation
        weight-collapse defence.
     d. Pick operating-point threshold giving ≥70% OOS rejection on val.
5. Lock weights + threshold + z-score stats to <out_dir>/phase8_open_set_calibration.json.
6. Test evaluation:
     a. Apply locked transformation to test records → scalar score per record.
     b. Test AUROC + 95% person-bootstrap CI: resample 178 PIDs with replacement;
        for each sampled PID, take ALL its trial-records; compute AUROC; 1000 iters,
        percentile CI.
     c. At locked threshold: report TPR (rejection rate on OOS) + FRR (false-reject
        rate on in-registry).
7. Same pipeline on rotated test records (if provided) → rotated AUROC + bootstrap CI.
8. Pass criterion (pre-registered, see plan):
     - Clean test AUROC ≥ 0.72 (point estimate)
     - Rotated test AUROC ≥ 0.60 (point estimate)
     - Weak-positive zone: clean ∈ [0.65, 0.72)
     - Fail: clean < 0.65

Usage:
  PYTHONPATH=. python identification/evaluation/evaluate_open_set.py \\
    --calibrate identification/runs/phase8_open_set_val/heldout_enrol.json \\
    --evaluate  identification/runs/phase8_deployed_yolo_reg/heldout_enrol.json \\
    --evaluate-rotated identification/runs/phase8_deployed_yolo_reg/heldout_enrol_rotated.json \\
    --out-dir   identification/runs/phase8_open_set
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_NAMES = [
    "sim_top1",
    "gap_top12",
    "mean_top5_sim",
    "gap_top1_vs_mean5",
    "vote_consistency",
]

# Pre-registered thresholds
PASS_CLEAN_AUROC = 0.72
PASS_ROTATED_AUROC = 0.60
WEAK_POSITIVE_FLOOR = 0.65
FALLBACK_LIFT_THRESHOLD = 0.03  # if LR < sim-only + 0.03, lock to sim-only
TPR_OPERATING_POINT = 0.70  # ≥70% OOS rejection at the chosen threshold


def _load_records(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    if data.get("skipped"):
        raise RuntimeError(f"{path} reports skipped=True: {data.get('reason')}")
    records = data.get("records", [])
    if not records:
        raise RuntimeError(f"{path} has no records")
    missing = [f for f in FEATURE_NAMES if f not in records[0]]
    if missing:
        raise RuntimeError(
            f"{path} is missing required open-set features: {missing}. "
            f"Re-run evaluate_pipeline.py (current schema includes top-5 features)."
        )
    return records


def _extract_features(records: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Returns (X [n,5], y [n] in {0=oos, 1=in_registry}, pids [n])."""
    X = np.zeros((len(records), len(FEATURE_NAMES)), dtype=np.float64)
    y = np.zeros(len(records), dtype=np.int32)
    pids: list[str] = []
    for i, r in enumerate(records):
        for j, fname in enumerate(FEATURE_NAMES):
            X[i, j] = float(r[fname])
        y[i] = 1 if r["label"] == "in_registry" else 0
        pids.append(str(r["pid"]))
    return X, y, pids


def _zscore_fit(X_in_registry: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit z-score on the in_registry subset of val (the 'normal' class).
    Locked-in stats used for all downstream transformations."""
    mu = X_in_registry.mean(axis=0)
    sd = X_in_registry.std(axis=0, ddof=0)
    sd = np.where(sd < 1e-9, 1.0, sd)  # guard against zero-variance feature
    return mu, sd


def _zscore_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / sd


def _person_bootstrap_auroc(
    y: np.ndarray,
    scores: np.ndarray,
    pids: list[str],
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Person-stratified bootstrap CI on AUROC.
    Resample PIDs with replacement; take ALL records per sampled PID; compute AUROC.
    """
    rng = np.random.default_rng(seed)
    unique_pids = sorted(set(pids))
    n_pids = len(unique_pids)
    pid_to_idxs: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(pids):
        pid_to_idxs[p].append(i)

    point = float(roc_auc_score(y, scores))
    boots = np.empty(n_boot, dtype=np.float64)
    n_degenerate = 0
    for b in range(n_boot):
        sampled = rng.choice(n_pids, size=n_pids, replace=True)
        idxs: list[int] = []
        for s in sampled:
            idxs.extend(pid_to_idxs[unique_pids[s]])
        y_b = y[idxs]
        s_b = scores[idxs]
        if len(set(y_b.tolist())) < 2:
            # Degenerate sample (all one class) — record sentinel and skip.
            n_degenerate += 1
            boots[b] = np.nan
            continue
        boots[b] = roc_auc_score(y_b, s_b)
    good = boots[~np.isnan(boots)]
    return {
        "point": point,
        "ci95_low": float(np.percentile(good, 2.5)) if len(good) else None,
        "ci95_high": float(np.percentile(good, 97.5)) if len(good) else None,
        "n_boot": n_boot,
        "n_degenerate": int(n_degenerate),
    }


def _operating_point(y: np.ndarray, scores: np.ndarray, target_tpr: float) -> dict:
    """Pick threshold giving the smallest score s.t. TPR (rejection of OOS) ≥ target.

    Convention: score is monotone increasing in 'this is in-registry'. So we
    REJECT (classify as OOS) when score < threshold. TPR (true positive on OOS) =
    fraction of y==0 records with score < threshold. FRR (false reject of
    in-registry) = fraction of y==1 records with score < threshold.

    NOTE: scikit-learn's roc_curve treats the positive class as y==1, so the
    standard "tpr" is on positives (in_registry). We invert here: TPR on OOS =
    1 - FPR on positives at the same threshold... easier to compute directly.
    """
    oos_scores = scores[y == 0]
    in_scores = scores[y == 1]
    if len(oos_scores) == 0 or len(in_scores) == 0:
        return {"skipped": True, "reason": "empty class"}

    # Sweep thresholds = unique scores; pick smallest with TPR_oos ≥ target.
    # TPR_oos at threshold t = mean(oos_scores < t).
    cand = np.sort(np.unique(scores))
    chosen_t = None
    for t in cand:
        tpr_oos = float((oos_scores < t).mean())
        if tpr_oos >= target_tpr:
            chosen_t = float(t)
            break
    if chosen_t is None:
        # No threshold achieves target_tpr (very flat OOS distribution).
        chosen_t = float(cand[-1])

    tpr_oos = float((oos_scores < chosen_t).mean())
    frr = float((in_scores < chosen_t).mean())
    return {
        "target_tpr_oos": target_tpr,
        "threshold": chosen_t,
        "tpr_oos": tpr_oos,
        "frr_in_registry": frr,
        "n_oos": int(len(oos_scores)),
        "n_in_registry": int(len(in_scores)),
    }


def _verdict(clean_auroc: float, rotated_auroc: float | None) -> dict:
    """Pre-registered Pass/Weak-positive/Fail logic."""
    rot_ok = (rotated_auroc is None) or (rotated_auroc >= PASS_ROTATED_AUROC)
    if clean_auroc >= PASS_CLEAN_AUROC and rot_ok:
        zone = "pass"
    elif clean_auroc >= WEAK_POSITIVE_FLOOR:
        zone = "weak_positive"
    else:
        zone = "fail"
    return {
        "zone": zone,
        "clean_auroc": clean_auroc,
        "rotated_auroc": rotated_auroc,
        "passes_clean_bar": clean_auroc >= PASS_CLEAN_AUROC,
        "passes_rotated_bar": rot_ok,
        "thresholds": {
            "pass_clean_auroc": PASS_CLEAN_AUROC,
            "pass_rotated_auroc": PASS_ROTATED_AUROC,
            "weak_positive_floor": WEAK_POSITIVE_FLOOR,
        },
    }


def calibrate_on_val(
    val_records: list[dict],
    seed: int = 42,
) -> dict:
    """Fits z-score + LR weights + threshold on val. Applies the fallback rule."""
    X_val, y_val, pids_val = _extract_features(val_records)
    mu, sd = _zscore_fit(X_val[y_val == 1])  # in-registry stats
    Z_val = _zscore_apply(X_val, mu, sd)

    # AUROC of sim_top1 alone (z-scored sim is monotonic in raw sim → same AUROC).
    sim_col = FEATURE_NAMES.index("sim_top1")
    sim_only_auroc = _person_bootstrap_auroc(y_val, Z_val[:, sim_col], pids_val, seed=seed)

    # L2 logistic regression on all 5 z-scored features.
    # Use balanced class weight since val is imbalanced (~5x more in_registry than oos).
    lr = LogisticRegression(C=1.0, penalty="l2", class_weight="balanced", max_iter=1000)
    lr.fit(Z_val, y_val)
    lr_scores_val = lr.decision_function(Z_val)
    lr_auroc = _person_bootstrap_auroc(y_val, lr_scores_val, pids_val, seed=seed)

    lift = lr_auroc["point"] - sim_only_auroc["point"]
    if lift >= FALLBACK_LIFT_THRESHOLD:
        mode = "logistic_5feature"
        weights = lr.coef_[0].tolist()
        bias = float(lr.intercept_[0])
        chosen_val_scores = lr_scores_val
        chosen_val_auroc = lr_auroc
    else:
        mode = "sim_top1_only"
        # Score = z(sim_top1). Pre-registered fallback.
        weights = [0.0] * len(FEATURE_NAMES)
        weights[sim_col] = 1.0
        bias = 0.0
        chosen_val_scores = Z_val[:, sim_col]
        chosen_val_auroc = sim_only_auroc

    op = _operating_point(y_val, chosen_val_scores, TPR_OPERATING_POINT)

    return {
        "feature_names": FEATURE_NAMES,
        "zscore_mu": mu.tolist(),
        "zscore_sd": sd.tolist(),
        "mode": mode,
        "weights": weights,
        "bias": bias,
        "fallback_threshold_lift": FALLBACK_LIFT_THRESHOLD,
        "val_sim_only_auroc": sim_only_auroc,
        "val_lr_auroc": lr_auroc,
        "val_lift": lift,
        "val_chosen_auroc": chosen_val_auroc,
        "operating_point": op,
        "n_val_records": len(val_records),
        "n_val_persons": len(set(pids_val)),
    }


def apply_score(records: list[dict], calibration: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Apply locked z-score + weights to records. Returns (scores, y, pids)."""
    X, y, pids = _extract_features(records)
    mu = np.array(calibration["zscore_mu"], dtype=np.float64)
    sd = np.array(calibration["zscore_sd"], dtype=np.float64)
    Z = _zscore_apply(X, mu, sd)
    w = np.array(calibration["weights"], dtype=np.float64)
    b = float(calibration["bias"])
    scores = Z @ w + b
    return scores, y, pids


def evaluate_locked(
    records: list[dict],
    calibration: dict,
    label: str,
    seed: int = 42,
) -> dict:
    scores, y, pids = apply_score(records, calibration)
    auroc = _person_bootstrap_auroc(y, scores, pids, seed=seed)
    op = _operating_point(y, scores, TPR_OPERATING_POINT)
    # Apply the LOCKED threshold from calibration (not re-derived per-set).
    locked_t = float(calibration["operating_point"]["threshold"])
    oos_scores = scores[y == 0]
    in_scores = scores[y == 1]
    locked_tpr = float((oos_scores < locked_t).mean()) if len(oos_scores) else None
    locked_frr = float((in_scores < locked_t).mean()) if len(in_scores) else None
    return {
        "label": label,
        "auroc": auroc,
        "operating_point_locked": {
            "threshold": locked_t,
            "tpr_oos": locked_tpr,
            "frr_in_registry": locked_frr,
            "n_oos": int(len(oos_scores)),
            "n_in_registry": int(len(in_scores)),
        },
        "operating_point_rederived": op,
        "n_records": len(records),
        "n_persons": len(set(pids)),
    }


def plot_roc(records_by_label: dict[str, list[dict]], calibration: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    for label, records in records_by_label.items():
        scores, y, _ = apply_score(records, calibration)
        # roc_curve treats positive=in_registry (y==1). We want the OOS-rejection
        # ROC: TPR = OOS-correctly-classified-as-OOS = fraction of y==0 below
        # threshold. Easier: invert score sign.
        fpr, tpr, _ = roc_curve(1 - y, -scores)
        auroc = roc_auc_score(1 - y, -scores)
        ax.plot(fpr, tpr, label=f"{label} (AUROC={auroc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False positive rate (in_registry queries incorrectly rejected)")
    ax.set_ylabel("True positive rate (OOS queries correctly rejected)")
    ax.set_title("Open-set detection ROC")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", type=str, required=True,
                        help="Path to val heldout_enrol.json for fitting weights + threshold.")
    parser.add_argument("--evaluate", type=str, required=True,
                        help="Path to test heldout_enrol.json (clean, primary evaluation).")
    parser.add_argument("--evaluate-rotated", type=str, default=None,
                        help="Optional: path to test heldout_enrol_rotated.json for the "
                             "rotation-stress adversarial slice.")
    parser.add_argument("--out-dir", type=str, default="identification/runs/phase8_open_set")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cal_path = (PROJECT_ROOT / args.calibrate).resolve()
    test_path = (PROJECT_ROOT / args.evaluate).resolve()
    rot_path = (PROJECT_ROOT / args.evaluate_rotated).resolve() if args.evaluate_rotated else None

    print(f"[calibrate] reading {cal_path}")
    val_records = _load_records(cal_path)
    calibration = calibrate_on_val(val_records, seed=args.seed)
    calibration["source"] = {
        "calibrate_json": str(cal_path.relative_to(PROJECT_ROOT)),
        "evaluate_json": str(test_path.relative_to(PROJECT_ROOT)),
        "evaluate_rotated_json": (
            str(rot_path.relative_to(PROJECT_ROOT)) if rot_path else None
        ),
    }
    with open(out_dir / "phase8_open_set_calibration.json", "w") as f:
        json.dump(calibration, f, indent=2)
    print(f"  mode={calibration['mode']}, "
          f"val_chosen_auroc={calibration['val_chosen_auroc']['point']:.4f} "
          f"[{calibration['val_chosen_auroc']['ci95_low']:.3f}, "
          f"{calibration['val_chosen_auroc']['ci95_high']:.3f}], "
          f"lift={calibration['val_lift']:.4f}")
    print(f"  operating_point: thr={calibration['operating_point']['threshold']:.4f}, "
          f"tpr_oos={calibration['operating_point']['tpr_oos']:.4f}, "
          f"frr={calibration['operating_point']['frr_in_registry']:.4f}")
    print(f"[calibrate] saved → {out_dir/'phase8_open_set_calibration.json'}")

    print(f"\n[evaluate] reading {test_path}")
    test_records = _load_records(test_path)
    clean_result = evaluate_locked(test_records, calibration, "test_clean", seed=args.seed)
    print(f"  CLEAN AUROC: {clean_result['auroc']['point']:.4f} "
          f"[{clean_result['auroc']['ci95_low']:.3f}, "
          f"{clean_result['auroc']['ci95_high']:.3f}]")
    print(f"  At locked threshold: tpr_oos={clean_result['operating_point_locked']['tpr_oos']:.4f}, "
          f"frr={clean_result['operating_point_locked']['frr_in_registry']:.4f}")

    rotated_result = None
    rot_records = None
    if rot_path is not None:
        print(f"\n[evaluate-rotated] reading {rot_path}")
        rot_records = _load_records(rot_path)
        rotated_result = evaluate_locked(rot_records, calibration, "test_rotated", seed=args.seed)
        print(f"  ROTATED AUROC: {rotated_result['auroc']['point']:.4f} "
              f"[{rotated_result['auroc']['ci95_low']:.3f}, "
              f"{rotated_result['auroc']['ci95_high']:.3f}]")
        print(f"  At locked threshold: tpr_oos={rotated_result['operating_point_locked']['tpr_oos']:.4f}, "
              f"frr={rotated_result['operating_point_locked']['frr_in_registry']:.4f}")

    verdict = _verdict(
        clean_result["auroc"]["point"],
        rotated_result["auroc"]["point"] if rotated_result else None,
    )
    print(f"\n[VERDICT] zone={verdict['zone']}, "
          f"clean={verdict['clean_auroc']:.4f}, "
          f"rotated={verdict['rotated_auroc']}, "
          f"passes_clean={verdict['passes_clean_bar']}, "
          f"passes_rotated={verdict['passes_rotated_bar']}")

    # ROC plot
    records_by_label = {"clean": test_records}
    if rot_records is not None:
        records_by_label["rotated"] = rot_records
    try:
        plot_roc(records_by_label, calibration, out_dir / "roc.png")
        print(f"[plot] saved → {out_dir/'roc.png'}")
    except Exception as e:
        print(f"[plot] WARN: ROC plot failed: {e}")

    results = {
        "calibration_source": str(cal_path.relative_to(PROJECT_ROOT)),
        "evaluate_source": str(test_path.relative_to(PROJECT_ROOT)),
        "evaluate_rotated_source": (
            str(rot_path.relative_to(PROJECT_ROOT)) if rot_path else None
        ),
        "calibration_mode": calibration["mode"],
        "calibration_weights": calibration["weights"],
        "calibration_bias": calibration["bias"],
        "calibration_zscore_mu": calibration["zscore_mu"],
        "calibration_zscore_sd": calibration["zscore_sd"],
        "val_lift": calibration["val_lift"],
        "val_chosen_auroc": calibration["val_chosen_auroc"],
        "test_clean": clean_result,
        "test_rotated": rotated_result,
        "verdict": verdict,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[results] saved → {out_dir/'results.json'}")


if __name__ == "__main__":
    main()
