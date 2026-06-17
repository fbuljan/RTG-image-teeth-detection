"""
Demographic prediction (sex + age) on the frozen deployed embedder.

Trains two heads on top of the deployed FDI-init embedder's GT-built registry
(per-person mean-pooled 128-d embeddings, L2-normalised):

  1. Sex classifier: 2-layer MLP (binary cross-entropy).
  2. Age regressor: 2-layer MLP (Huber loss, MAE evaluation).

Person-disjoint train/val/test split is inherited from the manifest (the same
split used for all retrieval evaluation).

Pre-registered Pass criterion:

  PASS (wire into demo) — both must hold on test:
    - Sex accuracy ≥ 80% on held-out persons
    - Age MAE ≤ 2.5 years within the 6-13y dense bucket

  MARGINAL (analysis only) — either holds:
    - Sex accuracy 65-80%, OR
    - Age MAE 2.5-4y on 6-13y bucket

  FAIL — one-paragraph negative result:
    - Sex < 65% OR Age MAE > 4y on 6-13y bucket

Adversarial slice (also pre-registered):
  If sex accuracy varies by >15pp across age buckets, the global number becomes
  a one-line caveat in the demo banner, not a confidence statement.

Usage:
  PYTHONPATH=. python identification/training/train_demographic_classifier.py \\
    --registry identification/registry \\
    --manifest identification/data/manifest.csv \\
    --out-dir identification/runs/demographic_v2 \\
    --seed 0
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from identification.models.retrieval_index import RetrievalIndex  # noqa: E402

EMBED_DIM = 128
N_BOOTSTRAP = 1000

# Pre-registered thresholds
PASS_SEX = 0.80
MARG_SEX = 0.65
PASS_AGE_MAE = 2.5  # years, on 6-13y dense bucket
MARG_AGE_MAE = 4.0
ADVERSARIAL_SEX_SPREAD = 0.15  # >15pp variance across age buckets


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------

class MLPHead(nn.Module):
    """Two-layer MLP with dropout. Used for both sex (output_dim=1) and age (output_dim=1)."""

    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.3, out_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_per_person_data(
    registry_dir: Path,
    manifest_csv: Path,
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """Returns (pid → 128-d embedding, pid → {'age', 'sex', 'split'})."""
    reg = RetrievalIndex.load(str(registry_dir / "index"), dim=EMBED_DIM)
    embeddings: dict[str, np.ndarray] = {}
    for i, pid in enumerate(reg.person_ids):
        embeddings[pid] = reg.index.reconstruct(i).astype(np.float32)

    meta: dict[str, dict] = {}
    with open(manifest_csv) as f:
        for row in csv.DictReader(f):
            pid = row["person_id"]
            if pid not in meta:
                meta[pid] = {
                    "age": float(row["age"]),
                    "sex": row["sex"],
                    "split": row["split"],
                }
    return embeddings, meta


def assemble_split(
    embeddings: dict[str, np.ndarray],
    meta: dict[str, dict],
    split: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Returns X (n, 128), y_sex (n,), y_age (n,), pids (n,) for the given split."""
    rows = []
    for pid, e in embeddings.items():
        m = meta.get(pid)
        if m is None or m["split"] != split:
            continue
        rows.append((pid, e, m["sex"], m["age"]))
    rows.sort(key=lambda r: r[0])  # deterministic order
    pids = [r[0] for r in rows]
    X = np.stack([r[1] for r in rows]).astype(np.float32)
    y_sex = np.array([1.0 if r[2] == "female" else 0.0 for r in rows], dtype=np.float32)
    y_age = np.array([r[3] for r in rows], dtype=np.float32)
    return X, y_sex, y_age, pids


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def train_sex_head(
    X_train: torch.Tensor, y_train: torch.Tensor,
    X_val: torch.Tensor, y_val: torch.Tensor,
    seed: int = 0, n_epochs: int = 200, lr: float = 1e-3, weight_decay: float = 1e-3,
    patience: int = 25,
) -> tuple[MLPHead, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLPHead(EMBED_DIM, hidden=64, dropout=0.3, out_dim=1)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

    best_val_acc, best_state, best_epoch, stale = 0.0, None, 0, 0
    log = []
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            logits = model(xb).squeeze(-1)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.detach()) * xb.shape[0]
        train_loss = total_loss / len(X_train)
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val).squeeze(-1)
            val_acc = ((val_logits > 0).float() == y_val).float().mean().item()
        log.append({"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc})
        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, {"best_val_acc": best_val_acc, "best_epoch": best_epoch,
                   "n_epochs_run": len(log), "log": log}


def train_age_head(
    X_train: torch.Tensor, y_train: torch.Tensor,
    X_val: torch.Tensor, y_val: torch.Tensor,
    seed: int = 0, n_epochs: int = 200, lr: float = 1e-3, weight_decay: float = 1e-3,
    patience: int = 25, huber_delta: float = 1.0,
) -> tuple[MLPHead, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLPHead(EMBED_DIM, hidden=64, dropout=0.3, out_dim=1)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.HuberLoss(delta=huber_delta)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

    best_val_mae, best_state, best_epoch, stale = float("inf"), None, 0, 0
    log = []
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.detach()) * xb.shape[0]
        train_loss = total_loss / len(X_train)
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val).squeeze(-1)
            val_mae = (val_pred - y_val).abs().mean().item()
        log.append({"epoch": epoch, "train_loss": train_loss, "val_mae": val_mae})
        if val_mae < best_val_mae - 1e-4:
            best_val_mae = val_mae
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, {"best_val_mae": best_val_mae, "best_epoch": best_epoch,
                   "n_epochs_run": len(log), "log": log}


# ---------------------------------------------------------------------------
# Bootstrap evaluation
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: np.ndarray, fn, rng: np.random.Generator, n_iter: int = N_BOOTSTRAP,
) -> tuple[float, float]:
    boots = []
    for _ in range(n_iter):
        idx = rng.integers(0, len(values), size=len(values))
        boots.append(fn(values[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_two_arrays(
    a: np.ndarray, b: np.ndarray, fn, rng: np.random.Generator, n_iter: int = N_BOOTSTRAP,
) -> tuple[float, float]:
    """Paired bootstrap for a metric that depends on both arrays (e.g. accuracy)."""
    boots = []
    n = len(a)
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        boots.append(fn(a[idx], b[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def eval_sex(
    model: MLPHead, X: torch.Tensor, y: torch.Tensor, ages: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    model.eval()
    with torch.no_grad():
        logits = model(X).squeeze(-1).numpy()
    preds = (logits > 0).astype(np.float32)
    y_np = y.numpy()
    acc = float((preds == y_np).mean())
    lo, hi = bootstrap_two_arrays(preds, y_np, lambda a, b: float((a == b).mean()), rng)
    # Per-age-bucket
    buckets = [("6-9", 6, 10), ("10-12", 10, 13), ("13-15", 13, 16), ("16-18", 16, 19)]
    per_bucket = []
    for name, lo_a, hi_a in buckets:
        mask = (ages >= lo_a) & (ages < hi_a)
        sub_p, sub_y = preds[mask], y_np[mask]
        if len(sub_p) == 0:
            per_bucket.append({"label": name, "n": 0, "acc": float("nan"),
                               "acc_ci_low": float("nan"), "acc_ci_high": float("nan")})
            continue
        a = float((sub_p == sub_y).mean())
        ci_lo, ci_hi = bootstrap_two_arrays(sub_p, sub_y, lambda x, y: float((x == y).mean()), rng)
        per_bucket.append({"label": name, "n": int(mask.sum()), "acc": a,
                           "acc_ci_low": ci_lo, "acc_ci_high": ci_hi})
    spread = max(b["acc"] for b in per_bucket if b["n"] > 0) - \
             min(b["acc"] for b in per_bucket if b["n"] > 0)
    return {
        "test_acc": acc,
        "test_acc_ci_low": lo,
        "test_acc_ci_high": hi,
        "n_test": int(len(y_np)),
        "logits": logits.tolist(),
        "preds": preds.tolist(),
        "targets": y_np.tolist(),
        "per_age_bucket": per_bucket,
        "max_min_bucket_spread": float(spread),
        "adversarial_threshold": ADVERSARIAL_SEX_SPREAD,
        "adversarial_trigger": spread > ADVERSARIAL_SEX_SPREAD,
    }


def eval_age(
    model: MLPHead, X: torch.Tensor, y: torch.Tensor, sexes: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    model.eval()
    with torch.no_grad():
        preds = model(X).squeeze(-1).numpy()
    y_np = y.numpy()
    full_mae = float(np.abs(preds - y_np).mean())
    full_lo, full_hi = bootstrap_two_arrays(preds, y_np, lambda p, t: float(np.abs(p - t).mean()), rng)
    # Dense 6-13y bucket
    dense_mask = (y_np >= 6) & (y_np < 13)
    dense_mae = float(np.abs(preds[dense_mask] - y_np[dense_mask]).mean())
    dense_lo, dense_hi = bootstrap_two_arrays(preds[dense_mask], y_np[dense_mask],
                                              lambda p, t: float(np.abs(p - t).mean()), rng)
    # Per-bucket
    buckets = [("6-9", 6, 10), ("10-12", 10, 13), ("13-15", 13, 16), ("16-18", 16, 19)]
    per_bucket = []
    for name, lo_a, hi_a in buckets:
        mask = (y_np >= lo_a) & (y_np < hi_a)
        if not mask.any():
            per_bucket.append({"label": name, "n": 0, "mae": float("nan"),
                               "mae_ci_low": float("nan"), "mae_ci_high": float("nan")})
            continue
        mae = float(np.abs(preds[mask] - y_np[mask]).mean())
        ci_lo, ci_hi = bootstrap_two_arrays(preds[mask], y_np[mask],
                                            lambda p, t: float(np.abs(p - t).mean()), rng)
        per_bucket.append({"label": name, "n": int(mask.sum()), "mae": mae,
                           "mae_ci_low": ci_lo, "mae_ci_high": ci_hi})
    # Per-sex (adversarial robustness)
    per_sex = []
    for sex_label, sex_val in [("male", 0.0), ("female", 1.0)]:
        m = sexes == sex_val
        if not m.any():
            continue
        mae = float(np.abs(preds[m] - y_np[m]).mean())
        ci_lo, ci_hi = bootstrap_two_arrays(preds[m], y_np[m],
                                            lambda p, t: float(np.abs(p - t).mean()), rng)
        per_sex.append({"label": sex_label, "n": int(m.sum()), "mae": mae,
                        "mae_ci_low": ci_lo, "mae_ci_high": ci_hi})
    return {
        "test_mae_full": full_mae,
        "test_mae_full_ci_low": full_lo,
        "test_mae_full_ci_high": full_hi,
        "test_mae_dense_6_13": dense_mae,
        "test_mae_dense_6_13_ci_low": dense_lo,
        "test_mae_dense_6_13_ci_high": dense_hi,
        "n_test": int(len(y_np)),
        "n_dense": int(dense_mask.sum()),
        "preds": preds.tolist(),
        "targets": y_np.tolist(),
        "per_age_bucket": per_bucket,
        "per_sex": per_sex,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_age_scatter(targets: np.ndarray, preds: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(targets, preds, s=15, alpha=0.6, color="steelblue")
    lo, hi = min(targets.min(), preds.min()) - 0.5, max(targets.max(), preds.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="y = x (perfect)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("True age (years)")
    ax.set_ylabel("Predicted age (years)")
    ax.set_title("Age prediction on test set\n"
                 f"MAE = {float(np.abs(preds - targets).mean()):.2f}y (full range)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------

def verdict(sex_acc: float, age_mae_dense: float, sex_spread: float) -> tuple[str, str]:
    if sex_acc >= PASS_SEX and age_mae_dense <= PASS_AGE_MAE:
        v = "PASS"
        text = (f"sex acc = {sex_acc:.3f} ≥ {PASS_SEX} AND age MAE (6-13y) = "
                f"{age_mae_dense:.2f}y ≤ {PASS_AGE_MAE}y. Both pre-registered Pass "
                "gates met. Wire into demo.")
    elif (MARG_SEX <= sex_acc < PASS_SEX) or (PASS_AGE_MAE < age_mae_dense <= MARG_AGE_MAE):
        v = "MARGINAL"
        text = (f"sex acc = {sex_acc:.3f}, age MAE (6-13y) = {age_mae_dense:.2f}y. "
                "At least one metric is in the marginal band; analysis only, no demo wiring.")
    else:
        v = "FAIL"
        text = (f"sex acc = {sex_acc:.3f} < {MARG_SEX} OR age MAE (6-13y) = "
                f"{age_mae_dense:.2f}y > {MARG_AGE_MAE}y. One-paragraph negative result.")
    if sex_spread > ADVERSARIAL_SEX_SPREAD:
        text += (f"  ALSO: sex accuracy varies by {sex_spread*100:.1f}pp across age "
                 f"buckets (>15pp threshold) → global sex number is a caveat, not a "
                 "confidence statement.")
    return v, text


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--registry", type=Path, required=True,
                   help="Registry dir containing index.faiss + index.ids.json")
    p.add_argument("--manifest", type=Path, required=True,
                   help="manifest.csv (provides age, sex, split per PID)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=25)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"[demographic] registry: {args.registry}")
    print(f"[demographic] manifest: {args.manifest}")
    print(f"[demographic] seed:     {args.seed}")
    print()

    embeddings, meta = load_per_person_data(args.registry, args.manifest)
    print(f"[demographic] loaded {len(embeddings)} person-mean embeddings from registry")

    X_tr, ys_tr, ya_tr, pids_tr = assemble_split(embeddings, meta, "train")
    X_va, ys_va, ya_va, pids_va = assemble_split(embeddings, meta, "val")
    X_te, ys_te, ya_te, pids_te = assemble_split(embeddings, meta, "test")
    print(f"[demographic] splits: train={len(pids_tr)}, val={len(pids_va)}, test={len(pids_te)}")
    print(f"[demographic] train sex chance baseline (female share): {ys_tr.mean():.3f}")
    print(f"[demographic] test  sex chance baseline (female share): {ys_te.mean():.3f}")
    print(f"[demographic] test age dense bucket (6-13y) n: {int(((ya_te >= 6) & (ya_te < 13)).sum())}")
    print()

    X_tr_t = torch.from_numpy(X_tr)
    X_va_t = torch.from_numpy(X_va)
    X_te_t = torch.from_numpy(X_te)

    # --- Sex ---
    print("[demographic] training sex head...")
    ys_tr_t = torch.from_numpy(ys_tr)
    ys_va_t = torch.from_numpy(ys_va)
    ys_te_t = torch.from_numpy(ys_te)
    sex_model, sex_train_log = train_sex_head(X_tr_t, ys_tr_t, X_va_t, ys_va_t, seed=args.seed,
                                              n_epochs=args.n_epochs, patience=args.patience)
    sex_result = eval_sex(sex_model, X_te_t, ys_te_t, ya_te, rng)
    sex_result["best_val_acc"] = sex_train_log["best_val_acc"]
    sex_result["best_epoch"] = sex_train_log["best_epoch"]
    sex_result["n_epochs_run"] = sex_train_log["n_epochs_run"]
    print(f"  best_val_acc={sex_train_log['best_val_acc']:.4f} at epoch {sex_train_log['best_epoch']} "
          f"(ran {sex_train_log['n_epochs_run']} epochs)")
    print(f"  test_acc={sex_result['test_acc']:.4f} "
          f"[{sex_result['test_acc_ci_low']:.3f}, {sex_result['test_acc_ci_high']:.3f}]")
    print(f"  per-age bucket spread = {sex_result['max_min_bucket_spread']*100:.1f}pp "
          f"(threshold {ADVERSARIAL_SEX_SPREAD*100:.0f}pp)")
    print()

    # --- Age ---
    print("[demographic] training age head...")
    ya_tr_t = torch.from_numpy(ya_tr)
    ya_va_t = torch.from_numpy(ya_va)
    ya_te_t = torch.from_numpy(ya_te)
    age_model, age_train_log = train_age_head(X_tr_t, ya_tr_t, X_va_t, ya_va_t, seed=args.seed,
                                              n_epochs=args.n_epochs, patience=args.patience)
    age_result = eval_age(age_model, X_te_t, ya_te_t, ys_te, rng)
    age_result["best_val_mae"] = age_train_log["best_val_mae"]
    age_result["best_epoch"] = age_train_log["best_epoch"]
    age_result["n_epochs_run"] = age_train_log["n_epochs_run"]
    print(f"  best_val_mae={age_train_log['best_val_mae']:.4f}y at epoch {age_train_log['best_epoch']} "
          f"(ran {age_train_log['n_epochs_run']} epochs)")
    print(f"  test_mae_full = {age_result['test_mae_full']:.3f}y "
          f"[{age_result['test_mae_full_ci_low']:.3f}, {age_result['test_mae_full_ci_high']:.3f}]")
    print(f"  test_mae_dense_6_13 = {age_result['test_mae_dense_6_13']:.3f}y "
          f"[{age_result['test_mae_dense_6_13_ci_low']:.3f}, {age_result['test_mae_dense_6_13_ci_high']:.3f}]")
    print()

    # --- Verdict ---
    v_label, v_text = verdict(sex_result["test_acc"], age_result["test_mae_dense_6_13"],
                              sex_result["max_min_bucket_spread"])
    print(f"[demographic] === VERDICT: {v_label} ===")
    print(f"  {v_text}")
    print()
    print("  Per-age bucket (sex acc):")
    for b in sex_result["per_age_bucket"]:
        print(f"    {b['label']:>5}: n={b['n']:>3}  acc={b['acc']:.3f} "
              f"[{b['acc_ci_low']:.3f}, {b['acc_ci_high']:.3f}]")
    print("  Per-age bucket (age MAE):")
    for b in age_result["per_age_bucket"]:
        print(f"    {b['label']:>5}: n={b['n']:>3}  MAE={b['mae']:.3f}y "
              f"[{b['mae_ci_low']:.3f}, {b['mae_ci_high']:.3f}]")
    print("  Per-sex (age MAE):")
    for s in age_result["per_sex"]:
        print(f"    {s['label']:>6}: n={s['n']:>3}  MAE={s['mae']:.3f}y "
              f"[{s['mae_ci_low']:.3f}, {s['mae_ci_high']:.3f}]")

    # Persist
    out = {
        "args": {
            "registry": str(args.registry),
            "manifest": str(args.manifest),
            "seed": args.seed,
            "n_epochs": args.n_epochs,
            "patience": args.patience,
            "n_bootstrap": N_BOOTSTRAP,
            "pass_sex": PASS_SEX,
            "marg_sex": MARG_SEX,
            "pass_age_mae_dense": PASS_AGE_MAE,
            "marg_age_mae_dense": MARG_AGE_MAE,
            "adversarial_sex_spread": ADVERSARIAL_SEX_SPREAD,
        },
        "splits": {
            "train_n": len(pids_tr), "val_n": len(pids_va), "test_n": len(pids_te),
            "train_sex_female_share": float(ys_tr.mean()),
            "test_sex_female_share": float(ys_te.mean()),
            "test_age_dense_6_13_n": int(((ya_te >= 6) & (ya_te < 13)).sum()),
        },
        "sex": sex_result,
        "age": age_result,
        "verdict": v_label,
        "verdict_text": v_text,
    }
    out_json = args.out_dir / "results.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[demographic] wrote {out_json}")

    # Save model weights
    torch.save(sex_model.state_dict(), args.out_dir / "sex_head.pt")
    torch.save(age_model.state_dict(), args.out_dir / "age_head.pt")
    print(f"[demographic] wrote sex_head.pt + age_head.pt")

    # Age scatter plot
    plot_age_scatter(np.array(age_result["targets"]), np.array(age_result["preds"]),
                     args.out_dir / "age_scatter.png")
    print(f"[demographic] wrote age_scatter.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
