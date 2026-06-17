#!/bin/bash
# Rotnorm orchestrator: rebuild manifest → train FDI cls → train embedder → build registry → evaluate.
# Run from repo root. Assumes identification/data/crops_gt_rotnorm/ already exists.

set -euo pipefail

PY=/opt/homebrew/Caskroom/miniforge/base/envs/rtg/bin/python

# Allow --skip-to-step to resume from a specific step (e.g. after a non-training failure).
SKIP_TO="${SKIP_TO:-}"
should_run() {
  if [ -z "$SKIP_TO" ]; then return 0; fi
  local current="$1"
  if [ "$current" = "$SKIP_TO" ]; then SKIP_TO=""; return 0; fi
  return 1
}

if should_run 8.1b; then
echo "=== 8.1b: build manifest ==="
$PY -m identification.scripts.build_manifest \
  --crops-dir identification/data/crops_gt_rotnorm \
  --output identification/data/manifest_rotnorm.csv

$PY -m identification.scripts.validate_crops \
  --manifest identification/data/manifest_rotnorm.csv \
  --output-clean identification/data/manifest_clean_rotnorm.csv \
  --output-rejected identification/data/rejection_log_rotnorm.csv \
  --output-grid identification/data/review_grid_rotnorm.png

# Filter to canonical 1,178 persons + 52 FDI classes
$PY -c "
import pandas as pd
new = pd.read_csv('identification/data/manifest_clean_rotnorm.csv')
orig = pd.read_csv('identification/data/manifest_clean.csv')
orig_pids = set(orig['person_id'])
orig_fdis = set(orig['tooth_fdi'].astype(str).unique())
new['tooth_fdi'] = new['tooth_fdi'].astype(str)
filt = new[new['person_id'].isin(orig_pids) & new['tooth_fdi'].isin(orig_fdis)].copy()
filt.to_csv('identification/data/manifest_clean_rotnorm.csv', index=False)
print(f'Filtered: {len(filt)} rows, {filt[\"person_id\"].nunique()} persons, {filt[\"tooth_fdi\"].nunique()} FDI classes')
"
fi

if should_run 8.1c; then
echo
echo "=== 8.1c: retrain FDI classifier on canonical crops ==="
rm -rf identification/runs/tooth_fdi_rotnorm/
$PY -u -m identification.training.train_classifier \
  --config identification/configs/tooth_classifier_rotnorm.yaml
fi

if should_run 8.1c-eval; then
echo
echo "=== 8.1c-eval: compare FDI val accuracy vs baseline ==="
mkdir -p identification/runs/phase8_rotnorm
$PY -m identification.evaluation.evaluate_fdi_val_acc \
  --checkpoint identification/runs/tooth_fdi_rotnorm/best.pt \
  --manifest identification/data/manifest_clean_rotnorm.csv \
  --baseline-checkpoint identification/runs/tooth_fdi_raw/best.pt \
  --baseline-manifest identification/data/manifest_clean.csv \
  --output identification/runs/phase8_rotnorm/fdi_val_acc.json || \
  echo "FDI val-acc check non-fatal — Stage C uses the deployed (non-rotnorm) FDI classifier at inference."
fi

if should_run 8.1d; then
echo
echo "=== 8.1d: retrain FDI-init embedder on canonical crops ==="
rm -rf identification/runs/embedding_fdi_init_rotnorm_v1/
$PY -u -m identification.training.train_embedding \
  --config identification/configs/embedding_fdi_init_rotnorm.yaml
fi

if should_run 8.1e; then
echo
echo "=== 8.1e: rebuild registry from canonical crops + new embedder ==="
rm -rf identification/registry_rotnorm/
$PY -m identification.scripts.build_registry \
  --checkpoint identification/runs/embedding_fdi_init_rotnorm_v1/best.pt \
  --manifest identification/data/manifest_clean_rotnorm.csv \
  --output-dir identification/registry_rotnorm
fi

if should_run 8.1f; then
echo
echo "=== 8.1f: re-run baseline evaluation against the rotnorm embedder ==="
$PY -u -m identification.evaluation.evaluate_pipeline_rotnorm \
  --embedder identification/runs/embedding_fdi_init_rotnorm_v1/best.pt \
  --registry-dir identification/registry_rotnorm \
  --output-dir identification/runs/phase8_rotnorm \
  --n-trials 5 \
  --rotation-deg 30 \
  --heldout-count 30 \
  --heldout-trials 5
fi

echo
echo "=== Rotnorm pipeline done. ==="
