#!/usr/bin/env bash
# deploy.sh — re-deploy the dental-ID demo backend to its HF Space.
#
# Strategy: build a slim staging tree containing ONLY what the Docker image
# needs at runtime, init a fresh throwaway git repo there, and force-push
# that to the Space. This sidesteps the multi-GB main repo history (LFS
# blobs, training-manifest CSVs, dataset_raw, etc.) — Space push goes from
# ~2.5 GB to ~5 MB.
#
# Frontend deploys separately via Vercel's GitHub integration — every push
# to `main` on the GitHub remote auto-deploys. This script touches the
# backend only.
#
# Required env vars (export or put in .env.deploy):
#   HF_USER       Hugging Face username
#   HF_TOKEN      write-scoped token
#   HF_REVISION   pinned commit SHA on the model-weights repo
#   HF_REPO_ID    defaults to "${HF_USER}/rtg-tooth-id-weights"

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [ -f .env.deploy ]; then
  set -a; source .env.deploy; set +a
fi

: "${HF_USER:?export HF_USER (huggingface username)}"
: "${HF_TOKEN:?export HF_TOKEN (huggingface write token)}"
: "${HF_REVISION:?export HF_REVISION (pinned model-repo commit sha)}"

BACKEND_URL="https://${HF_USER}-rtg-backend.hf.space"
HF_REPO_ID="${HF_REPO_ID:-${HF_USER}/rtg-tooth-id-weights}"
SPACE_REPO="https://huggingface.co/spaces/${HF_USER}/rtg-backend"

# ---------- STAGE 1: build slim tree ----------
STAGE_DIR="$(mktemp -d -t rtg-space-stage.XXXXXX)"
trap 'rm -rf "$STAGE_DIR"' EXIT

echo "[deploy] staging slim tree at $STAGE_DIR..."

# Repo-root files the Space needs.
cp Dockerfile README.md "$STAGE_DIR/"

# backend/ — copy source minus pycache and runtime-only state dirs that
# accumulated during local dev (sessions/, temp/, artefact cache). The
# Space recreates these at runtime as user uploads come in.
rsync -a \
  --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='.artefact_cache' \
  --exclude='sessions/' \
  --exclude='temp/' \
  backend/ "$STAGE_DIR/backend/"

# identification/ — copy ONLY Python source. The package is 8.2 GB on disk
# (training crops, runs, registry caches, CSVs) but the backend only imports
# a handful of .py modules. Allowlist beats blocklist here.
mkdir -p "$STAGE_DIR/identification"
cp identification/__init__.py "$STAGE_DIR/identification/"
for sub in configs data evaluation models training utils; do
  mkdir -p "$STAGE_DIR/identification/$sub"
  # find: only .py and .yaml under each subdir (config files use yaml).
  find "identification/$sub" \
    \( -name '*.py' -o -name '*.yaml' \) \
    -not -path '*/__pycache__/*' \
    -print0 \
  | rsync -a --files-from=- --from0 ./ "$STAGE_DIR/"
done

STAGE_SIZE=$(du -sh "$STAGE_DIR" | awk '{print $1}')
STAGE_FILES=$(find "$STAGE_DIR" -type f | wc -l | tr -d ' ')
echo "[deploy] slim tree: $STAGE_FILES files, $STAGE_SIZE"

# ---------- STAGE 2: init throwaway git repo + push ----------
echo "[deploy] initialising throwaway git repo + pushing to HF Space..."
cd "$STAGE_DIR"
git init -q -b main
git add .
GIT_AUTHOR_NAME=deploy GIT_AUTHOR_EMAIL=deploy@local \
GIT_COMMITTER_NAME=deploy GIT_COMMITTER_EMAIL=deploy@local \
  git commit -q -m "deploy: slim Space tree (rev ${HF_REVISION:0:7})"

git remote add hf-space "$SPACE_REPO"
git -c "credential.helper=!f() { echo username=${HF_USER}; echo password=${HF_TOKEN}; }; f" \
    push hf-space "HEAD:refs/heads/main" --force

cd "$ROOT"

# ---------- STAGE 3: wait for Space to come back healthy ----------
echo "[deploy] waiting for HF Space build (poll /api/health up to 15 min)..."
HEALTHY=0
for i in $(seq 1 90); do
  if curl -fsS --max-time 5 "${BACKEND_URL}/api/health" >/dev/null 2>&1; then
    echo "[deploy] backend up after $((i * 10))s"
    HEALTHY=1
    break
  fi
  sleep 10
done
if [ "${HEALTHY}" -ne 1 ]; then
  echo "[deploy] FAIL: backend did not come up within 15 min"
  echo "[deploy] Check the Space build log:"
  echo "  ${SPACE_REPO}/logs"
  exit 1
fi

# ---------- STAGE 4: smoke ----------
HEALTH="$(curl -fsS "${BACKEND_URL}/api/health")"
echo "[deploy] /api/health -> ${HEALTH}"
echo "${HEALTH}" | grep -q '"status"' || { echo "[deploy] FAIL: backend not healthy"; exit 1; }

echo "[deploy] DONE"
echo "  backend:    ${BACKEND_URL}"
echo "  model rev:  ${HF_REVISION}"
echo "  frontend:   deployed separately via Vercel GitHub integration."
