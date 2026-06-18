#!/usr/bin/env bash
# deploy.sh — one-shot re-deploy for the RTG dental-ID demo.
#
# Assumes:
#  - HF + Vercel accounts exist and CLI tools are installed.
#  - Model artefacts are already uploaded to the private HF Hub model repo
#    (one-time setup; see DEPLOY.md).
#  - Required env vars are exported (HF_USER, HF_TOKEN, HF_REVISION,
#    VERCEL_TOKEN), either from your shell or sourced from .env.deploy.
#
# What it does:
#  1. Push the backend/ subtree to the HF Space's `main` branch.
#  2. Wait for the Space to come back healthy (poll /api/health up to 15 min).
#  3. Smoke-test /api/health.
#  4. Deploy the frontend to Vercel prod with NEXT_PUBLIC_API_BASE set.
#  5. Smoke-test the frontend.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Optional .env.deploy for local convenience.
if [ -f .env.deploy ]; then
  set -a; source .env.deploy; set +a
fi

: "${HF_USER:?export HF_USER (huggingface username)}"
: "${HF_TOKEN:?export HF_TOKEN (huggingface write token)}"
: "${HF_REVISION:?export HF_REVISION (pinned model-repo commit sha)}"
: "${VERCEL_TOKEN:?export VERCEL_TOKEN}"

BACKEND_URL="https://${HF_USER}-rtg-backend.hf.space"
FRONTEND_PROJECT="${FRONTEND_PROJECT:-rtg-demo}"
HF_REPO_ID="${HF_REPO_ID:-${HF_USER}/rtg-tooth-id-weights}"

# ---------- STAGE 1: backend (Hugging Face Space) ----------
# Push the whole repo to the Space's main branch. The Space's Docker SDK
# expects Dockerfile + README.md at the repo root; gitignored directories
# (`dataset_raw/`, `identification/runs/`, generated registries) stay
# excluded. Runtime artefacts arrive separately from HF Hub via
# `_resolve_artefacts()` on container boot.
#
# Token-in-URL is avoided here — `git push` reads `HF_TOKEN` via the
# huggingface-cli credential helper instead, which keeps the token out of
# `.git/config` after the push.
echo "[deploy] pushing repo to HF Space..."
if ! git remote get-url hf-space >/dev/null 2>&1; then
  git remote add hf-space "https://huggingface.co/spaces/${HF_USER}/rtg-backend"
fi
# Inject the token via a one-off credential helper so it isn't persisted.
git -c "credential.helper=!f() { echo username=${HF_USER}; echo password=${HF_TOKEN}; }; f" \
    push hf-space "HEAD:refs/heads/main" --force

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
  echo "  https://huggingface.co/spaces/${HF_USER}/rtg-backend/logs"
  exit 1
fi

# ---------- STAGE 2: backend smoke ----------
HEALTH="$(curl -fsS "${BACKEND_URL}/api/health")"
echo "[deploy] /api/health -> ${HEALTH}"
echo "${HEALTH}" | grep -q '"status"' || { echo "[deploy] FAIL: backend not healthy"; exit 1; }

# ---------- STAGE 3: frontend (Vercel) ----------
echo "[deploy] deploying frontend to Vercel (prod)..."
cd frontend
VERCEL_OUT="$(vercel --prod --token "$VERCEL_TOKEN" --yes 2>&1 | tee /tmp/vercel.out)"
FRONT_URL="$(printf '%s' "${VERCEL_OUT}" | grep -oE 'https://[a-z0-9-]+\.vercel\.app' | tail -1)"
cd "$ROOT"

if [ -z "${FRONT_URL}" ]; then
  echo "[deploy] FAIL: could not parse Vercel URL from output"
  exit 1
fi

# ---------- STAGE 4: end-to-end smoke ----------
echo "[deploy] smoke-testing frontend at ${FRONT_URL}..."
curl -fsS --max-time 10 "${FRONT_URL}" >/dev/null || { echo "[deploy] FAIL: frontend not reachable"; exit 1; }

echo "[deploy] DONE"
echo "  frontend: ${FRONT_URL}"
echo "  backend:  ${BACKEND_URL}"
echo "  model rev: ${HF_REVISION}"
