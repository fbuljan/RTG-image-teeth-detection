#!/usr/bin/env bash
# deploy.sh — re-deploy the dental-ID demo backend to its HF Space.
#
# The frontend deploys separately via Vercel's GitHub integration —
# every push to `main` on the GitHub remote triggers a Vercel prod
# deploy; every push to any other branch gets a preview URL. So this
# script only touches the backend; for a full deploy you `./deploy.sh`
# AND `git push origin main` (in either order).
#
# Required env vars (export or put in .env.deploy):
#   HF_USER       Hugging Face username
#   HF_TOKEN      write-scoped token
#   HF_REVISION   pinned commit SHA on the model-weights repo
#   HF_REPO_ID    defaults to "${HF_USER}/rtg-tooth-id-weights"
#
# Setup steps that happened once (NOT done here, see DEPLOY.md):
#   - Created HF account + write token.
#   - Uploaded weights/registry/JSON metrics to a private HF Hub model repo.
#   - Created the Space at huggingface.co/spaces/<user>/rtg-backend.
#   - Set HF_REPO_ID, HF_REVISION, HF_TOKEN as Space secrets.
#   - Imported the GitHub repo into Vercel; set NEXT_PUBLIC_API_BASE.

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

# ---------- STAGE 1: backend (Hugging Face Space) ----------
# Push the whole repo to the Space's `main` branch. The Space's Docker SDK
# expects Dockerfile + README.md at the repo root. Gitignored directories
# (`dataset_raw/`, `identification/runs/`, generated registries) stay out;
# runtime artefacts arrive separately via `_resolve_artefacts()` on boot.
#
# Token-in-URL is avoided here — a one-off credential helper injects
# HF_TOKEN via stdin so it isn't persisted in .git/config.
echo "[deploy] pushing repo to HF Space (${HF_USER}/rtg-backend)..."
if ! git remote get-url hf-space >/dev/null 2>&1; then
  git remote add hf-space "https://huggingface.co/spaces/${HF_USER}/rtg-backend"
fi
git -c "credential.helper=!f() { echo username=${HF_USER}; echo password=${HF_TOKEN}; }; f" \
    push hf-space "HEAD:refs/heads/main" --force

# ---------- STAGE 2: wait for Space to come back healthy ----------
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

# ---------- STAGE 3: smoke ----------
HEALTH="$(curl -fsS "${BACKEND_URL}/api/health")"
echo "[deploy] /api/health -> ${HEALTH}"
echo "${HEALTH}" | grep -q '"status"' || { echo "[deploy] FAIL: backend not healthy"; exit 1; }

echo "[deploy] DONE"
echo "  backend:    ${BACKEND_URL}"
echo "  model rev:  ${HF_REVISION}"
echo "  frontend:   deployed separately via Vercel GitHub integration."
echo "              push to main on the GitHub remote and Vercel auto-deploys."
