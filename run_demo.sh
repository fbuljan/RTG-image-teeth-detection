#!/usr/bin/env bash
# Boot the dental identification demo.
#
# Starts the FastAPI backend (port 8000) and the Next.js frontend (port 3000)
# in two background processes, prints their PIDs, and waits for Ctrl+C.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${RTG_PYTHON:-/opt/homebrew/Caskroom/miniforge/base/envs/rtg/bin/python}"
LOG_DIR="${ROOT}/.demo_logs"
mkdir -p "${LOG_DIR}"

# 1. Ensure registry exists ---------------------------------------------------
if [[ ! -f "${ROOT}/identification/registry/index.faiss" ]]; then
  echo "[demo] Registry not found, building it (one-time, ~3 minutes)..."
  cd "${ROOT}"
  "${PY}" -m identification.scripts.build_registry
fi

# 2. Frontend deps ------------------------------------------------------------
if [[ ! -d "${ROOT}/frontend/node_modules" ]]; then
  echo "[demo] Installing frontend dependencies..."
  (cd "${ROOT}/frontend" && npm install --silent)
fi

# 3. Start backend ------------------------------------------------------------
echo "[demo] Starting FastAPI on http://127.0.0.1:8000 ..."
(cd "${ROOT}" && "${PY}" -m uvicorn backend.app:app --host 127.0.0.1 --port 8000) \
  > "${LOG_DIR}/backend.log" 2>&1 &
BACKEND_PID=$!
echo "[demo] backend pid=${BACKEND_PID} (logs: ${LOG_DIR}/backend.log)"

# Give the backend a moment to load all models before launching the UI
for i in {1..30}; do
  if curl -sf http://127.0.0.1:8000/api/health >/dev/null 2>&1; then break; fi
  sleep 1
done

# 4. Start frontend -----------------------------------------------------------
echo "[demo] Starting Next.js on http://localhost:3000 ..."
(cd "${ROOT}/frontend" && npm run dev --silent) \
  > "${LOG_DIR}/frontend.log" 2>&1 &
FRONTEND_PID=$!
echo "[demo] frontend pid=${FRONTEND_PID} (logs: ${LOG_DIR}/frontend.log)"

cleanup() {
  echo
  echo "[demo] Stopping..."
  kill "${BACKEND_PID}" "${FRONTEND_PID}" 2>/dev/null || true
  wait "${BACKEND_PID}" "${FRONTEND_PID}" 2>/dev/null || true
  echo "[demo] Stopped."
}
trap cleanup EXIT INT TERM

echo
echo "============================================================"
echo "  Open http://localhost:3000 in your browser."
echo "  Press Ctrl+C to stop both servers."
echo "============================================================"
wait
