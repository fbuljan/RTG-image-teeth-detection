# Dockerfile for the Hugging Face Space backend.
#
# Local development does NOT use this — `uvicorn backend.app:app --reload`
# runs against your conda env directly. This image only runs inside the
# Space at https://huggingface.co/spaces/<user>/rtg-backend.
#
# Build context: the repo root. The Space hosts the whole repo (gitignored
# directories like `dataset_raw/`, `identification/runs/`, and registries
# are not pushed). We copy `backend/` + `identification/` source code into
# the image; runtime artefacts (model weights, FAISS registry, JSON metrics)
# are downloaded from a pinned HF Hub model repo by `_resolve_artefacts()`
# on startup — see `backend/pipeline.py` and `DEPLOY.md`.

FROM python:3.11-slim

# System libs:
#   libgl1, libglib2.0-0 — OpenCV runtime (cv2 imports libGL even with the
#                          headless wheel for some helpers; cheap insurance).
#   git                  — huggingface_hub uses it for some download paths.
#   ca-certificates      — HTTPS to HF Hub.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces run as a non-root `user` (uid 1000) and expect the app
# at /home/user/app. Mirror that locally so local docker-run paths match.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/user/.cache/huggingface

WORKDIR /home/user/app

# Install Python deps first so the slow torch wheel layer is cached across
# code-only redeploys.
COPY --chown=user:user backend/requirements.txt /home/user/app/backend/requirements.txt
RUN pip install --no-cache-dir --user -r backend/requirements.txt

# Copy the application code. We need backend/ + the identification/ subtree
# the backend imports (data, models, evaluation/evaluate_embedding,
# training/train_demographic_classifier).
COPY --chown=user:user backend /home/user/app/backend
COPY --chown=user:user identification /home/user/app/identification

# HF Spaces expose container port 7860 by default (the SPACE-level proxy
# rewrites that to the public URL).
ENV PORT=7860
EXPOSE 7860

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
