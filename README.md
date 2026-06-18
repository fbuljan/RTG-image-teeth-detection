---
title: RTG Tooth ID Backend
emoji: 🦷
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# RTG Tooth ID — backend

FastAPI backend for a dental-X-ray person-identification demo (master-thesis
project). The frontend lives on Vercel; this Space hosts the inference
pipeline (YOLO segmentation + FDI classifier + tooth embedder + FAISS
retrieval against a 1,178-person registry).

## Surfaces

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/health` | Liveness check |
| `GET` | `/api/registry` | List all enrolled persons (fake names + tooth counts) |
| `GET` | `/api/registry/{person_id}/panoramic` | Download the panoramic |
| `GET` | `/api/intermediate/{query_id}/{file}` | Per-query overlay images |
| `POST` | `/api/identify` | Multipart upload, returns SSE stream |
| `POST` | `/api/identify-crops` | Pre-cropped tooth upload, SSE stream |
| `POST` | `/api/search-fragment` | Re-rank using a subset of a previous query's teeth |
| `POST` | `/api/enrol` | Session-scoped enrolment of a new panoramic |
| `GET` | `/api/model-card` | Static evaluation numbers for the UI |

## Runtime configuration

| Env var | Purpose |
| --- | --- |
| `HF_REPO_ID` | Private HF model repo with weights + registry + JSON metrics |
| `HF_REVISION` | Pinned commit SHA on the model repo (REQUIRED if `HF_REPO_ID` set) |
| `HF_TOKEN` | Read token for the private model repo (injected as Space secret) |
| `ALLOWED_ORIGINS` | Comma-separated list of frontend origins for CORS |
| `ALLOWED_ORIGIN_REGEX` | Optional regex for Vercel preview URLs |
| `MAX_UPLOAD_BYTES` | Per-request upload cap, default 15 MB |
| `DEMO_USE_YOLO_REGISTRY` | `0` to roll back to the legacy GT-built registry |

When `HF_REPO_ID` is set, model weights and registry artefacts are pulled
from the pinned HF Hub model repo at boot — see `_resolve_artefacts()` in
`backend/pipeline.py`. Without it (local dev) the backend reads everything
from in-repo paths.

## Local development

```bash
# Build the registry first (one-time)
python -m identification.scripts.build_registry

# Start the API
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

See `backend/README.md` for more on the runtime model layout, and `DEPLOY.md`
(gitignored, local copy in repo root) for the full deploy plan.
