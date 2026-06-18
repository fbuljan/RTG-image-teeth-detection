# Backend — FastAPI demo server

(The HF Space frontmatter lives in the repo-root `README.md` — the Space
treats that as its landing card, not this file.)

End-to-end dental-X-ray person-identification pipeline with SSE streaming.
Loads YOLO + FDI classifier + tooth embedder + FAISS registry once on
startup and serves these endpoints:

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/health` | Liveness check (returns device + registry size) |
| `GET` | `/api/registry` | List all enrolled persons (fake names + tooth counts) |
| `GET` | `/api/registry/{person_id}/panoramic` | Download the panoramic as `xray.png` |
| `GET` | `/api/intermediate/{query_id}/{file}` | Per-query overlay images |
| `POST` | `/api/identify` | Multipart upload, returns SSE stream |
| `POST` | `/api/identify-crops` | Pre-cropped tooth upload, SSE stream |
| `POST` | `/api/search-fragment` | Re-rank using a subset of a previous query's teeth |
| `POST` | `/api/enrol` | Session-scoped enrolment of a new panoramic |
| `GET` | `/api/model-card` | Static evaluation numbers for the UI |

## Local development

```bash
# Build the registry first (only needed once)
python -m identification.scripts.build_registry

# Start the API
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

Loads all models in ~5 seconds. `backend/temp/` holds per-query overlay
images and is swept of entries older than 1 hour on next upload.

## Deployed mode (Hugging Face Space)

When `HF_REPO_ID` is set, every model + registry artefact is downloaded from
the pinned HF Hub model repo at boot — see `_resolve_artefacts()` in
`backend/pipeline.py`. Required env vars on the Space:

| Env var | Purpose |
| --- | --- |
| `HF_REPO_ID` | e.g. `<user>/rtg-tooth-id-weights` |
| `HF_REVISION` | Pinned commit SHA on the model repo (DO NOT use `main`) |
| `HF_TOKEN` | Read token, injected as a Space secret |
| `ALLOWED_ORIGINS` | Comma-separated list of frontend origins for CORS |
| `ALLOWED_ORIGIN_REGEX` | Regex for Vercel preview URLs (optional) |
| `MAX_UPLOAD_BYTES` | Default 15 MB; tighten further if abused |

The container runs `uvicorn backend.app:app --host 0.0.0.0 --port 7860`.
First build takes ~10 minutes (CPU torch + ultralytics + faiss-cpu wheels).

## Configuration

Defaults live in `backend/pipeline.py::PipelineConfig`. Override by editing
that file or instantiating a different `PipelineConfig` in `backend/app.py`.

| Field | Default | Notes |
| --- | --- | --- |
| `yolo_weights` | `runs-detection/train3/weights/best.pt` | Trained tooth detector |
| `fdi_classifier` | `identification/runs/tooth_fdi_raw/best.pt` | FDI classifier |
| `embedder` | `identification/runs/embedding_fdi_init_v1/best.pt` | Best embedder |
| `registry_dir` | YOLO-built registry | FAISS index + metadata; rollback via `DEMO_USE_YOLO_REGISTRY=0` |
| `top_k` | `5` | Number of candidates returned |
| `min_teeth_warning` | `4` | Threshold below which a warning event is emitted |
