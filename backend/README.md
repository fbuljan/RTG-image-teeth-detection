# Backend — FastAPI demo server

End-to-end identification pipeline with SSE streaming. Loads YOLO + FDI
classifier + tooth embedder + FAISS registry once on startup and serves the
following endpoints:

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/health` | Liveness check (returns device + registry size) |
| `GET` | `/api/registry` | List all enrolled persons (fake names + tooth counts) |
| `GET` | `/api/registry/{person_id}/panoramic` | Download the panoramic as `xray.png` |
| `GET` | `/api/intermediate/{query_id}/{file}` | Per-query overlay images |
| `POST` | `/api/identify` | Multipart upload, returns SSE stream |

## Running

The server expects the conda env at `~/.../miniforge/base/envs/rtg` and the
identification registry at `identification/registry/` (built by
`identification.scripts.build_registry`).

```bash
# Build the registry first (only needed once)
python -m identification.scripts.build_registry

# Then start the API
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

It typically takes ~5 seconds to load all models on startup. The temp dir
(`backend/temp/`) holds intermediate overlay images per query and is cleaned
up on next upload (entries older than 1 hour are deleted).

## Configuration

Defaults live in `backend/pipeline.py::PipelineConfig`. Override by editing
that file or instantiating a different `PipelineConfig` in `backend/app.py`.

| Field | Default | Notes |
| --- | --- | --- |
| `yolo_weights` | `runs-detection/train3/weights/best.pt` | Trained tooth detector |
| `fdi_classifier` | `identification/runs/tooth_fdi_raw/best.pt` | FDI classifier |
| `embedder` | `identification/runs/embedding_fdi_init_v1/best.pt` | Best embedder |
| `registry_dir` | `identification/registry` | FAISS index + metadata |
| `top_k` | `5` | Number of candidates returned |
| `min_teeth_warning` | `4` | Threshold below which a warning event is emitted |
