"""FastAPI backend for the Phase 6 demo.

Endpoints:
    GET  /api/health                            -> liveness check
    GET  /api/registry                          -> list of enrolled persons
    GET  /api/registry/{person_id}/panoramic   -> downloads xray.png
    GET  /api/intermediate/{query_id}/{file}   -> per-query overlay images
    POST /api/identify                          -> SSE stream of pipeline events

Run locally:
    uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse

from backend.pipeline import PipelineConfig, PipelineModels, cleanup_temp_dir, run_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]

app = FastAPI(title="Tooth Identification Demo", version="0.1.0")

# Allow the Next.js dev server (default port 3000) to call us.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config = PipelineConfig()
models = PipelineModels(config=config)


@app.on_event("startup")
def _startup() -> None:
    config.temp_dir.mkdir(parents=True, exist_ok=True)
    cleanup_temp_dir(config.temp_dir, max_age_seconds=0)
    models.load_all()


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "device": models.device,
        "registry_size": len(models.registry_index) if models.registry_index else 0,
    }


@app.get("/api/registry")
def list_registry() -> dict:
    """List all enrolled persons (for the registry browser)."""
    persons = [
        {
            "person_id": meta["person_id"],
            "fake_name": meta["fake_name"],
            "n_teeth": meta["n_teeth"],
        }
        for meta in models.registry_meta.values()
    ]
    persons.sort(key=lambda p: p["fake_name"])
    return {"n_persons": len(persons), "persons": persons}


@app.get("/api/registry/examples")
def list_examples() -> dict:
    """Return a stable, known-good list of demo picks.

    Sorted by (n_teeth descending, person_id) so the same panoramics surface
    every restart. Filtered to typical adult dentition (28–32 teeth) where the
    YOLO detector and FDI classifier perform best.
    """
    candidates = [
        m for m in models.registry_meta.values()
        if 28 <= m.get("n_teeth", 0) <= 32
    ]
    candidates.sort(key=lambda m: (-m["n_teeth"], m["person_id"]))
    picks = [
        {
            "person_id": m["person_id"],
            "fake_name": m["fake_name"],
            "n_teeth": m["n_teeth"],
        }
        for m in candidates[:8]
    ]
    return {"n_examples": len(picks), "examples": picks}


@app.get("/api/registry/{person_id}/panoramic")
def download_panoramic(person_id: str):
    """Download the panoramic X-ray for the given person as `xray.png`."""
    meta = models.registry_meta.get(person_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Person not found in registry")
    panoramic_path = PROJECT_ROOT / meta["panoramic_path"]
    if not panoramic_path.exists():
        raise HTTPException(status_code=404, detail="Panoramic file missing on disk")
    return FileResponse(
        panoramic_path,
        media_type="image/png",
        filename="xray.png",
    )


@app.get("/api/intermediate/{query_id}/{filename}")
def serve_intermediate(query_id: str, filename: str):
    """Serve intermediate overlay images written during a pipeline run."""
    if "/" in query_id or "/" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid path")
    target = config.temp_dir / query_id / filename
    if not target.exists():
        raise HTTPException(status_code=404, detail="Intermediate file not found")
    return FileResponse(target, media_type="image/png")


@app.post("/api/identify")
async def identify(file: UploadFile = File(...)) -> EventSourceResponse:
    """Run the identification pipeline on the uploaded image, stream SSE events."""
    cleanup_temp_dir(config.temp_dir)

    query_id = uuid.uuid4().hex[:12]
    query_dir = config.temp_dir / query_id
    query_dir.mkdir(parents=True, exist_ok=True)

    upload_path = query_dir / "upload.png"
    with open(upload_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    async def event_stream():
        try:
            async for event in run_pipeline(upload_path, query_id, models):
                yield {
                    "event": event["event"],
                    "data": json.dumps(event["data"]),
                }
        except Exception as exc:  # noqa: BLE001 — surface any pipeline failure to the UI
            yield {
                "event": "error",
                "data": json.dumps({"message": f"Pipeline failed: {exc}"}),
            }

    return EventSourceResponse(event_stream())
