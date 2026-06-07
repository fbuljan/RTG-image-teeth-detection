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

import csv
import json
import shutil
import uuid
from pathlib import Path

import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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


def _load_csv(path: Path) -> list[dict]:
    """Load a CSV into a list of dicts, casting numeric fields to floats."""
    if not path.exists():
        return []
    rows: list[dict] = []
    with open(path) as f:
        for row in csv.DictReader(f):
            casted: dict = {}
            for k, v in row.items():
                if v is None:
                    casted[k] = None
                    continue
                try:
                    casted[k] = float(v)
                except (TypeError, ValueError):
                    casted[k] = v
            rows.append(casted)
    return rows


def _build_model_card() -> dict:
    """Assemble static facts about the deployed embedder for the UI."""
    run_dir = config.embedder.parent
    eval_metrics_path = run_dir / "eval_test" / "metrics.json"
    person_retrieval_path = run_dir / "analysis" / "person_retrieval" / "metrics.json"
    category_csv = run_dir / "analysis" / "per_tooth" / "per_category_metrics.csv"
    subgroup_csv = run_dir / "analysis" / "subgroups" / "all_subgroups.csv"
    config_yaml = run_dir / "config.yaml"

    card: dict = {
        "checkpoint": str(config.embedder.relative_to(PROJECT_ROOT)),
        "run_name": run_dir.name,
    }

    if eval_metrics_path.exists():
        with open(eval_metrics_path) as f:
            card["eval_test"] = json.load(f)

    if person_retrieval_path.exists():
        with open(person_retrieval_path) as f:
            payload = json.load(f)
        # Trim to mean-pooling sweep entries (the headline thesis result).
        sweep = [s for s in payload.get("sweep", []) if s.get("method") == "mean"]
        sweep.sort(key=lambda s: s["n_query"])
        card["multi_tooth_sweep"] = sweep
        # Forensic 1-vs-aggregated, mean gallery only.
        forensic = [
            f for f in payload.get("forensic_1tooth", [])
            if f.get("method") == "single_query_mean_gallery"
        ]
        card["forensic_1tooth"] = forensic

    card["per_category"] = _load_csv(category_csv)
    card["subgroups"] = _load_csv(subgroup_csv)

    if config_yaml.exists():
        with open(config_yaml) as f:
            full_cfg = yaml.safe_load(f) or {}
        card["training"] = {
            "backbone": full_cfg.get("model", {}).get("backbone"),
            "embedding_dim": full_cfg.get("model", {}).get("embedding_dim"),
            "dropout": full_cfg.get("model", {}).get("dropout"),
            "loss": full_cfg.get("loss", {}).get("type"),
            "loss_margin": full_cfg.get("loss", {}).get("margin"),
            "miner": full_cfg.get("miner", {}).get("type"),
            "optimizer": full_cfg.get("train", {}).get("optimizer"),
            "lr": full_cfg.get("train", {}).get("lr"),
            "scheduler": full_cfg.get("train", {}).get("scheduler"),
            "epochs": full_cfg.get("train", {}).get("epochs"),
            "weight_decay": full_cfg.get("train", {}).get("weight_decay"),
            "warmup_epochs": full_cfg.get("train", {}).get("warmup_epochs"),
            "sampler_p": full_cfg.get("sampler", {}).get("p"),
            "sampler_k": full_cfg.get("sampler", {}).get("k"),
            "crop_mode": full_cfg.get("data", {}).get("crop_mode"),
            "init_from_classifier": full_cfg.get("init_from_classifier"),
        }

    # YOLO detector + segmenter metrics from the saved validation summary.
    yolo_summary_path = PROJECT_ROOT / "runs-segmentation" / "metrics_summary.json"
    if yolo_summary_path.exists():
        with open(yolo_summary_path) as f:
            card["yolo"] = json.load(f)

    # Phase 7.1 ensemble metrics — both the GT-crop eval and the YOLO-crop
    # deployment-aligned eval (saved by evaluate_ensemble.py with --manifest).
    def _load_ensemble_payload(rel_path: str) -> dict | None:
        path = PROJECT_ROOT / rel_path
        if not path.exists():
            return None
        with open(path) as f:
            payload = json.load(f)
        sweep = [s for s in payload.get("sweep", []) if s.get("method") == "mean"]
        sweep.sort(key=lambda s: s["n_query"])
        forensic = [
            f for f in payload.get("forensic_1tooth", [])
            if f.get("method", "").startswith("single_query_mean")
        ]
        peak_per_method: dict[str, dict] = {}
        for s in payload.get("sweep", []):
            if s.get("n_query") == 16:
                peak_per_method[s["method"]] = s
        return {
            "members": list(payload.get("checkpoints", {}).keys()),
            "weights": payload.get("weights"),
            "multi_tooth_sweep": sweep,
            "forensic_1tooth": forensic,
            "peak_per_method": peak_per_method,
        }

    gt_ensemble = _load_ensemble_payload(
        "identification/runs/ensemble_v1/analysis/person_retrieval/metrics.json"
    )
    yolo_ensemble = _load_ensemble_payload(
        "identification/runs/ensemble_v1/analysis/person_retrieval_yolo/metrics.json"
    )
    if gt_ensemble is not None:
        card["ensemble"] = gt_ensemble  # headline / thesis numbers
    if yolo_ensemble is not None:
        card["ensemble_yolo"] = yolo_ensemble  # deployment-aligned numbers

    card["registry_size"] = len(models.registry_index) if models.registry_index else 0
    card["default_mode"] = config.default_mode
    card["ensemble_available"] = bool(models.ensemble_models)
    return card


_model_card_cache: dict | None = None


@app.get("/api/model-card")
def get_model_card() -> dict:
    """Return all the per-run metrics the UI needs to render the model card."""
    global _model_card_cache
    if _model_card_cache is None:
        _model_card_cache = _build_model_card()
    return _model_card_cache


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
async def identify(
    file: UploadFile = File(...),
    mode: str = Form("segmentation"),
) -> EventSourceResponse:
    """Run the identification pipeline on the uploaded image, stream SSE events.

    `mode` selects between "detection" and "segmentation" YOLO backends.
    The ensemble path is intentionally not exposed to the live demo (kept as
    an offline experiment); single-model FDI-init is the only deployed mode.
    """
    cleanup_temp_dir(config.temp_dir)

    if mode not in ("detection", "segmentation"):
        mode = config.default_mode
    ensemble_flag = False

    query_id = uuid.uuid4().hex[:12]
    query_dir = config.temp_dir / query_id
    query_dir.mkdir(parents=True, exist_ok=True)

    upload_path = query_dir / "upload.png"
    with open(upload_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    async def event_stream():
        try:
            async for event in run_pipeline(
                upload_path, query_id, models, mode=mode, ensemble=ensemble_flag,
            ):
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
