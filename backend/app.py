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
import io
import json
import re
import shutil
import uuid
from pathlib import Path

import yaml
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse

from backend.pipeline import (
    PipelineConfig,
    PipelineModels,
    cleanup_temp_dir,
    compute_query_vector_sync,
    run_crops_pipeline,
    run_fragment_search,
    run_pipeline,
)
from backend import sessions as session_store
from backend.pipeline import _open_set_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]

app = FastAPI(title="Tooth Identification Demo", version="0.1.0")

# Allow the Next.js dev server (default port 3000, plus 3005 as an alt port
# used during audits when the canonical 3000 is occupied by another process).
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3005",
        "http://127.0.0.1:3005",
    ],
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
    # Phase 9.7 — sweep expired sessions at boot. Without this, a session
    # whose user never returns leaks disk space forever (the 24h TTL only
    # fires when an enrol/list/delete endpoint is hit).
    config.sessions_dir.mkdir(parents=True, exist_ok=True)
    removed = session_store.cleanup_expired_sessions(config.sessions_dir)
    if removed:
        print(f"[app] swept {removed} expired session(s) at startup")


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

    # Multi-tooth sweep — DEPLOYED pipeline (Phase 8.0 YOLO-built registry).
    # Headline R1 @ n=16 = 82.6% [79.8, 86.0]. This is the same artefact the
    # deployment headline tooltip cites, so the on-page table is internally
    # consistent. The legacy Phase 5 single-model GT-crop sweep (which used
    # to power this table) reported 55.1% — superseded by the YOLO-built
    # registry rebuild; preserved on disk under
    # `embedding_fdi_init_v1/analysis/person_retrieval/metrics.json` for the
    # GT-only ensemble comparison block (loaded separately below as the
    # offline ensemble vs single-model anchor).
    deployed_sweep_path = (
        PROJECT_ROOT
        / "identification/runs/phase8_deployed_yolo_reg/yolo_eval.json"
    )
    if deployed_sweep_path.exists():
        with open(deployed_sweep_path) as f:
            deployed = json.load(f)
        # Normalize the deployed-sweep schema to match what the frontend
        # SweepEntry type expects (rank1_mean/rank5_mean/rank10_mean +
        # method tag). The deployed artefact omits mAP — fmtPct renders an
        # em-dash for null cells, which is honest: mAP wasn't measured in
        # this sweep protocol.
        sweep = []
        for row in sorted(deployed.get("sweep_full_registry", []), key=lambda r: r["n_query"]):
            sweep.append({
                "n_query": row["n_query"],
                "method": "mean",  # mean-pool aggregation; matches frontend filter
                "n_persons": row.get("n_persons"),
                "n_trials": row.get("n_trials"),
                "rank1_mean": row.get("rank1_mean"),
                "rank1_std": None,  # deployed artefact uses bootstrap CI instead
                "rank1_ci95_low": row.get("rank1_ci95_low"),
                "rank1_ci95_high": row.get("rank1_ci95_high"),
                "rank5_mean": row.get("rank5_mean"),
                "rank5_std": None,
                "rank10_mean": row.get("rank10_mean"),
                "rank10_std": None,
                "mAP_mean": None,  # not measured in deployed sweep protocol
                "mAP_std": None,
            })
        card["multi_tooth_sweep"] = sweep

    # The legacy Phase 5 GT-crop single-model sweep is still loaded as a
    # comparator for the offline GT ensemble block (EnsembleSweepTable's
    # "Single R-1" column). It does NOT drive the headline multi-tooth
    # block above — that's now the deployed YOLO-built-registry numbers.
    if person_retrieval_path.exists():
        with open(person_retrieval_path) as f:
            payload = json.load(f)
        legacy_sweep = [s for s in payload.get("sweep", []) if s.get("method") == "mean"]
        legacy_sweep.sort(key=lambda s: s["n_query"])
        card["multi_tooth_sweep_gt_anchor"] = legacy_sweep
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
    session_id: str | None = Header(None, alias="X-Session-Id"),
) -> EventSourceResponse:
    """Run the identification pipeline on the uploaded image, stream SSE events.

    `mode` selects between "detection" and "segmentation" YOLO backends.
    The ensemble path is intentionally not exposed to the live demo (kept as
    an offline experiment); single-model FDI-init is the only deployed mode.

    Phase 9.7 — when `X-Session-Id` is present and the session has ≥1
    enrolment, the session's FAISS index is merged into the top-K alongside
    the canonical 1,178-person index. Calibrated open-set / percentile / age
    are always computed off the canonical top-1 (not the merged top-1) — the
    Phase 8.6 calibration is canonical-only.
    """
    cleanup_temp_dir(config.temp_dir)

    if mode not in ("detection", "segmentation"):
        mode = config.default_mode
    ensemble_flag = False

    # Validate session_id; silently drop it on bad input rather than 400-ing
    # the whole identify flow (clients may speculatively send a header).
    effective_session_id = (
        session_id if session_store.is_valid_session_id(session_id) else None
    )

    query_id = uuid.uuid4().hex[:12]
    query_dir = config.temp_dir / query_id
    query_dir.mkdir(parents=True, exist_ok=True)

    upload_path = query_dir / "upload.png"
    with open(upload_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    async def event_stream():
        try:
            async for event in run_pipeline(
                upload_path,
                query_id,
                models,
                mode=mode,
                ensemble=ensemble_flag,
                session_id=effective_session_id,
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


# Phase 9.5 — fragment re-search: re-pool a subset of cached tooth embeddings
# without re-running detection/embedding. Sub-100ms response.

from pydantic import BaseModel  # noqa: E402


class FragmentSearchRequest(BaseModel):
    query_id: str
    tooth_indices: list[int]


_QUERY_ID_RE = re.compile(r"^[a-f0-9]{1,32}$")


@app.post("/api/search-fragment")
def search_fragment(req: FragmentSearchRequest) -> dict:
    """Re-search the registry using a subset of the previous query's teeth."""
    # Phase 9.5.1 — guard against path traversal. The /identify endpoint mints
    # query_id from uuid.uuid4().hex[:12] so it's always lowercase hex; mirror
    # the /api/intermediate validation rather than passing arbitrary strings
    # straight into a filesystem join.
    if not _QUERY_ID_RE.fullmatch(req.query_id):
        raise HTTPException(status_code=400, detail="Invalid query_id")
    try:
        return run_fragment_search(
            query_id=req.query_id,
            tooth_indices=req.tooth_indices,
            models=models,
            config=config,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ---------- Phase 9.6 — pre-cropped tooth upload ----------


# Bound the number of crops per request — matches the FDI label space (32)
# plus a small buffer. Anything larger means the user is uploading garbage.
MAX_CROPS_PER_QUERY = 32
# Per-crop file size. Tooth crops are typically tens of KB; cap at 5 MB so
# a single bad file doesn't OOM the embedder.
MAX_CROP_BYTES = 5 * 1024 * 1024


@app.post("/api/identify-crops")
async def identify_crops(
    files: list[UploadFile] = File(...),
    fdi_overrides_json: str | None = Form(None),
    session_id: str | None = Header(None, alias="X-Session-Id"),
) -> EventSourceResponse:
    """Phase 9.6 — identify from pre-cropped tooth images.

    Each file in `files` is a single tooth crop (PNG/JPG). `fdi_overrides_json`
    is an optional JSON-encoded array of length `len(files)` where each entry
    is either a string FDI label (e.g. "11", "23") or null (auto-detect).

    Reuses the canonical embedder + FAISS index. Calibration semantics are
    inherited from /api/identify — open-set, percentile, age all key off the
    canonical sims[0]. Session merge respected when `X-Session-Id` is set.
    """
    cleanup_temp_dir(config.temp_dir)

    if not files:
        raise HTTPException(status_code=400, detail="No files supplied")
    if len(files) > MAX_CROPS_PER_QUERY:
        raise HTTPException(
            status_code=413,
            detail=f"Too many crops — max {MAX_CROPS_PER_QUERY} per query.",
        )

    fdi_overrides: list[str | None] | None = None
    if fdi_overrides_json:
        try:
            parsed = json.loads(fdi_overrides_json)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"fdi_overrides_json invalid: {exc}")
        if not isinstance(parsed, list) or len(parsed) != len(files):
            raise HTTPException(
                status_code=400,
                detail="fdi_overrides_json must be a JSON array with one entry per file",
            )
        normalised: list[str | None] = []
        for v in parsed:
            if v is None or v == "":
                normalised.append(None)
            elif isinstance(v, str):
                normalised.append(v.strip() or None)
            else:
                raise HTTPException(status_code=400, detail="fdi_overrides_json entries must be strings or null")
        fdi_overrides = normalised

    effective_session_id = (
        session_id if session_store.is_valid_session_id(session_id) else None
    )

    # Read + validate every upload into a PIL Image up-front. Cheaper than
    # streaming and the OOD gate needs them all before any of them can run
    # through the embedder.
    from PIL import Image as _Image  # local import keeps PIL out of startup path
    crop_images = []
    for i, uf in enumerate(files):
        data = await uf.read()
        if len(data) > MAX_CROP_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"file {i} exceeds {MAX_CROP_BYTES // (1024*1024)} MB cap",
            )
        try:
            img = _Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"file {i} is not a readable image: {exc}")
        crop_images.append(img)

    query_id = uuid.uuid4().hex[:12]

    async def event_stream():
        try:
            async for event in run_crops_pipeline(
                crop_images,
                query_id,
                models,
                fdi_overrides=fdi_overrides,
                session_id=effective_session_id,
            ):
                yield {"event": event["event"], "data": json.dumps(event["data"])}
        except Exception as exc:  # noqa: BLE001
            yield {
                "event": "error",
                "data": json.dumps({"message": f"Crops pipeline failed: {exc}"}),
            }

    return EventSourceResponse(event_stream())


# ---------- Phase 9.7 — session enrolment ----------
#
# Three endpoints, all keyed on `X-Session-Id` header (a 16-char hex UUID
# minted client-side and persisted in localStorage). Storage layout mirrors
# the canonical registry — see backend/sessions.py.
#
# Calibrated open-set / percentile / age numbers from Phase 8.6/8.10 are NOT
# carried through to session enrolments — calibration was learned on the
# canonical 1,178-person distribution and is not transferable. The duplicate
# detector uses the same _open_set_score helper to compute a session-aware
# z-score on the would-be enrolment's top-1, but the answer is consumed only
# as a UX hint (yellow banner), never as a verdict.

DUPLICATE_Z_THRESHOLD = 0.7
SESSION_ID_HEADER = "X-Session-Id"
ENROLMENT_NAME_MAX_LEN = 40
# Panoramic uploads are normally 1-5 MB. Cap at 25 MB to defang an OOM
# attack — a malicious client could otherwise POST gigabyte-sized PNGs and
# materialize them in memory inside `await file.read()`.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


def _require_session_id(session_id: str | None) -> str:
    """Validate the session id, 400 if it's malformed."""
    if not session_store.is_valid_session_id(session_id):
        raise HTTPException(
            status_code=400,
            detail=f"Missing or invalid {SESSION_ID_HEADER} header",
        )
    return session_id  # type: ignore[return-value]


def _session_top1(query_vec, session_id: str) -> tuple[float | None, str | None]:
    """Return (top1_sim, top1_pid) when the session has an index, else (None, None)."""
    if not session_store.session_index_exists(config.sessions_dir, session_id):
        return None, None
    idx = session_store.load_session_index(
        config.sessions_dir, session_id, dim=query_vec.shape[0]
    )
    if idx is None or len(idx) == 0:
        return None, None
    sims, ids = idx.search(query_vec, k=1)
    return float(sims[0]), str(ids[0])


@app.post("/api/enrol")
async def enrol(
    file: UploadFile = File(...),
    fake_name: str = Form(...),
    note: str | None = Form(None),
    mode: str = Form("segmentation"),
    force: bool = Form(False),
    session_id: str | None = Header(None, alias=SESSION_ID_HEADER),
) -> dict:
    """Embed an uploaded panoramic and add it to the caller's session index.

    `force=True` skips duplicate detection — used by the frontend "Enrol
    anyway" button after the yellow banner.
    """
    sid = _require_session_id(session_id)
    session_store.cleanup_expired_sessions(config.sessions_dir)

    # Validate fake_name (≤40 chars, non-empty, no control chars).
    fake_name = (fake_name or "").strip()
    if not fake_name:
        raise HTTPException(status_code=400, detail="fake_name is required")
    if len(fake_name) > ENROLMENT_NAME_MAX_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"fake_name must be ≤{ENROLMENT_NAME_MAX_LEN} characters",
        )
    if any(ord(c) < 32 for c in fake_name):
        raise HTTPException(status_code=400, detail="fake_name contains control characters")

    if mode not in ("detection", "segmentation"):
        mode = config.default_mode

    # Persist the upload so we can both (a) hash it for provenance later
    # against the canonical registry and (b) keep a thumbnail next to the
    # session entry. We use the session dir for both.
    pano_bytes = await file.read()
    if len(pano_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MB cap",
        )
    # Per-request tmp filename — two concurrent enrols on the same session
    # would otherwise both write to `_tmp_upload.png` and clobber each
    # other's embedder input.
    tmp_pano = config.sessions_dir / sid / f"_tmp_{uuid.uuid4().hex[:16]}.png"
    tmp_pano.parent.mkdir(parents=True, exist_ok=True)
    tmp_pano.write_bytes(pano_bytes)

    try:
        embed_result = compute_query_vector_sync(tmp_pano, models, mode=mode)
    except ValueError as exc:
        if tmp_pano.exists():
            tmp_pano.unlink()
        raise HTTPException(status_code=422, detail=str(exc))

    query_vec = embed_result["query_vec"]
    n_teeth = embed_result["n_teeth"]
    embedding_dim = int(query_vec.shape[0])

    # Serialize the read-modify-write window for this session so two
    # concurrent enrols don't both read the same baseline meta + index and
    # then clobber each other's writes.
    with session_store.session_lock(sid):
        return _enrol_locked(
            sid=sid,
            query_vec=query_vec,
            n_teeth=n_teeth,
            embedding_dim=embedding_dim,
            pano_bytes=pano_bytes,
            fake_name=fake_name,
            note=note,
            force=force,
            tmp_pano=tmp_pano,
        )


def _enrol_locked(
    *,
    sid: str,
    query_vec,
    n_teeth: int,
    embedding_dim: int,
    pano_bytes: bytes,
    fake_name: str,
    note: str | None,
    force: bool,
    tmp_pano: Path,
) -> dict:
    """Caller must hold session_store.session_lock(sid)."""
    # Duplicate detection: combine canonical + session top-1 sims to compute
    # the highest similarity across both indexes, then z-score it against the
    # locked Phase 8.6 calibration. Only the z-score (not the canonical
    # verdict) is used here.
    canonical_sims, canonical_ids = models.registry_index.search(
        query_vec, k=1
    )
    canonical_top1_sim = float(canonical_sims[0])
    canonical_top1_pid = str(canonical_ids[0])

    session_top1_sim, session_top1_pid = _session_top1(query_vec, sid)

    if session_top1_sim is not None and session_top1_sim > canonical_top1_sim:
        max_top1_sim, max_top1_pid, max_top1_source = (
            session_top1_sim,
            session_top1_pid,
            "session",
        )
    else:
        max_top1_sim, max_top1_pid, max_top1_source = (
            canonical_top1_sim,
            canonical_top1_pid,
            "canonical",
        )

    open_set_score, _open_set_decision = _open_set_score(
        max_top1_sim, models.open_set_calibration
    )

    if (
        not force
        and open_set_score is not None
        and open_set_score > DUPLICATE_Z_THRESHOLD
    ):
        # Look up the matched person's display name for the banner.
        if max_top1_source == "session":
            session_meta = session_store.load_session_meta(config.sessions_dir, sid) or {}
            matched = next(
                (p for p in session_meta.get("persons", []) if p["person_id"] == max_top1_pid),
                None,
            )
            matched_name = matched.get("fake_name", max_top1_pid) if matched else max_top1_pid
        else:
            matched_meta = models.registry_meta.get(max_top1_pid, {})
            matched_name = matched_meta.get("fake_name", max_top1_pid)
        # Don't leak the temp upload back to the client — cleaner to require
        # them to re-POST with force=True.
        if tmp_pano.exists():
            tmp_pano.unlink()
        return {
            "status": "duplicate_likely",
            "duplicate_z_threshold": DUPLICATE_Z_THRESHOLD,
            "open_set_score": open_set_score,
            "matched_person_id": max_top1_pid,
            "matched_fake_name": matched_name,
            "matched_source": max_top1_source,
            "matched_similarity": max_top1_sim,
            "n_teeth": n_teeth,
        }

    # No duplicate (or caller forced through). Persist the enrolment.
    person_id = f"{session_store.SESSION_PID_PREFIX}{uuid.uuid4().hex[:12]}"
    try:
        person = session_store.add_enrolment(
            config.sessions_dir,
            sid,
            person_id=person_id,
            fake_name=fake_name,
            n_teeth=n_teeth,
            embedding=query_vec,
            embedding_dim=embedding_dim,
            panoramic_bytes=pano_bytes,
            note=note,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        if tmp_pano.exists():
            tmp_pano.unlink()

    return {
        "status": "enrolled",
        "person": person,
        "open_set_score_at_enrol": open_set_score,
    }


@app.get("/api/enrol")
def list_enrolments(
    session_id: str | None = Header(None, alias=SESSION_ID_HEADER),
) -> dict:
    """List the caller's session enrolments."""
    sid = _require_session_id(session_id)
    session_store.cleanup_expired_sessions(config.sessions_dir)
    persons = session_store.list_session_enrolments(config.sessions_dir, sid)
    return {
        "session_id": sid,
        "n_persons": len(persons),
        "persons": persons,
    }


@app.delete("/api/enrol/{person_id}")
def delete_enrolment_endpoint(
    person_id: str,
    session_id: str | None = Header(None, alias=SESSION_ID_HEADER),
) -> dict:
    """Remove a single enrolment from the caller's session."""
    sid = _require_session_id(session_id)
    if not session_store.is_valid_session_pid(person_id):
        raise HTTPException(status_code=400, detail="Invalid person_id")

    # Get embedding dim from the canonical index — sessions inherit it.
    dim = (
        models.registry_index.index.d
        if models.registry_index is not None
        else 128
    )
    with session_store.session_lock(sid):
        removed = session_store.delete_enrolment(
            config.sessions_dir, sid, person_id, embedding_dim=dim
        )
    if not removed:
        raise HTTPException(status_code=404, detail="Enrolment not found")
    return {"status": "deleted", "person_id": person_id}


@app.delete("/api/enrol")
def clear_session(
    session_id: str | None = Header(None, alias=SESSION_ID_HEADER),
) -> dict:
    """Drop all enrolments for the caller's session.

    Wipes the session dir wholesale — cheaper than repeatedly rebuilding the
    FAISS index per enrolment.
    """
    sid = _require_session_id(session_id)
    with session_store.session_lock(sid):
        sdir = session_store.session_dir(config.sessions_dir, sid)
        persons = session_store.list_session_enrolments(config.sessions_dir, sid)
        n_removed = len(persons)
        if sdir.exists():
            shutil.rmtree(sdir, ignore_errors=True)
    return {"status": "cleared", "n_removed": n_removed}


@app.get("/api/session/new")
def mint_session_id() -> dict:
    """Mint a new session id for clients that don't have one yet."""
    return {"session_id": session_store.new_session_id()}
