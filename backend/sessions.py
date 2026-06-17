"""
Session-scoped enrolment storage.

A session is a directory under ``backend/sessions/{session_id}/`` containing
the same three files as a canonical registry::

    index.faiss            # FAISS IndexFlatIP on L2-normed mean-pooled embeddings
    index.ids.json         # parallel array of session-scoped person_ids
    registry_meta.json     # {n_persons, embedding_dim, aggregation, persons: [...]}

Each session-scoped person_id has the prefix ``session:`` to prevent collisions
with canonical IDs and to let downstream code (``/identify`` results, the
frontend session badge) tell them apart without an extra round-trip.

Sessions expire after 24h — ``cleanup_expired_sessions`` is called on every
mutating endpoint, and idempotently on read paths.

The schema deliberately mirrors ``identification/registry_ensemble_yolo/``
so that ``RetrievalIndex.load()`` works against either path with no changes,
and so the merge in ``/identify`` is just two ``index.search()`` calls
followed by a flat sort.

Concurrency: assume single-process FastAPI (the dev/demo deployment). For
multi-worker prod we'd need a file-lock per session.
"""
from __future__ import annotations

import json
import re
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from identification.models.retrieval_index import RetrievalIndex


SESSION_ID_RE = re.compile(r"^[a-f0-9]{1,32}$")
# Session person_ids are minted server-side as `session:` + 12 hex chars.
# Validated at every write/delete entry point as defense-in-depth.
SESSION_PID_RE = re.compile(r"^session:[a-f0-9]{12}$")
SESSION_TTL_SECONDS = 24 * 60 * 60  # 24h per spec
SESSION_PID_PREFIX = "session:"

# Cap to keep a runaway client from filling the disk during a forensic demo.
MAX_ENROLMENTS_PER_SESSION = 100

# Per-session locks for the read-modify-write window in add_enrolment /
# delete_enrolment. We're single-process FastAPI so a threading.Lock is enough;
# multi-worker prod would need fcntl.flock on a session-scoped lock file.
import threading
_session_locks: dict[str, threading.Lock] = {}
_session_locks_guard = threading.Lock()


def session_lock(session_id: str) -> threading.Lock:
    """Return a process-wide lock keyed on session_id (lazily created)."""
    with _session_locks_guard:
        lock = _session_locks.get(session_id)
        if lock is None:
            lock = threading.Lock()
            _session_locks[session_id] = lock
        return lock


def is_valid_session_pid(person_id: str) -> bool:
    return bool(SESSION_PID_RE.fullmatch(person_id))


@dataclass
class SessionInfo:
    session_id: str
    path: Path
    created_at: float  # unix seconds
    n_enrolments: int


def new_session_id() -> str:
    """Generate a fresh session id (16 hex chars)."""
    return uuid.uuid4().hex[:16]


def is_valid_session_id(session_id: str | None) -> bool:
    return bool(session_id) and bool(SESSION_ID_RE.fullmatch(session_id))


def session_dir(sessions_root: Path, session_id: str) -> Path:
    if not is_valid_session_id(session_id):
        raise ValueError(f"invalid session_id: {session_id!r}")
    return sessions_root / session_id


def _meta_path(sdir: Path) -> Path:
    return sdir / "registry_meta.json"


def _now() -> float:
    return time.time()


def session_exists(sessions_root: Path, session_id: str) -> bool:
    if not is_valid_session_id(session_id):
        return False
    return session_dir(sessions_root, session_id).is_dir()


def load_session_meta(sessions_root: Path, session_id: str) -> dict | None:
    """Return the session's registry_meta dict, or None if the session does
    not exist / is empty.
    """
    sdir = session_dir(sessions_root, session_id)
    mpath = _meta_path(sdir)
    if not mpath.exists():
        return None
    with open(mpath) as f:
        return json.load(f)


def session_index_exists(sessions_root: Path, session_id: str) -> bool:
    """True iff session has at least one enrolment (i.e. an index on disk).

    A session id can be valid + the directory can exist without any enrolments
    (e.g. directory was pre-created on a failed enrol). Callers should use
    this to gate `/identify` merge — no index → no merge.
    """
    sdir = session_dir(sessions_root, session_id)
    return (sdir / "index.faiss").exists() and (sdir / "index.ids.json").exists()


def load_session_index(
    sessions_root: Path, session_id: str, dim: int
) -> RetrievalIndex | None:
    """Load the session's FAISS index, or return None if it doesn't exist.

    Returns None (not raise) for the common case of "session has no
    enrolments yet" so callers can early-out.
    """
    if not session_index_exists(sessions_root, session_id):
        return None
    return RetrievalIndex.load(
        str(session_dir(sessions_root, session_id) / "index"),
        dim=dim,
    )


def list_session_enrolments(sessions_root: Path, session_id: str) -> list[dict]:
    """Return the persons array for this session (empty list if none)."""
    meta = load_session_meta(sessions_root, session_id)
    if meta is None:
        return []
    return list(meta.get("persons", []))


def add_enrolment(
    sessions_root: Path,
    session_id: str,
    *,
    person_id: str,
    fake_name: str,
    n_teeth: int,
    embedding: np.ndarray,
    embedding_dim: int,
    panoramic_bytes: bytes,
    note: str | None = None,
) -> dict:
    """Append a new enrolment to the session index. Returns the person dict.

    Caller must have already done duplicate detection — this writes
    unconditionally. ``embedding`` must be L2-normed shape (D,).
    """
    if embedding.ndim != 1 or embedding.shape[0] != embedding_dim:
        raise ValueError(
            f"embedding must be 1-D shape ({embedding_dim},), got {embedding.shape}"
        )
    if not is_valid_session_pid(person_id):
        raise ValueError(
            f"person_id must match {SESSION_PID_RE.pattern}, got {person_id!r}"
        )

    sdir = session_dir(sessions_root, session_id)
    sdir.mkdir(parents=True, exist_ok=True)

    # Load existing index if present, else create a fresh one.
    index = load_session_index(sessions_root, session_id, dim=embedding_dim)
    if index is None:
        index = RetrievalIndex(dim=embedding_dim)
        meta_persons: list[dict] = []
        created_at = _now()
    else:
        existing = load_session_meta(sessions_root, session_id) or {}
        meta_persons = list(existing.get("persons", []))
        created_at = float(existing.get("created_at", _now()))

    if len(meta_persons) >= MAX_ENROLMENTS_PER_SESSION:
        raise ValueError(
            f"session full: max {MAX_ENROLMENTS_PER_SESSION} enrolments"
        )

    # Add to FAISS — embedding must be (1, D) contiguous float32.
    index.add(embedding[None, :].astype(np.float32), [person_id])
    index.save(str(sdir / "index"))

    # Save panoramic next to the meta so we can re-render and re-hash it later.
    pano_path = sdir / f"{person_id.replace(SESSION_PID_PREFIX, '')}.png"
    pano_path.write_bytes(panoramic_bytes)

    person = {
        "person_id": person_id,
        "fake_name": fake_name,
        "n_teeth": int(n_teeth),
        "enrolled_at": _now(),
        "panoramic_filename": pano_path.name,
    }
    if note:
        person["note"] = note[:200]  # bounded
    meta_persons.append(person)

    meta = {
        "session_id": session_id,
        "created_at": created_at,
        "updated_at": _now(),
        "embedding_dim": embedding_dim,
        "aggregation": "mean",
        "n_persons": len(meta_persons),
        "persons": meta_persons,
    }
    with open(_meta_path(sdir), "w") as f:
        json.dump(meta, f, indent=2)

    return person


def delete_enrolment(
    sessions_root: Path, session_id: str, person_id: str, embedding_dim: int
) -> bool:
    """Delete a person from the session index. Returns True if a row was removed.

    Rebuilds the index from scratch — FAISS IndexFlatIP doesn't support
    in-place removal cleanly and we're dealing with at most a few dozen rows.
    """
    meta = load_session_meta(sessions_root, session_id)
    if meta is None:
        return False
    persons = list(meta.get("persons", []))
    keep = [p for p in persons if p["person_id"] != person_id]
    if len(keep) == len(persons):
        return False

    sdir = session_dir(sessions_root, session_id)

    # If nothing left, blow away the index files entirely.
    if not keep:
        for f in ("index.faiss", "index.ids.json"):
            p = sdir / f
            if p.exists():
                p.unlink()
        meta["persons"] = []
        meta["n_persons"] = 0
        meta["updated_at"] = _now()
        with open(_meta_path(sdir), "w") as f:
            json.dump(meta, f, indent=2)
        # Drop the panoramic, too.
        removed = next((p for p in persons if p["person_id"] == person_id), None)
        if removed and "panoramic_filename" in removed:
            pf = sdir / removed["panoramic_filename"]
            if pf.exists():
                pf.unlink()
        return True

    # Rebuild index. We need the embeddings — load existing index, then
    # reconstruct each kept row's vector, then add to a fresh index.
    old_index = load_session_index(sessions_root, session_id, dim=embedding_dim)
    if old_index is None:
        return False
    new_index = RetrievalIndex(dim=embedding_dim)
    for p in keep:
        try:
            old_pos = old_index.person_ids.index(p["person_id"])
        except ValueError:
            # Stale meta — skip; we'll write whatever survives.
            continue
        vec = old_index.index.reconstruct(old_pos)
        new_index.add(vec[None, :].astype(np.float32), [p["person_id"]])
    new_index.save(str(sdir / "index"))

    # Drop the deleted person's panoramic.
    removed = next((p for p in persons if p["person_id"] == person_id), None)
    if removed and "panoramic_filename" in removed:
        pf = sdir / removed["panoramic_filename"]
        if pf.exists():
            pf.unlink()

    meta["persons"] = keep
    meta["n_persons"] = len(keep)
    meta["updated_at"] = _now()
    with open(_meta_path(sdir), "w") as f:
        json.dump(meta, f, indent=2)
    return True


def cleanup_expired_sessions(sessions_root: Path) -> int:
    """Delete sessions older than SESSION_TTL_SECONDS (created_at). Returns
    the number of sessions removed.
    """
    if not sessions_root.exists():
        return 0
    cutoff = _now() - SESSION_TTL_SECONDS
    removed = 0
    for sdir in sessions_root.iterdir():
        if not sdir.is_dir() or not is_valid_session_id(sdir.name):
            continue
        meta = load_session_meta(sessions_root, sdir.name)
        # If we can't read meta, fall back to mtime so we don't leak the dir.
        created_at = (
            float(meta.get("created_at", 0.0)) if meta else sdir.stat().st_mtime
        )
        if created_at < cutoff:
            shutil.rmtree(sdir, ignore_errors=True)
            removed += 1
    return removed
