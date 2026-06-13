// Backend API URL — defaults to local FastAPI server.
export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";

export type RegistryPerson = {
  person_id: string;
  fake_name: string;
  n_teeth: number;
};

export type RegistryListResponse = {
  n_persons: number;
  persons: RegistryPerson[];
};

export type SearchResult = {
  rank: number;
  person_id: string;
  fake_name: string;
  n_teeth: number | null;
  similarity: number;
  // Phase 9.3 — empirical percentile of this similarity against the 740
  // in-registry sim_top1 values from Phase 8.6 held-out enrolment.
  // Phase 9.7 — null when this result is a session enrolment OR not the
  // canonical top-1; calibration is canonical-only.
  similarity_percentile?: number | null;
  // Phase 9.7 — true when this result came from the caller's session
  // enrolments rather than the canonical 1,178-person registry.
  is_session?: boolean;
};

// Phase 9.2 — Phase 8.6 locked open-set decision.
export type OpenSetDecision = "in_registry" | "rejected" | "unknown";

// Phase 9.2 — provenance of the uploaded query.
// "self_match"  — exact bytes of an enrolled panoramic (tautological match).
// "novel"       — bytes do not match any enrolled image.
// "heldout"     — reserved for Phase 9.8 curated OOS picks.
// "unknown"     — could not classify (filesystem error etc.).
export type QueryProvenance = "self_match" | "novel" | "heldout" | "unknown";

// Phase 9.9 — embed-stage progress events now carry the FDI labels +
// confidences of teeth just embedded, so the UI can show "13, 12, 11…" live
// instead of an opaque counter.
export type EmbedProgressItem = {
  fdi: string;
  fdi_confidence: number;
};

export type StageEvent =
  | { event: "stage_start"; data: { stage: string; message: string; total?: number } }
  | { event: "stage_complete"; data: StageCompleteData }
  | {
      event: "progress";
      data: {
        stage: string;
        current: number;
        total: number;
        embedded?: EmbedProgressItem[];
      };
    }
  | { event: "warning"; data: { code: string; message: string } }
  | { event: "error"; data: { message: string } }
  | { event: "done"; data: Record<string, unknown> };

export type ToothContribution = {
  fdi: string;
  fdi_confidence: number;
  // Phase 9.9 — null for the crops path (no YOLO) and for cached fragment
  // queries written before the cache schema was widened; UI renders em-dash.
  yolo_confidence?: number | null;
  similarity_to_top1: number;
};

// Phase 9.9 — structured drop record (one entry per tooth lost to FDI dedup).
// Replaces the bare `n_dropped` count on the search-stage payload while
// preserving the count for compact summaries.
export type DropReason = {
  fdi: string;
  reason: "duplicate";
  fdi_confidence: number;
  yolo_confidence: number | null;
  kept_index: number | null;
  kept_fdi_confidence: number | null;
};

export type StageCompleteData = {
  stage: string;
  n_teeth?: number;
  n_uncertain?: number;
  n_dropped?: number;
  // Phase 9.9 — structured drop list (added on `fdi` and `search` stage_complete
  // events; absent on others).
  dropped?: DropReason[];
  n_embeddings?: number;
  annotated_image_url?: string;
  results?: SearchResult[];
  confidence?: "high" | "medium" | "uncertain" | "low";
  top1_top2_gap?: number;
  timings_ms?: Record<string, number>;
  n_query_teeth?: number;
  elapsed_ms?: number;
  tooth_contributions?: ToothContribution[];
  // Phase 9.2/9.3 — calibrated open-set + provenance (locked Phase 8.6 cal).
  open_set_score?: number | null;
  open_set_decision?: OpenSetDecision;
  open_set_threshold?: number | null;
  query_provenance?: QueryProvenance;
  expected_person_id?: string | null;
  sim_top1_percentile?: number | null;
  // Phase 9.4 — Phase 8.10 age estimate (sex head NOT wired; failed Pass).
  age_estimate?: AgeEstimate | null;
  // Phase 9.5 — emitted on `embed` and `search` stage_complete events. Used
  // by FragmentSelector to cache the query embeddings per query_id, then
  // POST /api/search-fragment for sub-100ms re-pooling on a subset.
  query_id?: string;
  per_tooth?: PerTooth[];
  // Phase 9.6 — pre-cropped tooth upload. `validate` stage replaces
  // detect/fdi for the crops path; `crops_mode` is set on the search-stage
  // event so the UI can switch the results-header copy.
  n_uploaded?: number;
  n_failed_ood?: number;
  n_kept?: number;
  n_dropped_duplicates?: number;
  per_crop?: PerCrop[];
  crops_mode?: boolean;
};

// Phase 9.6 — per-input-crop record emitted by the validate stage.
export type PerCrop = {
  input_index: number;
  auto_fdi: string;
  auto_fdi_confidence: number;
  chosen_fdi: string;
  source: "auto" | "override";
  failed_ood: boolean;
  dropped_as_duplicate: boolean;
  kept: boolean;
};

export type AgeEstimate = {
  // Raw model output, un-clamped. Kept for transparency under expert details.
  value: number;
  // Clamped to training range [6, 18] for headline display.
  value_display: number;
  ci_low: number;
  ci_high: number;
  ci_half: number;
  // True when prediction was outside the dense 6-13y bucket OR hit the
  // training-range boundary. Display widens the CI and the chip becomes neutral.
  saturation_risk: boolean;
  // Number of teeth pooled into the query (16 = full panoramic, < 8 = fragment).
  pool_size: number;
  small_pool: boolean;
  training_range: [number, number];
};

// Phase 9.5 — per-tooth metadata emitted by the embed stage so the frontend
// can let the user pick a subset for re-search.
export type PerTooth = {
  index: number;
  fdi: string;
  fdi_confidence: number;
  // Phase 9.9 — surfaced in the technical-details per-tooth table. Optional
  // for back-compat with the crops path (no YOLO involved).
  yolo_confidence?: number | null;
  bbox: [number, number, number, number];
};

// Phase 9.5 — POST /api/search-fragment payload shape (mirrors search-stage data).
export type FragmentSearchResponse = StageCompleteData & {
  query_id: string;
  tooth_indices: number[];
};

export async function searchFragment(
  queryId: string,
  toothIndices: number[],
): Promise<FragmentSearchResponse> {
  const res = await fetch(`${API_BASE}/api/search-fragment`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query_id: queryId, tooth_indices: toothIndices }),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`Fragment search failed: ${res.status} ${text}`);
  }
  return res.json();
}

export async function fetchRegistry(): Promise<RegistryListResponse> {
  const res = await fetch(`${API_BASE}/api/registry`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch registry: ${res.status}`);
  return res.json();
}

// ---------- Phase 9.7 — session enrolment ----------

const SESSION_STORAGE_KEY = "dental-demo.session_id";

/** Return the caller's session id, minting + persisting a new one on first
 *  use. Lives in localStorage so it survives reloads but is scoped per
 *  browser. The backend regex requires lowercase hex, ≤32 chars. */
export async function getOrMintSessionId(): Promise<string> {
  if (typeof window === "undefined") {
    // SSR / tests — generate but don't persist.
    const r = await fetch(`${API_BASE}/api/session/new`);
    const d = await r.json();
    return d.session_id;
  }
  const stored = window.localStorage.getItem(SESSION_STORAGE_KEY);
  if (stored && /^[a-f0-9]{1,32}$/.test(stored)) return stored;
  const r = await fetch(`${API_BASE}/api/session/new`);
  if (!r.ok) throw new Error(`Failed to mint session id: ${r.status}`);
  const d = await r.json();
  window.localStorage.setItem(SESSION_STORAGE_KEY, d.session_id);
  return d.session_id;
}

/** Force a brand-new session id (used by "clear my enrolments" → "start
 *  fresh"). Returns the new id. */
export async function rotateSessionId(): Promise<string> {
  if (typeof window !== "undefined") {
    window.localStorage.removeItem(SESSION_STORAGE_KEY);
  }
  return getOrMintSessionId();
}

export type SessionEnrolment = {
  person_id: string;
  fake_name: string;
  n_teeth: number;
  enrolled_at: number;
  panoramic_filename?: string;
  note?: string;
};

export type EnrolResponse =
  | {
      status: "enrolled";
      person: SessionEnrolment;
      open_set_score_at_enrol: number | null;
    }
  | {
      status: "duplicate_likely";
      duplicate_z_threshold: number;
      open_set_score: number;
      matched_person_id: string;
      matched_fake_name: string;
      matched_source: "session" | "canonical";
      matched_similarity: number;
      n_teeth: number;
    };

export async function postEnrol(opts: {
  sessionId: string;
  file: File;
  fakeName: string;
  note?: string;
  mode?: "segmentation" | "detection";
  force?: boolean;
}): Promise<EnrolResponse> {
  const fd = new FormData();
  fd.append("file", opts.file);
  fd.append("fake_name", opts.fakeName);
  if (opts.note) fd.append("note", opts.note);
  fd.append("mode", opts.mode ?? "segmentation");
  if (opts.force) fd.append("force", "true");
  const r = await fetch(`${API_BASE}/api/enrol`, {
    method: "POST",
    headers: { "X-Session-Id": opts.sessionId },
    body: fd,
  });
  if (!r.ok) {
    const txt = await r.text().catch(() => r.statusText);
    throw new Error(`Enrol failed: ${r.status} ${txt}`);
  }
  return r.json();
}

export type EnrolmentListResponse = {
  session_id: string;
  n_persons: number;
  persons: SessionEnrolment[];
};

export async function fetchEnrolments(sessionId: string): Promise<EnrolmentListResponse> {
  const r = await fetch(`${API_BASE}/api/enrol`, {
    headers: { "X-Session-Id": sessionId },
    cache: "no-store",
  });
  if (!r.ok) throw new Error(`Failed to list enrolments: ${r.status}`);
  return r.json();
}

export async function deleteEnrolment(sessionId: string, personId: string): Promise<void> {
  const r = await fetch(`${API_BASE}/api/enrol/${encodeURIComponent(personId)}`, {
    method: "DELETE",
    headers: { "X-Session-Id": sessionId },
  });
  if (!r.ok) throw new Error(`Delete failed: ${r.status}`);
}

export async function clearSessionEnrolments(sessionId: string): Promise<void> {
  const r = await fetch(`${API_BASE}/api/enrol`, {
    method: "DELETE",
    headers: { "X-Session-Id": sessionId },
  });
  if (!r.ok) throw new Error(`Clear failed: ${r.status}`);
}

export function panoramicDownloadUrl(personId: string): string {
  return `${API_BASE}/api/registry/${encodeURIComponent(personId)}/panoramic`;
}

export function intermediateUrl(path: string): string {
  // Backend returns paths like /api/intermediate/<id>/<file>. Locally-created
  // blob: URLs (used for the immediate upload preview) and absolute URLs are
  // returned unchanged.
  if (path.startsWith("http") || path.startsWith("blob:") || path.startsWith("data:")) {
    return path;
  }
  return `${API_BASE}${path}`;
}
