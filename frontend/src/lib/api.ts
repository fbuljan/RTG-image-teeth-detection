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
  similarity_percentile?: number | null;
};

// Phase 9.2 — Phase 8.6 locked open-set decision.
export type OpenSetDecision = "in_registry" | "rejected" | "unknown";

// Phase 9.2 — provenance of the uploaded query.
// "self_match"  — exact bytes of an enrolled panoramic (tautological match).
// "novel"       — bytes do not match any enrolled image.
// "heldout"     — reserved for Phase 9.8 curated OOS picks.
// "unknown"     — could not classify (filesystem error etc.).
export type QueryProvenance = "self_match" | "novel" | "heldout" | "unknown";

export type StageEvent =
  | { event: "stage_start"; data: { stage: string; message: string; total?: number } }
  | { event: "stage_complete"; data: StageCompleteData }
  | { event: "progress"; data: { stage: string; current: number; total: number } }
  | { event: "warning"; data: { code: string; message: string } }
  | { event: "error"; data: { message: string } }
  | { event: "done"; data: Record<string, unknown> };

export type ToothContribution = {
  fdi: string;
  fdi_confidence: number;
  similarity_to_top1: number;
};

export type StageCompleteData = {
  stage: string;
  n_teeth?: number;
  n_uncertain?: number;
  n_dropped?: number;
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
};

export async function fetchRegistry(): Promise<RegistryListResponse> {
  const res = await fetch(`${API_BASE}/api/registry`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch registry: ${res.status}`);
  return res.json();
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
