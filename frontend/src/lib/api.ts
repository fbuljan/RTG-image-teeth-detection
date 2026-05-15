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
};

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
