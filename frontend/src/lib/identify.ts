// Streaming SSE client for /api/identify.
//
// FastAPI's EventSourceResponse only supports GET when consumed via the
// browser's `EventSource` constructor. We need to POST a multipart upload, so
// we manually parse the chunked text/event-stream response from `fetch`.

import { API_BASE, type StageEvent } from "./api";

export type PipelineMode = "detection" | "segmentation";

export async function* streamIdentify(
  file: File,
  options: {
    mode?: PipelineMode;
    signal?: AbortSignal;
    sessionId?: string;
  } = {},
): AsyncGenerator<StageEvent> {
  const form = new FormData();
  form.append("file", file);
  form.append("mode", options.mode ?? "segmentation");

  // Opt-in session merge. If the caller didn't pass a session id (e.g. they
  // haven't enrolled yet), we just omit the header and the backend returns
  // the canonical-only ranking.
  const headers: Record<string, string> = {};
  if (options.sessionId) headers["X-Session-Id"] = options.sessionId;

  const res = await fetch(`${API_BASE}/api/identify`, {
    method: "POST",
    body: form,
    headers,
    signal: options.signal,
  });
  if (!res.ok || !res.body) {
    throw new Error(`Pipeline request failed: ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  // Normalize CRLF → LF up front so the buffer split only needs to look for
  // "\n\n". sse-starlette emits "\r\n\r\n" between events; the bare "\n\n"
  // check we used to do would never match and the stream would silently stall.
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");

    let idx;
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const chunk = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      const parsed = parseSse(chunk);
      if (parsed) yield parsed as StageEvent;
    }
  }
}

// Pre-cropped tooth upload. Multipart files[] with optional
// fdi_overrides_json string. Same SSE shape as /api/identify but with a
// `validate` stage in place of `detect`/`fdi` and no panoramic-image
// overlays.
export async function* streamIdentifyCrops(
  files: File[],
  options: {
    fdiOverrides?: (string | null)[];
    sessionId?: string;
    signal?: AbortSignal;
  } = {},
): AsyncGenerator<StageEvent> {
  if (files.length === 0) throw new Error("No crops supplied");

  const form = new FormData();
  for (const f of files) form.append("files", f);
  if (options.fdiOverrides) {
    form.append("fdi_overrides_json", JSON.stringify(options.fdiOverrides));
  }

  const headers: Record<string, string> = {};
  if (options.sessionId) headers["X-Session-Id"] = options.sessionId;

  const res = await fetch(`${API_BASE}/api/identify-crops`, {
    method: "POST",
    body: form,
    headers,
    signal: options.signal,
  });
  if (!res.ok || !res.body) {
    const txt = await res.text().catch(() => res.statusText);
    throw new Error(`Crops pipeline request failed: ${res.status} ${txt}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
    let idx;
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const chunk = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      const parsed = parseSse(chunk);
      if (parsed) yield parsed as StageEvent;
    }
  }
}


function parseSse(chunk: string): StageEvent | null {
  let event = "";
  const dataLines: string[] = [];
  for (const line of chunk.split("\n")) {
    if (line.startsWith("event:")) event = line.slice("event:".length).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice("data:".length).trim());
  }
  if (!event) return null;
  let data: unknown = {};
  if (dataLines.length) {
    try {
      data = JSON.parse(dataLines.join("\n"));
    } catch {
      data = { raw: dataLines.join("\n") };
    }
  }
  return { event, data } as StageEvent;
}
