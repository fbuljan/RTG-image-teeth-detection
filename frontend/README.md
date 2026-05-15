# Frontend — Next.js demo UI

Single-page UI for the dental identification demo. Communicates with the
FastAPI backend at `http://127.0.0.1:8000` (override via
`NEXT_PUBLIC_API_BASE`) using `fetch` for REST endpoints and a manually
parsed SSE stream for `POST /api/identify` (since `EventSource` does not
support multipart uploads).

## Running

```bash
npm install         # only the first time
npm run dev         # http://localhost:3000
```

The backend must already be running on port 8000.

## Layout

The whole demo is one page with vertical sections:

1. **Header** — title and short instructions.
2. **Registry** — searchable table with `Download X-ray` per row.
3. **Upload zone** — drag-and-drop, preview, `Identify` button.
4. **Pipeline** — appears once a query starts; status text + visual overlays
   (raw → YOLO bboxes → FDI labels) swap in as each stage finishes.
5. **Results** — top-5 cards with similarity bars, confidence indicator,
   ground-truth match highlight, technical-details disclosure.

All components live in [`src/components/`](src/components/) and are tiny —
no UI framework beyond Tailwind. The SSE parser is in
[`src/lib/identify.ts`](src/lib/identify.ts).
