"use client";

import { useEffect, useRef, useState } from "react";

import { InfoHint } from "@/components/InfoHint";

// Pre-cropped tooth upload tab — multi-file picker with per-crop FDI override.

const FDI_PATTERN = /^[1-8][1-8]$/; // basic FDI: 11..48, 51..85 etc. (loose)
const MAX_CROPS = 32;

type CropEntry = {
  file: File;
  previewUrl: string;
  fdiOverride: string;
};

export function CropsUploadZone({
  busy,
  onIdentifyCrops,
}: {
  busy: boolean;
  onIdentifyCrops: (files: File[], fdiOverrides: (string | null)[]) => void;
}) {
  const [entries, setEntries] = useState<CropEntry[]>([]);
  const inputRef = useRef<HTMLInputElement | null>(null);
  // Track every preview URL we mint so we can revoke them on unmount even if
  // a tab switch tears the component down before the user clicks Clear all.
  // Using a ref so the unmount cleanup sees the most recent set, not the
  // snapshot from first render (the previous `urls = entries.map(...)` with
  // empty deps captured `[]` and leaked every preview).
  const allUrlsRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    return () => {
      for (const u of allUrlsRef.current) URL.revokeObjectURL(u);
      allUrlsRef.current.clear();
    };
  }, []);

  function addFiles(files: FileList | null) {
    if (!files || files.length === 0) return;
    const incoming: CropEntry[] = [];
    for (const f of Array.from(files)) {
      if (!f.type.startsWith("image/")) continue;
      const url = URL.createObjectURL(f);
      allUrlsRef.current.add(url);
      incoming.push({ file: f, previewUrl: url, fdiOverride: "" });
    }
    setEntries((prev) => {
      const merged = [...prev, ...incoming];
      if (merged.length > MAX_CROPS) {
        alert(`Max ${MAX_CROPS} crops per query — truncating.`);
        for (const e of merged.slice(MAX_CROPS)) {
          URL.revokeObjectURL(e.previewUrl);
          allUrlsRef.current.delete(e.previewUrl);
        }
        return merged.slice(0, MAX_CROPS);
      }
      return merged;
    });
    if (inputRef.current) inputRef.current.value = "";
  }

  function removeAt(idx: number) {
    setEntries((prev) => {
      URL.revokeObjectURL(prev[idx].previewUrl);
      allUrlsRef.current.delete(prev[idx].previewUrl);
      return prev.filter((_, i) => i !== idx);
    });
  }

  function updateFdi(idx: number, value: string) {
    setEntries((prev) =>
      prev.map((e, i) => (i === idx ? { ...e, fdiOverride: value } : e)),
    );
  }

  function submit() {
    if (entries.length === 0 || busy) return;
    // Validate override syntax client-side BEFORE shipping a request that
    // would 400 on the server anyway. Empty override = auto-detect.
    const overrides: (string | null)[] = [];
    for (const [i, e] of entries.entries()) {
      const v = e.fdiOverride.trim();
      if (v === "") {
        overrides.push(null);
      } else if (!FDI_PATTERN.test(v)) {
        alert(`Crop ${i + 1}: "${v}" is not a valid FDI label (e.g. 11, 23, 47). Leave blank for auto-detect.`);
        return;
      } else {
        overrides.push(v);
      }
    }
    onIdentifyCrops(entries.map((e) => e.file), overrides);
  }

  function clearAll() {
    for (const e of entries) {
      URL.revokeObjectURL(e.previewUrl);
      allUrlsRef.current.delete(e.previewUrl);
    }
    setEntries([]);
  }

  return (
    <section className="rounded-2xl border border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-slate-900">
      <header className="flex flex-col gap-2 border-b border-slate-200 px-6 py-4 dark:border-slate-800">
        <div className="flex items-center gap-2">
          <h2 className="text-lg font-semibold">Upload tooth crops</h2>
          <InfoHint
            text={
              "1-32 pre-cropped single-tooth images. Each is classified, embedded, mean-pooled, and searched against the registry. Override the FDI label per crop if auto-detection misfires."
            }
          />
        </div>
        <p className="text-sm text-slate-500 dark:text-slate-400">
          Drop in individual tooth crops when you don&apos;t have a full panoramic.
          Expected R1: N=4 ≈ 21%, N=8 ≈ 45%, N=16 ≈ 83%.
        </p>
      </header>

      <div className="space-y-4 px-6 py-4">
        <div className="flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            disabled={busy || entries.length >= MAX_CROPS}
            className="rounded-md border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50 dark:border-slate-700 dark:text-slate-200 dark:hover:bg-slate-800"
          >
            {entries.length === 0 ? "Choose crops…" : "Add more crops…"}
          </button>
          {entries.length > 0 && (
            <button
              type="button"
              onClick={clearAll}
              disabled={busy}
              className="rounded-md border border-rose-300 px-3 py-1.5 text-xs font-medium text-rose-700 hover:bg-rose-50 disabled:opacity-50 dark:border-rose-800/60 dark:text-rose-300 dark:hover:bg-rose-950/30"
            >
              Clear all
            </button>
          )}
          <span className="text-xs text-slate-500 dark:text-slate-400">
            {entries.length} / {MAX_CROPS} selected
          </span>
          <input
            ref={inputRef}
            type="file"
            accept="image/png,image/jpeg"
            multiple
            className="hidden"
            onChange={(e) => addFiles(e.target.files)}
          />
        </div>

        {entries.length === 0 ? (
          <div className="rounded-xl border-2 border-dashed border-slate-300 bg-slate-50 px-6 py-10 text-center text-sm text-slate-500 dark:border-slate-700 dark:bg-slate-950">
            <p className="font-medium">No crops yet</p>
            <p className="mt-1 text-xs">Pick 1-32 PNG/JPG tooth crops to identify.</p>
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4">
            {entries.map((e, i) => (
              <div
                key={`${e.file.name}-${i}`}
                className="flex flex-col gap-2 rounded-lg border border-slate-200 bg-white p-2 dark:border-slate-700 dark:bg-slate-900"
              >
                <div className="flex h-24 items-center justify-center overflow-hidden rounded bg-slate-100 dark:bg-slate-800">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={e.previewUrl}
                    alt={`crop ${i + 1}`}
                    className="max-h-24 w-auto object-contain"
                  />
                </div>
                <p className="truncate text-[11px] text-slate-500" title={e.file.name}>
                  {e.file.name}
                </p>
                <div className="flex items-center gap-1">
                  <input
                    type="text"
                    value={e.fdiOverride}
                    onChange={(ev) => updateFdi(i, ev.target.value)}
                    placeholder="Auto"
                    maxLength={2}
                    disabled={busy}
                    className="w-full rounded border border-slate-300 px-2 py-0.5 text-xs dark:border-slate-700 dark:bg-slate-800"
                    aria-label={`FDI override for crop ${i + 1}`}
                  />
                  <button
                    type="button"
                    onClick={() => removeAt(i)}
                    disabled={busy}
                    aria-label={`Remove crop ${i + 1}`}
                    className="rounded border border-slate-300 px-1.5 py-0.5 text-xs text-slate-500 hover:bg-slate-50 disabled:opacity-50 dark:border-slate-700 dark:hover:bg-slate-800"
                  >
                    ✕
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="flex items-center justify-between gap-3 border-t border-slate-200 px-6 py-4 dark:border-slate-800">
        <div className="text-xs text-slate-500 dark:text-slate-400">
          {entries.length === 0
            ? "Pick at least one tooth crop."
            : `Ready to identify ${entries.length} crop${entries.length === 1 ? "" : "s"}.`}
        </div>
        <button
          type="button"
          disabled={entries.length === 0 || busy}
          onClick={submit}
          className="rounded-lg bg-amber-500 px-5 py-2 text-sm font-semibold text-white shadow disabled:cursor-not-allowed disabled:opacity-60 hover:bg-amber-600"
        >
          {busy ? "Running…" : "Identify from crops"}
        </button>
      </div>
    </section>
  );
}
