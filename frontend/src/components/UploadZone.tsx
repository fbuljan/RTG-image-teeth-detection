"use client";

import { useEffect, useRef, useState } from "react";

import { InfoHint } from "@/components/InfoHint";
import type { PipelineMode } from "@/lib/identify";

type Props = {
  onIdentify: (file: File) => void;
  busy: boolean;
  mode: PipelineMode;
  onModeChange: (mode: PipelineMode) => void;
};

const MODE_HINT =
  "How teeth are cut out before identification. Segmentation traces each tooth's outline; detection uses a bounding box. Segmentation crops are closer to what the embedder was trained on.";

export function UploadZone({
  onIdentify,
  busy,
  mode,
  onModeChange,
}: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [hovering, setHovering] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  function handleFiles(files: FileList | null) {
    if (!files || files.length === 0) return;
    const f = files[0];
    if (!f.type.startsWith("image/")) {
      alert("Please drop a PNG or JPG image of a panoramic X-ray.");
      return;
    }
    setFile(f);
  }

  return (
    <section className="rounded-2xl border border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-slate-900">
      <header className="flex flex-col gap-3 border-b border-slate-200 px-6 py-4 dark:border-slate-800 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-lg font-semibold">Upload X-ray</h2>
          <p className="text-sm text-slate-500 dark:text-slate-400">
            Drag the X-ray you just downloaded here, then click <em>Identify</em>.
          </p>
        </div>
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:gap-4">
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
              Cropping
            </span>
            <InfoHint text={MODE_HINT} />
            <div
              role="radiogroup"
              aria-label="Cropping backend"
              className="inline-flex rounded-full border border-slate-200 bg-slate-100 p-0.5 text-xs dark:border-slate-700 dark:bg-slate-800"
            >
              {(["segmentation", "detection"] as PipelineMode[]).map((m) => {
                const active = mode === m;
                return (
                  <button
                    key={m}
                    type="button"
                    role="radio"
                    aria-checked={active}
                    disabled={busy}
                    onClick={() => onModeChange(m)}
                    className={`rounded-full px-3 py-1 font-medium transition disabled:cursor-not-allowed disabled:opacity-50 ${
                      active
                        ? "bg-white text-slate-900 shadow dark:bg-slate-950 dark:text-slate-100"
                        : "text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-100"
                    }`}
                  >
                    {m === "segmentation" ? "Segmentation" : "Detection"}
                  </button>
                );
              })}
            </div>
          </div>

        </div>
      </header>

      <div
        onDragEnter={(e) => {
          e.preventDefault();
          setHovering(true);
        }}
        onDragOver={(e) => {
          e.preventDefault();
          setHovering(true);
        }}
        onDragLeave={() => setHovering(false)}
        onDrop={(e) => {
          e.preventDefault();
          setHovering(false);
          handleFiles(e.dataTransfer.files);
        }}
        className={`m-6 flex min-h-[260px] flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-10 text-center transition ${
          hovering
            ? "border-amber-400 bg-amber-50 dark:bg-amber-900/30"
            : "border-slate-300 bg-slate-50 dark:border-slate-700 dark:bg-slate-950"
        }`}
      >
        {previewUrl ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={previewUrl}
            alt="Uploaded X-ray preview"
            className="max-h-72 w-auto rounded-lg shadow"
          />
        ) : (
          <>
            <p className="text-sm font-medium">Drop xray.png here</p>
            <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
              or click to browse your files
            </p>
          </>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          className="mt-4 rounded-lg border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-white dark:border-slate-700 dark:text-slate-200 dark:hover:bg-slate-900"
        >
          {file ? "Choose a different file" : "Choose file"}
        </button>
      </div>

      <div className="flex items-center justify-between gap-3 border-t border-slate-200 px-6 py-4 dark:border-slate-800">
        <div className="text-sm text-slate-500 dark:text-slate-400">
          {file ? file.name : "No file selected."}
        </div>
        <button
          type="button"
          disabled={!file || busy}
          onClick={() => file && onIdentify(file)}
          className="rounded-lg bg-amber-500 px-5 py-2 text-sm font-semibold text-white shadow disabled:cursor-not-allowed disabled:opacity-60 hover:bg-amber-600"
        >
          {busy ? "Running…" : "Identify"}
        </button>
      </div>
    </section>
  );
}
