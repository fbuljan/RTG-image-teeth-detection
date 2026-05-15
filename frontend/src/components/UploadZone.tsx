"use client";

import { useEffect, useRef, useState } from "react";

type Props = {
  onIdentify: (file: File) => void;
  busy: boolean;
};

export function UploadZone({ onIdentify, busy }: Props) {
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
      <header className="border-b border-slate-200 px-6 py-4 dark:border-slate-800">
        <h2 className="text-lg font-semibold">Upload X-ray</h2>
        <p className="text-sm text-slate-500 dark:text-slate-400">
          Drag the X-ray you just downloaded here, then click <em>Identify</em>.
        </p>
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
