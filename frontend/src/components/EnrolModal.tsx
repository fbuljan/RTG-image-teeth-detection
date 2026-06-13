"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { postEnrol, type EnrolResponse } from "@/lib/api";

// Phase 9.7 — Session enrolment flow.
//
// Three-step modal: (1) pick file + name, (2) POST /api/enrol and surface
// the result (enrolled / duplicate-likely banner), (3) confirmation with a
// "verify by re-querying" button that closes the modal and re-uploads the
// same file through /api/identify (the parent owns the verify step).
//
// Calibration honesty: the spec explicitly says session enrolments do NOT
// inherit Phase 8.6 calibrated trust. The "verify by re-querying" CTA only
// promises that the FAISS index round-trips correctly (sim ≈ 1.0); it does
// not imply a calibrated identification of the enrolled person.

type Step = "pick" | "submitting" | "confirm";

const NAME_MAX = 40;

export function EnrolModal({
  sessionId,
  onClose,
  onEnrolled,
  onVerifyRequest,
}: {
  sessionId: string;
  onClose: () => void;
  onEnrolled: () => void;
  onVerifyRequest: (file: File, personId: string) => void;
}) {
  const [step, setStep] = useState<Step>("pick");
  const [file, setFile] = useState<File | null>(null);
  const [fakeName, setFakeName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [duplicate, setDuplicate] = useState<
    Extract<EnrolResponse, { status: "duplicate_likely" }> | null
  >(null);
  const [enrolled, setEnrolled] = useState<
    Extract<EnrolResponse, { status: "enrolled" }> | null
  >(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const reset = useCallback(() => {
    setStep("pick");
    setFile(null);
    setFakeName("");
    setError(null);
    setDuplicate(null);
    setEnrolled(null);
  }, []);

  // Esc to close.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  // Clear the duplicate banner whenever the user changes file or name —
  // otherwise the visible "matches Glass Sparrow" banner can describe an
  // old payload while "Enrol anyway" submits the newly-picked file under
  // the same banner. Honesty: the banner must always describe the form
  // it sits next to, or vanish.
  useEffect(() => {
    setDuplicate(null);
  }, [file, fakeName]);

  const submit = useCallback(
    async (force: boolean) => {
      if (!file || !fakeName.trim()) return;
      setStep("submitting");
      setError(null);
      try {
        const result = await postEnrol({
          sessionId,
          file,
          fakeName: fakeName.trim(),
          force,
        });
        if (result.status === "duplicate_likely") {
          setDuplicate(result);
          setStep("pick"); // banner shows under pick form
        } else {
          setEnrolled(result);
          setDuplicate(null);
          setStep("confirm");
          onEnrolled();
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        setStep("pick");
      }
    },
    [file, fakeName, sessionId, onEnrolled],
  );

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/60 px-4"
      onClick={onClose}
    >
      <div
        className="w-full max-w-lg overflow-hidden rounded-2xl bg-white shadow-2xl dark:bg-slate-900"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-slate-200 px-6 py-4 dark:border-slate-800">
          <h2 className="text-lg font-semibold">
            {step === "confirm" ? "Enrolment confirmed" : "Enrol a new person"}
          </h2>
          <button
            type="button"
            onClick={onClose}
            className="text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200"
            aria-label="Close"
          >
            ✕
          </button>
        </div>

        {(step === "pick" || step === "submitting") && (
          <div className="space-y-4 px-6 py-5 text-sm">
            <div>
              <label className="mb-1 block font-medium text-slate-700 dark:text-slate-200">
                Display name
              </label>
              <input
                type="text"
                maxLength={NAME_MAX}
                value={fakeName}
                onChange={(e) => setFakeName(e.target.value)}
                disabled={step === "submitting"}
                placeholder="e.g. Patient #42"
                className="w-full rounded-md border border-slate-300 px-3 py-2 dark:border-slate-700 dark:bg-slate-800"
              />
              <p className="mt-1 text-xs text-slate-500">
                Shown next to this person in your top-K results. Max {NAME_MAX} characters. No age,
                sex, or clinical metadata is requested or stored.
              </p>
            </div>

            <div>
              <label className="mb-1 block font-medium text-slate-700 dark:text-slate-200">
                Panoramic X-ray (PNG/JPG)
              </label>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/png,image/jpeg"
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                disabled={step === "submitting"}
                className="block w-full text-sm text-slate-500 file:mr-3 file:rounded-md file:border-0 file:bg-slate-100 file:px-3 file:py-2 file:text-sm file:font-medium file:text-slate-700 hover:file:bg-slate-200 dark:file:bg-slate-800 dark:file:text-slate-200"
              />
              {file && (
                <p className="mt-1 text-xs text-slate-500">
                  {file.name} ({(file.size / 1024).toFixed(0)} KB)
                </p>
              )}
            </div>

            {duplicate && (
              <div className="rounded-lg border border-amber-300 bg-amber-50 px-3 py-2 text-amber-900 dark:border-amber-800/60 dark:bg-amber-950/40 dark:text-amber-200">
                <p className="font-semibold">Possible duplicate</p>
                <p className="mt-1">
                  This panoramic looks like{" "}
                  <strong>{duplicate.matched_fake_name}</strong>{" "}
                  {duplicate.matched_source === "session" ? (
                    <em>(already enrolled in your session)</em>
                  ) : (
                    <em>(matches a person in the canonical registry)</em>
                  )}{" "}
                  — z-score {duplicate.open_set_score.toFixed(2)} ≫ {duplicate.duplicate_z_threshold} threshold,
                  similarity {duplicate.matched_similarity.toFixed(4)}.
                </p>
                <div className="mt-2 flex gap-2">
                  <button
                    type="button"
                    onClick={() => submit(true)}
                    className="rounded-md bg-amber-600 px-3 py-1 text-xs font-medium text-white hover:bg-amber-700"
                  >
                    Enrol anyway
                  </button>
                  <button
                    type="button"
                    onClick={() => setDuplicate(null)}
                    className="rounded-md border border-amber-400 px-3 py-1 text-xs font-medium text-amber-700 hover:bg-amber-100 dark:text-amber-200 dark:hover:bg-amber-900/40"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            )}

            {error && (
              <div className="rounded-lg border border-rose-300 bg-rose-50 px-3 py-2 text-rose-800 dark:border-rose-800/60 dark:bg-rose-950/40 dark:text-rose-200">
                {error}
              </div>
            )}

            <div className="flex items-center justify-end gap-2 border-t border-slate-200 pt-4 dark:border-slate-800">
              <button
                type="button"
                onClick={onClose}
                className="rounded-md border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:text-slate-200 dark:hover:bg-slate-800"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={() => submit(false)}
                disabled={!file || !fakeName.trim() || step === "submitting"}
                className="rounded-md bg-emerald-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-emerald-700 disabled:opacity-50"
              >
                {step === "submitting" ? "Embedding…" : "Enrol"}
              </button>
            </div>
          </div>
        )}

        {step === "confirm" && enrolled && (
          <div className="space-y-4 px-6 py-5 text-sm">
            <p>
              <strong>{enrolled.person.fake_name}</strong> added to your
              session registry (24h scope). Embedding ran on{" "}
              <strong>{enrolled.person.n_teeth}</strong> teeth.
            </p>
            <p className="text-xs text-slate-500">
              Canonical 1,178-person registry is unchanged. The
              <em> verify by re-querying</em> button below re-uploads the same
              file through the identify endpoint. A correct rank-1 self-match
              demonstrates the retrieval round-trip works — it is NOT a calibrated
              identification claim (the locked calibration is canonical-only).
            </p>
            <div className="flex items-center justify-end gap-2 border-t border-slate-200 pt-4 dark:border-slate-800">
              <button
                type="button"
                onClick={onClose}
                className="rounded-md border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:text-slate-200 dark:hover:bg-slate-800"
              >
                Done
              </button>
              <button
                type="button"
                onClick={() => {
                  if (file && enrolled) {
                    onVerifyRequest(file, enrolled.person.person_id);
                  }
                  reset();
                  onClose();
                }}
                className="rounded-md bg-sky-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-sky-700"
              >
                Verify by re-querying
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
