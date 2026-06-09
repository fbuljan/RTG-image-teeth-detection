"use client";

import { useState } from "react";

// Phase 9.8 — About-this-demo modal.
//
// Surfaces the registry-composition caveat, the academic vs demo metric
// distinction, and pointers to genuine OOS retrieval flows. The audit
// (workflow wl51scjss, 2026-06-08) identified UI honesty as the single
// load-bearing demo issue; this modal is the explicit fix.

export function AboutDemo() {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="rounded-lg border border-slate-300 px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:text-slate-200 dark:hover:bg-slate-800"
      >
        About this demo
      </button>
      {open && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/60 px-4"
          onClick={() => setOpen(false)}
        >
          <div
            className="max-h-[90vh] max-w-2xl overflow-y-auto rounded-2xl bg-white p-6 shadow-2xl dark:bg-slate-900"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="mb-4 flex items-start justify-between">
              <h2 className="text-xl font-semibold">About this demo</h2>
              <button
                type="button"
                onClick={() => setOpen(false)}
                className="text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200"
                aria-label="Close"
              >
                ✕
              </button>
            </div>

            <section className="space-y-3 text-sm text-slate-700 dark:text-slate-200">
              <h3 className="text-base font-semibold">What&apos;s in the registry</h3>
              <p>
                The registry contains <strong>all 1,178 persons</strong> from the academic dataset
                — the train, val, and test splits combined. The example panoramics on
                this page are pulled directly from that registry, so re-uploading one
                is a <strong>self-match</strong> by construction (similarity ≈ 1.0). The
                provenance pill on each result tells you whether you&apos;re in self-match,
                novel-upload, or curated-OOS mode.
              </p>
              <p>
                A real forensic deployment would scope the registry to enrolled cases
                only (the Phase 9.7 session-enrolment flow demonstrates this — your
                additions live in a 24h session-scoped index, separate from the canonical
                read-only one).
              </p>
            </section>

            <section className="mt-6 space-y-3 text-sm text-slate-700 dark:text-slate-200">
              <h3 className="text-base font-semibold">What the academic numbers mean</h3>
              <ul className="ml-5 list-disc space-y-1">
                <li>
                  <strong>R1 = 82.6% [79.8, 86.0]</strong> closed-set retrieval, n=178 held-out test
                  PIDs, full-panoramic query against the 1,178-person registry, mean-pooled
                  YOLO-cropped teeth (Phase 8.0 re-baseline 2026-06-08).
                </li>
                <li>
                  <strong>Open-set AUROC = 0.832 [0.796, 0.870]</strong> clean
                  / <strong>0.609 [0.552, 0.665]</strong> rotated, locked threshold from
                  Phase 8.6. The Verdict + calibration strip you see on every result is
                  driven by this calibration.
                </li>
                <li>
                  <strong>Single-tooth R1 ≈ 3.6%</strong> against the 1,178-person registry —
                  forty-two times chance (1/1178 ≈ 0.085%). The 23× lift from 3.6% to 82.6%
                  is the empirical signature of mean-pool aggregation working as intended.
                </li>
                <li>
                  <strong>Age MAE = 0.93y</strong> on the 6-13y dense bucket (Phase 8.10
                  reported number on GT-mean embeddings; the live demo uses YOLO-mean
                  embeddings, so the real-world error is wider — see the age chip tooltip).
                </li>
                <li>
                  <strong>Sex is NOT shown.</strong> The Phase 8.10 sex head failed at chance
                  (0.556 acc, CI overlaps chance baseline 0.539). Wiring it would mislead
                  users; we honour the pre-registered Pass/Fail rule.
                </li>
              </ul>
            </section>

            <section className="mt-6 space-y-3 text-sm text-slate-700 dark:text-slate-200">
              <h3 className="text-base font-semibold">Try a genuine retrieval</h3>
              <p>
                The dataset has one panoramic per person, so there&apos;s no built-in
                &quot;different image of the same person&quot; flow. To see the system actually
                discriminate (not just confirm a byte-identical match):
              </p>
              <ul className="ml-5 list-disc space-y-1">
                <li>
                  <strong>Use the Fragment-size chips</strong> on any result: at N=4 or N=8,
                  the query is no longer the full enrolled mean, so the calibration honestly
                  separates correct from wrong identifications (Phase 5 priors: N=4 → R1 ≈ 21%,
                  N=8 → R1 ≈ 45%).
                </li>
                <li>
                  <strong>Upload a panoramic of someone not in the example list.</strong> The
                  system will return its nearest neighbor among 1,178 people — but the verdict
                  should flag the result as &quot;probably not enrolled&quot; if calibration
                  is doing its job.
                </li>
                <li>
                  <strong>Rotate or crop the X-ray before uploading.</strong> The bytes change
                  → provenance flips to novel → you see the rotated-stress regime (Phase 8.6
                  rotated AUROC 0.609; some queries will be over-confidently kept, others
                  correctly rejected).
                </li>
              </ul>
            </section>

            <section className="mt-6 space-y-3 text-xs text-slate-500 dark:text-slate-400">
              <h3 className="text-xs font-semibold uppercase tracking-wide">
                Residual limitations
              </h3>
              <ul className="ml-5 list-disc space-y-1">
                <li>
                  21.4% in-registry false-rejection rate at the locked operating point
                  (Phase 8.6 design choice; trades off against 70% OOS true-rejection).
                </li>
                <li>
                  Rotated AUROC 0.609 means ~40% of correctly-identified rotated queries
                  will be flagged as &quot;probably not enrolled&quot; — a safe-failure mode
                  but a calibration limitation.
                </li>
                <li>
                  Age regression saturates at 16-18y (regression-ceiling effect; dental
                  development is largely complete by 17).
                </li>
                <li>
                  Partial-fragment calibration is not re-derived: the Phase 8.6 thresholds
                  were learned on full-panoramic queries, so at N&lt;8 the verdict reflects
                  the closest in-distribution comparison the calibration can offer, not a
                  fresh fragment-specific decision.
                </li>
                <li>
                  Session enrolments (Phase 9.7) are non-durable: they expire after 24h.
                </li>
              </ul>
            </section>
          </div>
        </div>
      )}
    </>
  );
}
