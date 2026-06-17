"use client";

import { useState } from "react";

// About-this-demo modal — registry-composition + headline numbers.

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
                All <strong>1,178 persons</strong> from the academic dataset are
                enrolled. The example panoramics on this page are themselves in the
                registry, so re-uploading one is a self-match by construction
                (similarity ≈ 1.0). The session-enrolment flow lets you add your own
                person to a separate 24h browser-scoped index.
              </p>
            </section>

            <section className="mt-5 space-y-3 text-sm text-slate-700 dark:text-slate-200">
              <h3 className="text-base font-semibold">See real retrieval</h3>
              <p>
                To see the system actually discriminate (not just confirm a byte-identical
                match):
              </p>
              <ul className="ml-5 list-disc space-y-1">
                <li>
                  Use the <strong>Fragment-size chips</strong> on any result. At N=4 or
                  N=8 the query is no longer the full enrolled mean, so the search has
                  real work to do.
                </li>
                <li>
                  Upload a panoramic of someone not in the example list — the system
                  returns its nearest neighbours.
                </li>
              </ul>
            </section>

            <section className="mt-5 space-y-2 text-sm text-slate-700 dark:text-slate-200">
              <h3 className="text-base font-semibold">Headline numbers</h3>
              <ul className="ml-5 list-disc space-y-0.5">
                <li><strong>R1 = 82.6%</strong> on the 1,178-person registry (full panoramic, n=178 held-out test set).</li>
                <li>R5 = 97.3% · R10 = 99.2%.</li>
                <li>Fragment R1: N=4 ≈ 21% · N=8 ≈ 45% · N=16 ≈ 83%.</li>
                <li><strong>Open-set AUROC = 0.832</strong> on clean queries / <strong>0.609</strong> on rotated queries.</li>
                <li><strong>Age MAE = 0.93y</strong> on the 6-13y range.</li>
                <li>Person-level sex prediction failed at chance and is not shown.</li>
              </ul>
            </section>

            <section className="mt-5 space-y-2 text-sm text-slate-700 dark:text-slate-200">
              <h3 className="text-base font-semibold">Limitations</h3>
              <ul className="ml-5 list-disc space-y-0.5">
                <li>
                  Training set is pediatric / adolescent (6-18y); behaviour on
                  adult dentition with restorations, prostheses, or edentulous
                  regions is untested.
                </li>
                <li>
                  One panoramic per person — cross-visit re-identification (a
                  different X-ray of the same person) was never measured.
                </li>
                <li>
                  In-registry false-rejection rate at the locked operating point
                  is 21.4% on clean queries, 62.8% on ±30° rotated queries.
                </li>
                <li>
                  R1 drops from 82.6% upright to 43.1% at ±30° rotation; the
                  system is not rotation-invariant.
                </li>
                <li>
                  Mixed-dentition retrieval is brittle: R1 = 68.8% on the 6-9y
                  subset vs 91.4% on the all-permanent subset. The latter is an
                  upper bound, not a deployment-ready number.
                </li>
              </ul>
            </section>
          </div>
        </div>
      )}
    </>
  );
}
