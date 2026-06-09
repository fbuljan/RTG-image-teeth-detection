"use client";

import { useState } from "react";
import { InfoHint } from "@/components/InfoHint";
import type {
  AgeEstimate,
  OpenSetDecision,
  PerTooth,
  QueryProvenance,
  SearchResult,
  ToothContribution,
} from "@/lib/api";
import { searchFragment } from "@/lib/api";

export type ResultsState = {
  results: SearchResult[];
  confidence: "high" | "medium" | "uncertain" | "low";
  topGap: number;
  timings: Record<string, number>;
  nQueryTeeth: number;
  nDropped: number;
  toothContributions?: ToothContribution[];
  selectedPersonId?: string;
  selectedFakeName?: string;
  // Phase 9.3 — Phase 8.6 calibrated open-set + provenance.
  openSetScore: number | null;
  openSetDecision: OpenSetDecision;
  openSetThreshold: number | null;
  queryProvenance: QueryProvenance;
  expectedPersonId: string | null;
  simTop1Percentile: number | null;
  // Phase 9.4 — Phase 8.10 age estimate (sex NOT wired; failed Pass).
  ageEstimate: AgeEstimate | null;
  // Phase 9.5 — fragment-search support.
  queryId: string | null;
  perTooth: PerTooth[];
};

type Props = {
  state: ResultsState;
  onReset: () => void;
  onFragmentResult?: (result: import("@/lib/api").FragmentSearchResponse) => void;
};

// Phase 5 priors (pre-registered): R1 vs gallery as a function of n_query.
// Used by the FragmentSelector to display "expected outcome at this N" so a
// rank-1 miss reads as a confirmed data point not a credibility loss.
const FRAGMENT_PRIORS: Record<number, { r1: number; r5: number }> = {
  1: { r1: 0.029, r5: 0.083 },
  2: { r1: 0.088, r5: 0.183 },
  4: { r1: 0.209, r5: 0.346 },
  8: { r1: 0.446, r5: 0.617 },
  16: { r1: 0.826, r5: 0.898 },
};

function deterministicSample<T>(items: T[], n: number, seed: number): number[] {
  // Mulberry32 PRNG so shuffles are reproducible per (seed, n).
  let s = seed >>> 0;
  const rand = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const idx = items.map((_, i) => i);
  for (let i = idx.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  return idx.slice(0, n);
}

function FragmentSelector({
  state,
  onFragmentResult,
}: {
  state: ResultsState;
  onFragmentResult?: (result: import("@/lib/api").FragmentSearchResponse) => void;
}) {
  const [shuffleSeed, setShuffleSeed] = useState(1);
  const [activeN, setActiveN] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const total = state.perTooth.length;
  if (total === 0 || !state.queryId) return null;

  const sizes = [1, 2, 4, 8, 16, total].filter((n, i, arr) => n <= total && arr.indexOf(n) === i);

  async function runAt(n: number, seedOverride?: number) {
    if (!state.queryId) return;
    setBusy(true);
    setActiveN(n);
    try {
      // Use seedOverride when supplied (the Shuffle button bumps the seed and
      // re-runs in the same handler — without an explicit override, the
      // setShuffleSeed update isn't visible to this closure until the next
      // render, so the first Shuffle click would re-sample with the OLD seed).
      const indices = deterministicSample(state.perTooth, n, seedOverride ?? shuffleSeed);
      const result = await searchFragment(state.queryId, indices);
      onFragmentResult?.(result);
    } catch (e) {
      console.error("fragment search failed", e);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 dark:border-slate-700 dark:bg-slate-800/40">
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <span className="font-semibold text-slate-700 dark:text-slate-200">Fragment size:</span>
        {sizes.map((n) => {
          const prior = FRAGMENT_PRIORS[n];
          const isAmber = n <= 4;
          const isActive = activeN === n;
          return (
            <button
              key={n}
              type="button"
              disabled={busy}
              onClick={() => runAt(n)}
              className={`rounded-md px-2.5 py-1 text-xs font-medium ring-1 ring-inset transition disabled:opacity-50 ${
                isActive
                  ? "bg-sky-500 text-white ring-sky-500"
                  : isAmber
                  ? "bg-amber-500/10 text-amber-700 ring-amber-500/30 hover:bg-amber-500/20 dark:text-amber-300"
                  : "bg-emerald-500/10 text-emerald-700 ring-emerald-500/30 hover:bg-emerald-500/20 dark:text-emerald-300"
              }`}
              title={
                prior
                  ? `Phase 5 prior at N=${n}: R1 = ${(prior.r1 * 100).toFixed(0)}%, R5 = ${(prior.r5 * 100).toFixed(0)}%`
                  : `Use all ${n} detected teeth`
              }
            >
              {n === total ? `All (${n})` : n}
            </button>
          );
        })}
        <button
          type="button"
          disabled={busy || activeN === null}
          onClick={() => {
            const nextSeed = shuffleSeed + 1;
            setShuffleSeed(nextSeed);
            if (activeN !== null) runAt(activeN, nextSeed);
          }}
          className="ml-2 rounded-md border border-slate-300 px-2.5 py-1 text-xs font-medium text-slate-700 hover:bg-white disabled:opacity-50 dark:border-slate-600 dark:text-slate-200 dark:hover:bg-slate-700"
        >
          Shuffle
        </button>
        <InfoHint
          text={
            "Re-run the search with a random subset of N teeth from this query. Phase 5 priors (full-registry R1): N=1 ≈ 3%, N=2 ≈ 9%, N=4 ≈ 21%, N=8 ≈ 45%, N=16 ≈ 83%. A wrong rank-1 at small N is expected, not a failure — it shows the system's honest operating regime when only a fragment is available."
          }
        />
      </div>
    </div>
  );
}

// ---------- Phase 9.3 — Verdict (3-state, replaces 4-tier confidence) ----------

type VerdictTone = {
  label: string;
  badge: string;     // colored badge classes
  note: string;
};

const VERDICT_COPY: Record<
  "likely_enrolled" | "borderline" | "rejected" | "calibration_unavailable",
  VerdictTone
> = {
  likely_enrolled: {
    label: "Likely enrolled",
    badge: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-300",
    note:
      "The calibrated open-set score is above the Phase 8.6 threshold and the top-1 match leads the runners-up by a clear margin. System believes the query person is in the registry.",
  },
  borderline: {
    label: "Borderline",
    badge: "bg-amber-500/20 text-amber-700 dark:text-amber-300",
    note:
      "The calibrated open-set score is above threshold, but the gap to the runner-up is narrow. The top-1 identity is plausible but not strongly supported.",
  },
  rejected: {
    label: "Probably not enrolled",
    badge: "bg-rose-500/20 text-rose-700 dark:text-rose-300",
    note:
      "The calibrated open-set score is below the Phase 8.6 threshold. System believes the query person is NOT in the registry. The candidates listed below are the nearest neighbors, not predictions.",
  },
  calibration_unavailable: {
    label: "Calibration unavailable",
    badge: "bg-slate-300 text-slate-800 dark:bg-slate-700 dark:text-slate-100",
    note:
      "Open-set calibration JSON could not be loaded, so the system cannot decide whether the query person is enrolled. Top-K below is shown as nearest neighbors only; treat raw similarities as uncalibrated and do not interpret them as identification evidence.",
  },
};

// Phase 8.6 lower tercile on top1-top2 gap — heuristic, NOT derived from the
// locked calibration JSON (which has no tercile field). If you regenerate the
// calibration set, re-derive this and lift it into the JSON.
const BORDERLINE_GAP_TERCILE = 0.001;

function classifyVerdict(state: ResultsState): keyof typeof VERDICT_COPY {
  // Audit fail-open fix: ONLY in_registry queries can render likely_enrolled /
  // borderline. The previous version returned "likely_enrolled" for any
  // decision that wasn't "rejected" — including "unknown" (calibration JSON
  // missing), which rendered a green badge with copy claiming "The calibrated
  // open-set score is above the Phase 8.6 threshold" when there was no
  // calibrated score at all.
  if (state.openSetDecision === "rejected") return "rejected";
  if (state.openSetDecision !== "in_registry") return "calibration_unavailable";
  if (state.topGap < BORDERLINE_GAP_TERCILE) return "borderline";
  return "likely_enrolled";
}

// ---------- Phase 9.3 — Provenance pill ----------

const PROVENANCE_COPY: Record<QueryProvenance, { label: string; chip: string; banner: string | null }> = {
  self_match: {
    label: "Self-match demo",
    chip: "bg-amber-500/15 text-amber-700 ring-amber-500/30 dark:text-amber-300",
    banner:
      "This is the same X-ray that built the enrolled registry entry. A correct rank-1 here demonstrates that the pipeline runs end-to-end on a known image — it is NOT a measurement of identification on a new photo.",
  },
  novel: {
    label: "Novel upload",
    chip: "bg-slate-200 text-slate-700 ring-slate-300 dark:bg-slate-800 dark:text-slate-200 dark:ring-slate-600",
    banner: null,
  },
  heldout: {
    label: "Held-out · OOS",
    chip: "bg-purple-500/15 text-purple-700 ring-purple-500/30 dark:text-purple-300",
    banner:
      "This is a curated out-of-distribution test image. The system should ideally reject it as 'probably not enrolled.' Rotated AUROC was 0.609 in Phase 8.6, so some of these will slip past the rejection threshold.",
  },
  unknown: {
    label: "Provenance unknown",
    chip: "bg-slate-100 text-slate-600 ring-slate-300 dark:bg-slate-800 dark:text-slate-300",
    banner: null,
  },
};

// ---------- Phase 9.3 — Calibration strip ----------
// Renders the query's open-set z-score relative to the locked Phase 8.6
// threshold so a viewer can see at a glance how confidently the system
// classified them as in_registry vs rejected.

function CalibrationStrip({
  score,
  threshold,
  decision,
}: {
  score: number | null;
  threshold: number | null;
  decision: OpenSetDecision;
}) {
  if (score === null || threshold === null) return null;
  // Visualize in z-score space. Most in-registry queries land around z = 1–2;
  // most OOS queries land around z = −2 to −5; the threshold is z ≈ −0.68.
  // Clamp display to [−5, +3] so extreme rotated panoramics (z = −18) don't
  // squish the strip.
  const lo = -5;
  const hi = 3;
  const pct = (x: number) => 100 * (Math.min(hi, Math.max(lo, x)) - lo) / (hi - lo);
  const scorePct = pct(score);
  const thrPct = pct(threshold);
  const isAccepted = decision === "in_registry";
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400">
        <span>Probably not enrolled</span>
        <span>Likely enrolled</span>
      </div>
      <div className="relative h-3 rounded-full bg-gradient-to-r from-rose-300/50 via-amber-300/50 to-emerald-300/50">
        {/* Threshold marker (vertical line) */}
        <div
          className="absolute top-[-3px] bottom-[-3px] w-[2px] bg-slate-700 dark:bg-slate-200"
          style={{ left: `calc(${thrPct}% - 1px)` }}
          title={`Locked threshold (z = ${threshold.toFixed(3)})`}
        />
        {/* Score marker */}
        <div
          className={`absolute top-[-5px] bottom-[-5px] w-[6px] rounded-full border-2 border-white dark:border-slate-900 ${
            isAccepted ? "bg-emerald-600" : "bg-rose-600"
          }`}
          style={{ left: `calc(${scorePct}% - 3px)` }}
          title={`Your z-score: ${score.toFixed(3)}`}
        />
      </div>
      <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400">
        <span>z = {lo}</span>
        <span className="font-mono">
          your z = <strong>{score.toFixed(3)}</strong>
          {" · "}
          threshold = {threshold.toFixed(3)}
        </span>
        <span>z = +{hi}</span>
      </div>
    </div>
  );
}

// ---------- Tooltips ----------

const PERCENTILE_HINT =
  "Only defined for rank #1 — the reference distribution is held-out *rank-1* hits (740 values from Phase 8.6 enrolment). 73% means 'this top-1 similarity is higher than 73% of correct identifications observed during evaluation.' Ranks 2-5 show '—' because they're runners-up, not correct identifications, so applying the same percentile would be a category error.";

const VERDICT_HINT =
  "Calibrated open-set verdict, using the locked Phase 8.6 threshold (z = −0.680 on z-scored sim_top1). Likely enrolled = passes threshold with margin. Borderline = passes threshold but top-1 vs top-2 gap is in the lower tercile. Probably not enrolled = below threshold; the system thinks the query person is NOT in the registry.";

const PROVENANCE_HINT =
  "Self-match: the uploaded image is byte-identical to an enrolled image (sim ≈ 1.0 is tautological). Novel upload: the bytes don't match any enrolled image. Held-out · OOS: a curated out-of-distribution test image (Phase 9.8). The dataset has one panoramic per person, so 'self-match' and 'enrolled' are the same set here.";

const AGE_HINT_CONFIDENT =
  "Estimated dental age from the Phase 8.10 regression head on the frozen embedder. Pre-registered MAE = 0.93y on the 6-13y dense bucket on GT-mean embeddings; the live demo uses YOLO-mean embeddings (GT→YOLO distribution shift, ~1.8y empirical MAE in smoke testing). CI ±2.5y is conservative — covers the worst observed per-bucket MAE (2.09y in 16-18y) with a buffer. Suppressed when the open-set verdict is 'probably not enrolled' (embedder is out of distribution → head output meaningless). Sex is intentionally NOT shown — the sex head failed at chance (0.556 acc).";

const AGE_HINT_SATURATED =
  "The prediction is outside the dense 6-13y training bucket or hit the training-range boundary [6, 18]. Outside dense the head is less reliable: 16-18y reported MAE = 2.09y on GT-mean embeddings (regression-ceiling effect — dental development is largely complete by 17). CI widened to ±3.5y. Note: the dense-bucket flag is a *raw-prediction* heuristic, not ground-truth — an adult whose embedding saturates the head to e.g. 10.5y will NOT be marked saturated, so treat any in-bucket estimate as 'best-case under the GT→YOLO shift,' not a guarantee.";

const AGE_HINT_SMALL_POOL =
  "Pool size < 8 teeth. The age head was trained on per-person mean embeddings (16-tooth pools); 1-4 tooth fragments are an unvalidated pool-size shift on top of the GT→YOLO shift, so the CI is widened to ±4y as a conservative indicative spread (no per-pool-size MAE was measured in Phase 8.10).";

// ---------- ResultsCards ----------

export function ResultsCards({ state, onReset, onFragmentResult }: Props) {
  const verdictKey = classifyVerdict(state);
  const verdict = VERDICT_COPY[verdictKey];
  const provenance = PROVENANCE_COPY[state.queryProvenance];

  const isRejected = state.openSetDecision === "rejected";
  const isCalibrationUnavailable = verdictKey === "calibration_unavailable";
  const isUncalibrated = isRejected || isCalibrationUnavailable;
  const topSim = state.results[0]?.similarity ?? 1;
  const bottomSim =
    state.results.length > 1
      ? state.results[state.results.length - 1].similarity
      : topSim - 0.01;
  const range = Math.max(1e-6, topSim - bottomSim);

  // For OOS-rejected results we desaturate the list to communicate "these
  // are nearest neighbors, not predictions."
  const listOpacityClass = isUncalibrated ? "opacity-60" : "";

  return (
    <section className="rounded-2xl border border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-slate-900">
      <header className="flex flex-col gap-3 border-b border-slate-200 px-6 py-4 dark:border-slate-800 sm:flex-row sm:items-start sm:justify-between">
        <div className="space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="text-lg font-semibold">Results</h2>
            <span
              className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ring-inset ${provenance.chip}`}
            >
              {provenance.label}
              <InfoHint text={PROVENANCE_HINT} />
            </span>
            {state.ageEstimate && (() => {
              const a = state.ageEstimate;
              const isRisky = a.saturation_risk || a.small_pool;
              const hintText = a.small_pool
                ? AGE_HINT_SMALL_POOL
                : a.saturation_risk
                  ? AGE_HINT_SATURATED
                  : AGE_HINT_CONFIDENT;
              const caveat = a.small_pool
                ? `(pool n=${a.pool_size})`
                : a.saturation_risk
                  ? "(saturation risk)"
                  : null;
              return (
                <span
                  className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ring-inset ${
                    isRisky
                      ? "bg-slate-200 text-slate-600 ring-slate-300 dark:bg-slate-800 dark:text-slate-300 dark:ring-slate-600"
                      : "bg-sky-500/15 text-sky-700 ring-sky-500/30 dark:text-sky-300"
                  }`}
                >
                  Estimated age: {a.value_display.toFixed(1)}
                  {" ± "}
                  {a.ci_half.toFixed(1)}y
                  {caveat && <span className="ml-1 text-[10px] opacity-70">{caveat}</span>}
                  <InfoHint text={hintText} />
                </span>
              );
            })()}
          </div>
          <p className="text-sm text-slate-500 dark:text-slate-400">
            {`Queried with ${state.nQueryTeeth} teeth`}
            {state.nDropped > 0 ? ` · ${state.nDropped} duplicates dropped` : ""}
          </p>
        </div>
        <span
          className={`inline-flex flex-shrink-0 items-center rounded-full px-3 py-1 text-xs font-semibold ${verdict.badge}`}
        >
          {verdict.label}
          <InfoHint text={VERDICT_HINT} />
        </span>
      </header>

      <div className="space-y-4 px-6 py-5">
        {/* Provenance disclaimer banner — only when self-match, only on demand */}
        {provenance.banner && (
          <div className={`rounded-lg border px-4 py-3 text-sm ${
            state.queryProvenance === "self_match"
              ? "border-amber-300 bg-amber-50 text-amber-900 dark:border-amber-700 dark:bg-amber-900/30 dark:text-amber-200"
              : "border-purple-300 bg-purple-50 text-purple-900 dark:border-purple-700 dark:bg-purple-900/30 dark:text-purple-200"
          }`}>
            {provenance.banner}
          </div>
        )}

        {/* Calibration strip (z-score vs locked threshold) */}
        <CalibrationStrip
          score={state.openSetScore}
          threshold={state.openSetThreshold}
          decision={state.openSetDecision}
        />

        {/* Verdict note */}
        <p className="text-sm italic text-slate-500 dark:text-slate-400">{verdict.note}</p>

        {/* Phase 9.5 — partial-fragment explorer */}
        <FragmentSelector state={state} onFragmentResult={onFragmentResult} />

        {/* Ground-truth row (neutral slate, never green/red) — only when we know
            the expected PID from the provenance hash. */}
        {state.expectedPersonId && (
          <div className="rounded-lg bg-slate-100 px-4 py-3 text-sm text-slate-700 dark:bg-slate-800 dark:text-slate-200">
            <div className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">
              Ground truth (from upload hash)
            </div>
            <div className="mt-1 font-mono text-xs text-slate-600 dark:text-slate-300">
              {state.expectedPersonId}
            </div>
            <div className="mt-1 text-xs text-slate-500 dark:text-slate-400">
              {state.results[0]?.person_id === state.expectedPersonId
                ? "Top-1 result matches this person."
                : state.results.some((r) => r.person_id === state.expectedPersonId)
                ? "Expected person is in the top-5 but not at rank 1."
                : "Expected person is not in the top-5."}
            </div>
          </div>
        )}

        {/* Top-K list (desaturated when rejected OR calibration is unavailable) */}
        <div className={listOpacityClass}>
          {isRejected && (
            <p className="mb-2 text-xs italic text-slate-500 dark:text-slate-400">
              Closest candidates (below identification threshold — listed for context, not as predictions).
            </p>
          )}
          {isCalibrationUnavailable && (
            <p className="mb-2 text-xs italic text-slate-500 dark:text-slate-400">
              Closest candidates (calibration JSON missing — listed for context, NOT as calibrated identifications).
            </p>
          )}
          <div
            className="grid grid-cols-[2.5rem_1fr_auto_6rem] items-center gap-3 px-4 pb-1 text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400"
          >
            <span>Rank</span>
            <span>Candidate</span>
            <span className="hidden items-center text-center sm:flex sm:justify-center">
              Similarity
            </span>
            <span className="flex items-center justify-end text-right">
              Percentile
              <InfoHint text={PERCENTILE_HINT} />
            </span>
          </div>
          <ol className="space-y-2">
            {state.results.map((r, idx) => {
              // Per-rank tier colors (less prominent than the verdict tone).
              const tier =
                idx === 0
                  ? { bar: "bg-slate-500", bg: "bg-slate-200 dark:bg-slate-800" }
                  : { bar: "bg-slate-400", bg: "bg-slate-200 dark:bg-slate-800" };
              const relative = (r.similarity - bottomSim) / range;
              const widthPct = Math.max(8, Math.round(8 + relative * 92));
              const isGroundTruth = r.person_id === state.expectedPersonId;
              const pct = r.similarity_percentile;
              return (
                <li
                  key={r.person_id}
                  className={`grid grid-cols-[2.5rem_1fr_auto_6rem] items-center gap-3 rounded-xl border px-4 py-3 ${
                    isGroundTruth
                      ? "border-slate-400 bg-slate-50 dark:border-slate-500 dark:bg-slate-800/40"
                      : "border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900"
                  }`}
                >
                  <span className="text-lg font-semibold text-slate-400">#{r.rank}</span>
                  <div>
                    <div className="font-medium">
                      {r.fake_name}
                      {isGroundTruth && (
                        <span className="ml-2 text-xs font-normal text-slate-500 dark:text-slate-400">
                          (ground truth)
                        </span>
                      )}
                    </div>
                    <div className="text-xs text-slate-500 dark:text-slate-400">
                      {r.n_teeth ?? "?"} teeth
                    </div>
                  </div>
                  <div className={`hidden h-2 w-32 overflow-hidden rounded-full sm:block ${tier.bg}`}>
                    <div
                      className={`h-full ${tier.bar}`}
                      style={{ width: `${widthPct}%` }}
                      aria-hidden="true"
                    />
                  </div>
                  <span className="text-right font-mono text-sm tabular-nums">
                    {pct === null || pct === undefined
                      ? "—"
                      : `${(pct * 100).toFixed(0)}%`}
                  </span>
                </li>
              );
            })}
          </ol>
        </div>
      </div>

      <details className="border-t border-slate-200 px-6 py-4 text-sm dark:border-slate-800">
        <summary className="cursor-pointer font-medium text-slate-700 dark:text-slate-200">
          Technical details
        </summary>

        <div className="mt-3 space-y-4">
          <div>
            <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
              Raw similarities (cosine)
            </h4>
            <ol className="mt-2 space-y-0.5 font-mono text-xs tabular-nums text-slate-600 dark:text-slate-300">
              {state.results.map((r) => (
                <li key={r.person_id}>
                  #{r.rank} {r.fake_name}: {r.similarity.toFixed(6)} · gap-to-#2:{" "}
                  {r.rank === 1 ? state.topGap.toFixed(6) : "—"}
                </li>
              ))}
            </ol>
          </div>

          <div>
            <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
              Pipeline timings (ms)
            </h4>
            <dl className="mt-2 grid grid-cols-2 gap-3 sm:grid-cols-4">
              {Object.entries(state.timings).map(([k, v]) => (
                <div key={k}>
                  <dt className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">
                    {k}
                  </dt>
                  <dd className="font-mono text-sm tabular-nums">{v.toFixed(1)}</dd>
                </div>
              ))}
            </dl>
          </div>

          {state.toothContributions && state.toothContributions.length > 0 && (
            <ToothContributions contributions={state.toothContributions} />
          )}

          <p className="text-xs text-slate-500 dark:text-slate-400">
            Aggregation: mean pooling · embedder: FDI-init · gallery size: 1,178 persons.
            Open-set calibration: Phase 8.6 (locked z-threshold).
          </p>
        </div>
      </details>

      <footer className="border-t border-slate-200 px-6 py-3 text-right dark:border-slate-800">
        <button
          type="button"
          onClick={onReset}
          className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:text-slate-200 dark:hover:bg-slate-800"
        >
          Try another
        </button>
      </footer>
    </section>
  );
}

const CONTRIBUTION_HINT =
  "For each detected tooth, the dot product of its individual embedding with the top-1 person's gallery profile. Higher = that tooth pushed harder toward this match. Sorted by contribution.";

function ToothContributions({ contributions }: { contributions: ToothContribution[] }) {
  const minSim = Math.min(...contributions.map((c) => c.similarity_to_top1));
  const maxSim = Math.max(...contributions.map((c) => c.similarity_to_top1));
  const range = Math.max(1e-6, maxSim - minSim);

  return (
    <div>
      <h4 className="flex items-center text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
        Per-tooth contribution to the top match
        <InfoHint text={CONTRIBUTION_HINT} />
      </h4>
      <div className="mt-2 overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="text-xs text-slate-400 dark:text-slate-500">
            <tr>
              <th className="py-1 text-left">FDI</th>
              <th className="py-1 text-right">Similarity → top-1</th>
              <th className="py-1 text-left pl-3">&nbsp;</th>
              <th className="py-1 text-right">FDI confidence</th>
            </tr>
          </thead>
          <tbody>
            {contributions.map((c, i) => {
              const widthPct = Math.max(
                4,
                Math.round(((c.similarity_to_top1 - minSim) / range) * 100),
              );
              const isTop = i < 3;
              return (
                <tr
                  key={`${c.fdi}-${i}`}
                  className="border-t border-slate-100 dark:border-slate-800"
                >
                  <td className="py-1 font-mono">{c.fdi}</td>
                  <td className="py-1 text-right font-mono tabular-nums">
                    {c.similarity_to_top1.toFixed(4)}
                  </td>
                  <td className="py-1 pl-3">
                    <div className="h-2 w-32 overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
                      <div
                        className={`h-full ${isTop ? "bg-emerald-500" : "bg-slate-400"}`}
                        style={{ width: `${widthPct}%` }}
                      />
                    </div>
                  </td>
                  <td className="py-1 text-right font-mono tabular-nums">
                    {(c.fdi_confidence * 100).toFixed(0)}%
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
