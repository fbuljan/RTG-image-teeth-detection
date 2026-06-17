"use client";

import { useEffect, useRef, useState } from "react";
import { InfoHint } from "@/components/InfoHint";
import type {
  AgeEstimate,
  DropReason,
  ExpectedMatch,
  OpenSetDecision,
  PerCrop,
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
  // Structured list of dropped teeth (currently only FDI duplicates). Empty
  // when the panoramic path saw no dedup collisions; undefined for stale
  // callers.
  dropReasons?: DropReason[];
  toothContributions?: ToothContribution[];
  selectedPersonId?: string;
  selectedFakeName?: string;
  // Calibrated open-set + provenance.
  openSetScore: number | null;
  openSetDecision: OpenSetDecision;
  openSetThreshold: number | null;
  queryProvenance: QueryProvenance;
  expectedPersonId: string | null;
  // Backend-supplied full-registry rank of the expected person — populated
  // whenever expectedPersonId is set. Lets the UI show "expected at #N"
  // when the right person didn't make the visible top-K.
  expectedMatch?: ExpectedMatch | null;
  // Age estimate (sex NOT wired; failed the marginal-accuracy floor).
  ageEstimate: AgeEstimate | null;
  queryId: string | null;
  perTooth: PerTooth[];
  // True when the query came in via /api/identify-crops; flips the results-
  // header copy from "Queried with N teeth" to "Matched from N pre-cropped
  // teeth."
  cropsMode?: boolean;
  // Per-input-crop outcomes (auto-FDI label, OOD status, duplicate-drop
  // status). Only populated in crops mode.
  perCrop?: PerCrop[];
};

type Props = {
  state: ResultsState;
  onReset: () => void;
  onFragmentResult?: (result: import("@/lib/api").FragmentSearchResponse) => void;
  // Session id used to thread session-enrolment merge into fragment search.
  // Without it, the auto-fired N=16 fragment overwrites the parent identify's
  // session-aware top-K with a canonical-only ranking.
  sessionId?: string | null;
};

// Deployed-protocol full-registry priors (sweep_full_registry from the
// deployed-YOLO registry eval — same protocol that defines R1@n=16=82.6% on
// the 1,178-person registry). Used by the FragmentSelector to display
// "expected outcome at this N" so a rank-1 miss reads as a confirmed data
// point not a credibility loss.
const FRAGMENT_PRIORS: Record<number, { r1: number; r5: number }> = {
  1: { r1: 0.029, r5: 0.127 },
  2: { r1: 0.088, r5: 0.281 },
  4: { r1: 0.209, r5: 0.460 },
  8: { r1: 0.446, r5: 0.733 },
  16: { r1: 0.826, r5: 0.973 },
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
  sessionId,
}: {
  state: ResultsState;
  onFragmentResult?: (result: import("@/lib/api").FragmentSearchResponse) => void;
  sessionId?: string | null;
}) {
  const [shuffleSeed, setShuffleSeed] = useState(1);
  const [activeN, setActiveN] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  // Track which queryId we've already auto-fired against, so re-renders
  // don't loop and a fresh query (new queryId) re-triggers.
  const autoFiredFor = useRef<string | null>(null);
  const total = state.perTooth.length;
  const cropsMode = !!state.cropsMode;

  // Panoramic mode: "All N" is excluded because the gallery profile was
  // built from the same teeth → sim ≈ 1.0 tautology. The fragment chips
  // do the real retrieval.
  //
  // Crops mode: the user's upload IS already a fragment by construction
  // (they chose how many crops to send). "All N" here means "use every
  // crop you uploaded against the registry's full 32-tooth gallery" —
  // genuine retrieval, not tautological. Include it as the default.
  const baseSizes = [1, 2, 4, 8, 16].filter((n) => n < total);
  const sizes = cropsMode ? [...baseSizes, total] : baseSizes;

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
      const result = await searchFragment(state.queryId, indices, sessionId ?? undefined);
      onFragmentResult?.(result);
    } catch (e) {
      console.error("fragment search failed", e);
    } finally {
      setBusy(false);
    }
  }

  // Panoramic mode only: auto-fire the largest fragment (capped at 16) so
  // the initial display shows real retrieval instead of the tautological
  // full-set sim ≈ 1.0. Crops mode skips this — the user's upload IS the
  // query they want measured, no auto-subsampling.
  useEffect(() => {
    if (cropsMode) return;
    if (!state.queryId || total === 0) return;
    if (autoFiredFor.current === state.queryId) return;
    const defaultN = sizes.length > 0 ? sizes[sizes.length - 1] : null;
    if (defaultN === null) return;
    autoFiredFor.current = state.queryId;
    runAt(defaultN);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.queryId, total, cropsMode]);

  if (total === 0 || !state.queryId) return null;

  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 dark:border-slate-700 dark:bg-slate-800/40">
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <span className="font-semibold text-slate-700 dark:text-slate-200">Fragment size:</span>
        {sizes.map((n) => {
          const prior = FRAGMENT_PRIORS[n];
          const isAmber = n <= 4;
          const isActive = activeN === n;
          const isAll = n === total;
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
                  ? `Pre-registered prior at N=${n}: R1 = ${(prior.r1 * 100).toFixed(0)}%, R5 = ${(prior.r5 * 100).toFixed(0)}%`
                  : `Query with ${n} detected teeth`
              }
            >
              {isAll ? `All (${n})` : n}
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
            "Re-run the search using only N of the uploaded teeth. Expected R1 at each size: N=1 ≈ 3%, N=2 ≈ 9%, N=4 ≈ 21%, N=8 ≈ 45%, N=16 ≈ 83%."
          }
        />
      </div>
    </div>
  );
}

// ---------- Calibration strip (disabled) ----------
// The strip plotted the query's z-scored open-set similarity against the
// locked decision threshold so a viewer could see "probably enrolled vs
// not." It read as the demo excusing itself: in a closed-world registry
// every query IS in the registry by construction, so the strip turned a
// successful retrieval into a hedged "we think we got it." Disabled,
// kept in source for the open-world story (real forensic deployment).

// eslint-disable-next-line @typescript-eslint/no-unused-vars
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
  // Distance from threshold in z-score space; positive = above threshold
  // (accepted), negative = below (rejected). Used in the inline tooltip so
  // the user can read "+0.94 above threshold" at a glance.
  const gap = score - threshold;
  const gapSign = gap >= 0 ? "+" : "";
  const gapAbsStr = Math.abs(gap).toFixed(3);
  const gapPhrase = gap >= 0 ? "above" : "below";
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
          title={`Your z-score: ${score.toFixed(3)} (${gapSign}${gapAbsStr} ${gapPhrase} threshold)`}
        />
      </div>
      <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400">
        <span>z = {lo}</span>
        <span className="font-mono" title={`Distance from locked threshold: ${gapSign}${gapAbsStr} z-units (${gapPhrase} threshold)`}>
          your z = <strong>{score.toFixed(3)}</strong>
          {" · "}
          threshold = {threshold.toFixed(3)}
          {" · "}
          gap = <strong className={isAccepted ? "text-emerald-700 dark:text-emerald-400" : "text-rose-700 dark:text-rose-400"}>{gapSign}{gapAbsStr}</strong>
        </span>
        <span>z = +{hi}</span>
      </div>
    </div>
  );
}

// ---------- Tooltips ----------

const SIMILARITY_HINT =
  "Cosine similarity between the query's mean-pooled embedding and each candidate's gallery profile. All embeddings live close to the unit sphere, so most similarities are above 0.9 — what matters is the relative gap between candidates, not the absolute value.";

const CROPS_CALIBRATION_HINT =
  "The open-set threshold was tuned on full panoramics (~16-tooth pools). Crops queries pool over fewer teeth, so the z-score lands well below threshold even on legitimate matches — the reject verdict is the calibration-honest answer, not a model failure. The Top-5 below is shown for transparency, not as a recommendation.";

const OPEN_SET_HINT =
  "z-scored top-1 similarity vs the locked enrolment threshold. Positive gap = the system would accept this query as in-registry; negative = it would reject. Tuned on held-out enrolment data, AUROC 0.832 clean / 0.609 rotated.";

const IN_DB_HINT =
  "Top-1 similarity is above the locked open-set threshold — the system would treat this query as a known person. 'Probably' because the threshold has measurable error: AUROC 0.832 on clean panoramics, 0.609 on rotated.";

const NOT_IN_DB_HINT =
  "Top-1 similarity is below the locked open-set threshold — the system would treat this query as a stranger. 'Probably' because the threshold has measurable error: AUROC 0.832 on clean panoramics, 0.609 on rotated. Drops further for partial-fragment queries.";

const AGE_HINT_CONFIDENT =
  "Dental age estimate from a regression head on the embedder. Pre-registered MAE 0.93y on the 6-13y dense bucket; ±2.5y CI covers the worst per-bucket error observed during evaluation.";

const AGE_HINT_SATURATED =
  "Outside the dense 6-13y training bucket or near the [6, 18] boundary. Head is less reliable here (2.09y MAE in the 16-18y bucket). CI widened to ±3.5y.";

const AGE_HINT_SMALL_POOL =
  "Pool size < 8 teeth. The head was trained on full per-person pools (~33 teeth). Small-pool error wasn't measured during evaluation, so CI widened to ±4y as a conservative band.";

// ---------- ResultsCards ----------

export function ResultsCards({ state, onReset, onFragmentResult, sessionId }: Props) {
  const topSim = state.results[0]?.similarity ?? 1;
  const bottomSim =
    state.results.length > 1
      ? state.results[state.results.length - 1].similarity
      : topSim - 0.01;
  const range = Math.max(1e-6, topSim - bottomSim);

  return (
    <section className="rounded-2xl border border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-slate-900">
      <header className="flex flex-col gap-3 border-b border-slate-200 px-6 py-4 dark:border-slate-800 sm:flex-row sm:items-start sm:justify-between">
        <div className="space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="text-lg font-semibold">Top matches</h2>
            {/* Open-set verdict pill — panoramic mode only. Crops mode shows
                a dedicated amber callout further down because the locked
                threshold mis-fires on small-pool queries. */}
            {!state.cropsMode && state.openSetDecision === "in_registry" && (
              <span
                className="inline-flex items-center rounded-full bg-emerald-500/15 px-2.5 py-0.5 text-xs font-medium text-emerald-700 ring-1 ring-inset ring-emerald-500/30 dark:text-emerald-300"
              >
                Probably in database
                <InfoHint text={IN_DB_HINT} />
              </span>
            )}
            {!state.cropsMode && state.openSetDecision === "rejected" && (
              <span
                className="inline-flex items-center rounded-full bg-slate-200 px-2.5 py-0.5 text-xs font-medium text-slate-700 ring-1 ring-inset ring-slate-300 dark:bg-slate-800 dark:text-slate-200 dark:ring-slate-600"
              >
                Probably not in database
                <InfoHint text={NOT_IN_DB_HINT} />
              </span>
            )}
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
            {state.cropsMode
              ? (() => {
                  const uploaded = state.perCrop?.length ?? state.nQueryTeeth;
                  const failedOod = state.perCrop?.filter((c) => c.failed_ood).length ?? 0;
                  const droppedDup = state.perCrop?.filter((c) => c.dropped_as_duplicate).length ?? state.nDropped;
                  const parts = [`${uploaded} crop${uploaded === 1 ? "" : "s"} uploaded`, `${state.nQueryTeeth} kept`];
                  if (failedOod > 0) parts.push(`${failedOod} rejected (OOD)`);
                  if (droppedDup > 0) parts.push(`${droppedDup} duplicate${droppedDup === 1 ? "" : "s"} dropped`);
                  return parts.join(" · ");
                })()
              : `${state.nQueryTeeth} teeth${state.nDropped > 0 ? ` · ${state.nDropped} duplicate${state.nDropped === 1 ? "" : "s"} dropped` : ""}`}
          </p>
          {state.nDropped > 0 && state.dropReasons && state.dropReasons.length > 0 && (
            <details className="mt-1 text-xs text-slate-500 dark:text-slate-400">
              <summary className="cursor-pointer select-none hover:text-slate-700 dark:hover:text-slate-300">
                Which teeth
              </summary>
              <ul className="mt-1.5 space-y-1 pl-2">
                {state.dropReasons.map((d, i) => (
                  <li key={`${d.fdi}-${i}`} className="font-mono">
                    FDI {d.fdi} ({(d.fdi_confidence * 100).toFixed(0)}%) — duplicate
                    {typeof d.kept_fdi_confidence === "number" && (
                      <span className="opacity-70">
                        {" "}
                        of {(d.kept_fdi_confidence * 100).toFixed(0)}%
                      </span>
                    )}
                  </li>
                ))}
              </ul>
            </details>
          )}
        </div>
      </header>

      <div className="space-y-4 px-6 py-5">
        {/* Partial-fragment explorer. */}
        <FragmentSelector state={state} onFragmentResult={onFragmentResult} sessionId={sessionId} />

        {/* Crops-mode honest verdict. The open-set head was calibrated on
            full panoramics, so it rejects crops queries by construction even
            when retrieval is correct. Surface that up-front instead of
            leading with a confident-looking Top-5. */}
        {state.cropsMode && state.openSetDecision === "rejected" && (
          <div className="rounded-lg border border-amber-300/70 bg-amber-50 px-4 py-3 text-sm dark:border-amber-700/60 dark:bg-amber-950/30">
            <div className="flex items-start gap-2">
              <span className="mt-0.5 inline-flex h-5 w-5 flex-none items-center justify-center rounded-full bg-amber-500 text-[11px] font-bold text-white">!</span>
              <div className="space-y-1">
                <p className="font-semibold text-amber-800 dark:text-amber-200">
                  Open-set verdict: not confident this is in the registry
                </p>
                <p className="text-xs text-amber-700 dark:text-amber-300">
                  z = {state.openSetScore !== null ? state.openSetScore.toFixed(2) : "—"}
                  {" "}vs threshold {state.openSetThreshold !== null ? state.openSetThreshold.toFixed(2) : "—"}
                  {". "}The threshold was tuned for full-panoramic queries
                  (~16 teeth pooled). Smaller crops pools always land below it,
                  so this reject does <em>not</em> mean the top match is wrong —
                  just that the calibration can&apos;t confirm it. Treat the
                  Top-5 as a ranked candidate list, not a verified ID.
                  <InfoHint text={CROPS_CALIBRATION_HINT} />
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Expected-match check — only when we know who the upload should
            resolve to (example panoramic) and only renders the one-line
            verdict, no PID hex string. When the expected person dropped out
            of top-K, backend supplies its full-registry rank so we can show
            "expected at #42 (sim 0.881)" instead of just "not in top-5". */}
        {state.expectedPersonId && (() => {
          const top1Match = state.results[0]?.person_id === state.expectedPersonId;
          const inTopK = state.results.some((r) => r.person_id === state.expectedPersonId);
          let label: string;
          if (top1Match) {
            label = "✓ Expected match at #1";
          } else if (inTopK) {
            label = "Expected match in top-5, not #1";
          } else if (state.expectedMatch) {
            const m = state.expectedMatch;
            label = `Expected match dropped to #${m.rank} (sim ${m.similarity.toFixed(4)})`;
          } else {
            label = "Expected match not in top-5";
          }
          const tone = top1Match
            ? "text-emerald-700 dark:text-emerald-300"
            : "text-slate-500 dark:text-slate-400";
          return <p className={`text-xs ${tone}`}>{label}</p>;
        })()}

        {/* Top-K list */}
        <div>
          <div
            className="grid grid-cols-[2.5rem_1fr_auto_5rem] items-center gap-3 px-4 pb-1 text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400"
          >
            <span>Rank</span>
            <span>Candidate</span>
            <span className="hidden items-center text-center sm:flex sm:justify-center">
              &nbsp;
            </span>
            <span className="flex items-center justify-end text-right">
              Similarity
              <InfoHint text={SIMILARITY_HINT} />
            </span>
          </div>
          <ol className="space-y-2">
            {state.results.map((r, idx) => {
              const tier =
                idx === 0
                  ? { bar: "bg-slate-500", bg: "bg-slate-200 dark:bg-slate-800" }
                  : { bar: "bg-slate-400", bg: "bg-slate-200 dark:bg-slate-800" };
              const relative = (r.similarity - bottomSim) / range;
              const widthPct = Math.max(8, Math.round(8 + relative * 92));
              const isGroundTruth = r.person_id === state.expectedPersonId;
              return (
                <li
                  key={r.person_id}
                  className={`grid grid-cols-[2.5rem_1fr_auto_5rem] items-center gap-3 rounded-xl border px-4 py-3 ${
                    isGroundTruth
                      ? "border-slate-400 bg-slate-50 dark:border-slate-500 dark:bg-slate-800/40"
                      : "border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900"
                  }`}
                >
                  <span className="text-lg font-semibold text-slate-400">#{r.rank}</span>
                  <div>
                    <div className="font-medium">
                      {r.fake_name}
                      {r.is_session && (
                        <span
                          className="ml-2 inline-flex items-center rounded-full bg-emerald-500/15 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-emerald-700 ring-1 ring-inset ring-emerald-500/30 dark:text-emerald-300"
                          title="Enrolled in this browser session."
                        >
                          session
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
                    {r.similarity.toFixed(4)}
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

          {state.cropsMode && state.perCrop && state.perCrop.length > 0 && (
            <PerCropOutcomes perCrop={state.perCrop} />
          )}

          {state.openSetScore !== null && state.openSetThreshold !== null && (
            <div>
              <h4 className="flex items-center text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
                Open-set decision
                <InfoHint text={OPEN_SET_HINT} />
              </h4>
              <p className="mt-2 font-mono text-xs tabular-nums text-slate-600 dark:text-slate-300">
                z = <strong>{state.openSetScore.toFixed(3)}</strong>
                {" · "}threshold = {state.openSetThreshold.toFixed(3)}
                {" · "}gap ={" "}
                <strong className={
                  state.openSetDecision === "in_registry"
                    ? "text-emerald-700 dark:text-emerald-400"
                    : state.openSetDecision === "rejected"
                      ? "text-rose-700 dark:text-rose-400"
                      : "text-slate-700 dark:text-slate-300"
                }>
                  {state.openSetScore - state.openSetThreshold >= 0 ? "+" : ""}
                  {(state.openSetScore - state.openSetThreshold).toFixed(3)}
                </strong>
                {" · "}decision ={" "}
                {state.openSetDecision === "in_registry"
                  ? "accept (in registry)"
                  : state.openSetDecision === "rejected"
                    ? "reject (probably not in registry)"
                    : state.openSetDecision}
              </p>
            </div>
          )}

          <p className="text-xs text-slate-500 dark:text-slate-400">
            Aggregation: mean pooling · gallery size: 1,178 persons. Open-set
            calibration locked from held-out enrolment evaluation (AUROC 0.832
            clean / 0.609 rotated).
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
      {/* No overflow wrapper: the table fits in the Technical Details
          panel, and any overflow-x setting forces overflow-y to clip per
          CSS spec — which would hide the absolute-positioned tooltip in
          the YOLO conf column header. */}
      <div className="mt-2">
        <table className="w-full text-sm">
          <thead className="text-xs text-slate-400 dark:text-slate-500">
            <tr>
              <th className="py-1 text-left">FDI</th>
              <th className="py-1 text-right">Similarity → top-1</th>
              <th className="py-1 text-left pl-3">&nbsp;</th>
              <th className="py-1 text-right">FDI conf.</th>
              {/* YOLO detection confidence column. */}
              <th className="py-1 pl-3 text-right">
                <span className="inline-flex items-center">
                  YOLO conf.
                  <InfoHint text={YOLO_CONF_HINT} />
                </span>
              </th>
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
                  <td className="py-1 pl-3 text-right font-mono tabular-nums">
                    {typeof c.yolo_confidence === "number"
                      ? `${(c.yolo_confidence * 100).toFixed(0)}%`
                      : <span className="opacity-50">—</span>}
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

const YOLO_CONF_HINT =
  "Detector confidence that the tooth box is, in fact, a tooth — not the identification score. Em-dash for pre-cropped uploads (no YOLO).";

const PER_CROP_HINT =
  "What the backend did with each crop you uploaded: which FDI it inferred (or accepted as an override), and whether it survived OOD gating and FDI deduplication. Only the 'kept' crops were embedded and pooled into the query.";

function PerCropOutcomes({ perCrop }: { perCrop: PerCrop[] }) {
  return (
    <div>
      <h4 className="flex items-center text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
        Per-crop outcomes
        <InfoHint text={PER_CROP_HINT} />
      </h4>
      <div className="mt-2 overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="text-xs text-slate-400 dark:text-slate-500">
            <tr>
              <th className="py-1 pr-2 text-left">#</th>
              <th className="py-1 pr-2 text-left">FDI</th>
              <th className="py-1 pr-2 text-right">Conf.</th>
              <th className="py-1 pr-2 text-left">Source</th>
              <th className="py-1 pr-2 text-left">Outcome</th>
            </tr>
          </thead>
          <tbody>
            {perCrop.map((c) => {
              const outcome = c.failed_ood
                ? { label: "rejected (OOD)", tone: "text-rose-700 dark:text-rose-400" }
                : c.dropped_as_duplicate
                  ? { label: "dropped (duplicate)", tone: "text-amber-700 dark:text-amber-400" }
                  : c.kept
                    ? { label: "kept", tone: "text-emerald-700 dark:text-emerald-400" }
                    : { label: "—", tone: "text-slate-500 dark:text-slate-400" };
              return (
                <tr
                  key={c.input_index}
                  className="border-t border-slate-100 dark:border-slate-800"
                >
                  <td className="py-1 pr-2 font-mono">{c.input_index + 1}</td>
                  <td className="py-1 pr-2 font-mono">
                    {c.chosen_fdi}
                    {c.source === "override" && c.chosen_fdi !== c.auto_fdi && (
                      <span className="ml-1 text-[10px] text-slate-500" title={`Auto-detected ${c.auto_fdi}`}>
                        (auto: {c.auto_fdi})
                      </span>
                    )}
                  </td>
                  <td className="py-1 pr-2 text-right font-mono tabular-nums">
                    {(c.auto_fdi_confidence * 100).toFixed(0)}%
                  </td>
                  <td className="py-1 pr-2">
                    <span className={`inline-flex items-center rounded px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide ring-1 ring-inset ${
                      c.source === "override"
                        ? "bg-sky-500/10 text-sky-700 ring-sky-500/30 dark:text-sky-300"
                        : "bg-slate-200/60 text-slate-600 ring-slate-300 dark:bg-slate-800 dark:text-slate-300 dark:ring-slate-700"
                    }`}>
                      {c.source}
                    </span>
                  </td>
                  <td className={`py-1 pr-2 text-xs ${outcome.tone}`}>
                    {outcome.label}
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
