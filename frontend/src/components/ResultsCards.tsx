"use client";

import { InfoHint } from "@/components/InfoHint";
import type {
  OpenSetDecision,
  QueryProvenance,
  SearchResult,
  ToothContribution,
} from "@/lib/api";

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
};

type Props = {
  state: ResultsState;
  onReset: () => void;
};

// ---------- Phase 9.3 — Verdict (3-state, replaces 4-tier confidence) ----------

type VerdictTone = {
  label: string;
  badge: string;     // colored badge classes
  note: string;
};

const VERDICT_COPY: Record<"likely_enrolled" | "borderline" | "rejected", VerdictTone> = {
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
};

function classifyVerdict(state: ResultsState): keyof typeof VERDICT_COPY {
  if (state.openSetDecision === "rejected") return "rejected";
  // For in-registry queries, borderline if top1-top2 gap < 0.001 (Phase 8.6 lower tercile).
  if (state.topGap < 0.001) return "borderline";
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
  "Empirical percentile of this similarity within the 740 in-registry sim_top1 values from Phase 8.6 held-out enrolment. 73% means 'this similarity is higher than 73% of correct identifications observed during evaluation.' Much more legible than raw cosine, which clusters at 0.99+ for both correct and incorrect matches.";

const VERDICT_HINT =
  "Calibrated open-set verdict, using the locked Phase 8.6 threshold (z = −0.680 on z-scored sim_top1). Likely enrolled = passes threshold with margin. Borderline = passes threshold but top-1 vs top-2 gap is in the lower tercile. Probably not enrolled = below threshold; the system thinks the query person is NOT in the registry.";

const PROVENANCE_HINT =
  "Self-match: the uploaded image is byte-identical to an enrolled image (sim ≈ 1.0 is tautological). Novel upload: the bytes don't match any enrolled image. Held-out · OOS: a curated out-of-distribution test image (Phase 9.8). The dataset has one panoramic per person, so 'self-match' and 'enrolled' are the same set here.";

// ---------- ResultsCards ----------

export function ResultsCards({ state, onReset }: Props) {
  const verdictKey = classifyVerdict(state);
  const verdict = VERDICT_COPY[verdictKey];
  const provenance = PROVENANCE_COPY[state.queryProvenance];

  const isRejected = state.openSetDecision === "rejected";
  const topSim = state.results[0]?.similarity ?? 1;
  const bottomSim =
    state.results.length > 1
      ? state.results[state.results.length - 1].similarity
      : topSim - 0.01;
  const range = Math.max(1e-6, topSim - bottomSim);

  // For OOS-rejected results we desaturate the list to communicate "these
  // are nearest neighbors, not predictions."
  const listOpacityClass = isRejected ? "opacity-60" : "";

  return (
    <section className="rounded-2xl border border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-slate-900">
      <header className="flex flex-col gap-3 border-b border-slate-200 px-6 py-4 dark:border-slate-800 sm:flex-row sm:items-start sm:justify-between">
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <h2 className="text-lg font-semibold">Results</h2>
            <span
              className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ring-inset ${provenance.chip}`}
            >
              {provenance.label}
              <InfoHint text={PROVENANCE_HINT} />
            </span>
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

        {/* Top-K list (desaturated when rejected) */}
        <div className={listOpacityClass}>
          {isRejected && (
            <p className="mb-2 text-xs italic text-slate-500 dark:text-slate-400">
              Closest candidates (below identification threshold — listed for context, not as predictions).
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
