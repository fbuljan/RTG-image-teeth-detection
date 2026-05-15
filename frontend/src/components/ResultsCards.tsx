"use client";

import { InfoHint } from "@/components/InfoHint";
import type { SearchResult, ToothContribution } from "@/lib/api";

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
};

type Props = {
  state: ResultsState;
  onReset: () => void;
};

// Copy aligned with the Phase 6 plan failure-mode table.
const CONFIDENCE_COPY: Record<ResultsState["confidence"], { label: string; tone: string; note: string }> = {
  high: {
    label: "High confidence",
    tone: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-300",
    note: "Top match is well-separated from the runners-up.",
  },
  medium: {
    label: "Medium confidence",
    tone: "bg-amber-500/15 text-amber-700 dark:text-amber-300",
    note: "Top match leads, but the gap to the runner-up is small.",
  },
  uncertain: {
    label: "Uncertain",
    tone: "bg-orange-500/20 text-orange-700 dark:text-orange-300",
    note: "Top candidates are very close — system is uncertain. Manual review recommended.",
  },
  low: {
    label: "Low confidence",
    tone: "bg-rose-500/20 text-rose-700 dark:text-rose-300",
    note: "No strong match found in the registry.",
  },
};

// Color tiers for the similarity bar. The actual cosine similarities are all
// >0.99 so absolute thresholds aren't useful — we color *relative to this
// result set* so the visual hierarchy mirrors the rank ordering.
const BAR_TIERS: Array<{ bar: string; bg: string }> = [
  { bar: "bg-emerald-500", bg: "bg-emerald-100 dark:bg-emerald-950" },
  { bar: "bg-amber-500", bg: "bg-amber-100 dark:bg-amber-950" },
  { bar: "bg-amber-400", bg: "bg-amber-100/70 dark:bg-amber-950/70" },
  { bar: "bg-slate-400", bg: "bg-slate-100 dark:bg-slate-800" },
  { bar: "bg-slate-300", bg: "bg-slate-100 dark:bg-slate-800" },
];

const SIMILARITY_HINT =
  "Bar length is the candidate's similarity rescaled within these top-5 results. Color shows rank order (green = #1, amber = runners-up, slate = lower). Useful for eyeballing how close the matches are — the actual number is in the Score column.";

const SCORE_HINT =
  "Cosine similarity between the query vector and this person's gallery profile. Both are 128-dim L2-normalized embeddings, so the score is a dot product in [-1, 1] (1.0 = same direction). The query is built by averaging the embeddings of every detected tooth in your X-ray; each gallery profile is the same average over all the registered teeth for that person.";

const CONFIDENCE_HINT =
  "Heuristic from top-1 score and the gap to #2, calibrated against 20 test queries:\n• High: gap ≥ 0.003 (top quartile)\n• Medium: 0.001 ≤ gap < 0.003 (middle)\n• Uncertain: gap < 0.001 (worst quartile — top candidates indistinguishable)\n• Low: top-1 score below 0.7 (sanity floor — none of these queries reach it).";

export function ResultsCards({ state, onReset }: Props) {
  const conf = CONFIDENCE_COPY[state.confidence];
  // Scale the bar length against the spread of the visible results so tiny
  // differences (e.g. 0.9997 vs 0.9957) are still visually distinguishable.
  const topSim = state.results[0]?.similarity ?? 1;
  const bottomSim =
    state.results.length > 1
      ? state.results[state.results.length - 1].similarity
      : topSim - 0.01;
  const range = Math.max(1e-6, topSim - bottomSim);

  return (
    <section className="rounded-2xl border border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-slate-900">
      <header className="flex flex-col gap-3 border-b border-slate-200 px-6 py-4 dark:border-slate-800 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-lg font-semibold">Results</h2>
          <p className="text-sm text-slate-500 dark:text-slate-400">
            {`Queried with ${state.nQueryTeeth} teeth · gap to #2: ${state.topGap.toFixed(5)}`}
            {state.nDropped > 0 ? ` · ${state.nDropped} duplicates dropped` : ""}
          </p>
        </div>
        <span className={`inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold ${conf.tone}`}>
          {conf.label}
          <InfoHint text={CONFIDENCE_HINT} />
        </span>
      </header>

      <div className="space-y-3 px-6 py-5">
        {state.selectedFakeName && (
          <div className="rounded-lg bg-slate-100 px-4 py-3 text-sm text-slate-700 dark:bg-slate-800 dark:text-slate-200">
            Expected:&nbsp;
            <span className="font-semibold">{state.selectedFakeName}</span>
            {state.results[0]?.person_id === state.selectedPersonId ? (
              <span className="ml-2 rounded-full bg-emerald-500/15 px-2 py-0.5 text-xs font-semibold text-emerald-700 dark:text-emerald-300">
                ✓ matched at rank 1
              </span>
            ) : state.results.some((r) => r.person_id === state.selectedPersonId) ? (
              <span className="ml-2 rounded-full bg-amber-500/15 px-2 py-0.5 text-xs font-semibold text-amber-700 dark:text-amber-300">
                Not at rank 1, but in top 5
              </span>
            ) : (
              <span className="ml-2 rounded-full bg-rose-500/15 px-2 py-0.5 text-xs font-semibold text-rose-700 dark:text-rose-300">
                Not in top 5
              </span>
            )}
          </div>
        )}

        <p className="text-sm italic text-slate-500 dark:text-slate-400">{conf.note}</p>

        <div
          className="grid grid-cols-[2.5rem_1fr_auto_6rem] items-center gap-3 px-4 pb-1 text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400"
        >
          <span>Rank</span>
          <span>Candidate</span>
          <span className="hidden items-center text-center sm:flex sm:justify-center">
            Similarity
            <InfoHint text={SIMILARITY_HINT} />
          </span>
          <span className="flex items-center justify-end text-right">
            Score
            <InfoHint text={SCORE_HINT} />
          </span>
        </div>
        <ol className="space-y-2">
          {state.results.map((r, idx) => {
            const isExpected = r.person_id === state.selectedPersonId;
            // Relative bar length: normalize within the visible 5 so the worst
            // entry still shows a stub and the best fills the bar.
            const relative = (r.similarity - bottomSim) / range;
            const widthPct = Math.max(8, Math.round(8 + relative * 92));
            const tier = BAR_TIERS[Math.min(idx, BAR_TIERS.length - 1)];
            return (
              <li
                key={r.person_id}
                className={`grid grid-cols-[2.5rem_1fr_auto_6rem] items-center gap-3 rounded-xl border px-4 py-3 ${
                  isExpected
                    ? "border-emerald-300 bg-emerald-50 dark:border-emerald-700 dark:bg-emerald-900/30"
                    : "border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900"
                }`}
              >
                <span className="text-lg font-semibold text-slate-400">#{r.rank}</span>
                <div>
                  <div className="font-medium">{r.fake_name}</div>
                  <div className="text-xs text-slate-500 dark:text-slate-400">
                    {r.n_teeth ?? "?"} teeth
                  </div>
                </div>
                <div
                  className={`hidden h-2 w-32 overflow-hidden rounded-full sm:block ${tier.bg}`}
                >
                  <div
                    className={`h-full ${tier.bar}`}
                    style={{ width: `${widthPct}%` }}
                    aria-hidden="true"
                  />
                </div>
                <span className="text-right font-mono text-sm tabular-nums">
                  {r.similarity.toFixed(6)}
                </span>
              </li>
            );
          })}
        </ol>
      </div>

      <details className="border-t border-slate-200 px-6 py-4 text-sm dark:border-slate-800">
        <summary className="cursor-pointer font-medium text-slate-700 dark:text-slate-200">
          Technical details
        </summary>

        <div className="mt-3 space-y-4">
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
