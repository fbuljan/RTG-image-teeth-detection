"use client";

import { intermediateUrl } from "@/lib/api";

export type StageState = "idle" | "active" | "done";

export type PipelineState = {
  stageA: StageState;
  fdi: StageState;
  embed: StageState;
  search: StageState;
  status: string;
  warnings: string[];
  error?: string;
  currentImageUrl: string | null;
  embedProgress?: { current: number; total: number };
  toothCount?: number;
  mode: "detection" | "segmentation";
};

type Props = {
  state: PipelineState;
};

const STAGE_KEYS: Array<{ key: keyof PipelineState }> = [
  { key: "stageA" },
  { key: "fdi" },
  { key: "embed" },
  { key: "search" },
];

const STAGE_LABELS: Record<keyof PipelineState, string> | Record<string, string> = {};
function labelFor(key: string, mode: "detection" | "segmentation"): string {
  if (key === "stageA") return mode === "segmentation" ? "Segment" : "Detect";
  if (key === "fdi") return "Number";
  if (key === "embed") return "Embed";
  if (key === "search") return "Search";
  return key;
}

export function PipelineProgress({ state }: Props) {
  const showProgress =
    state.embed === "active" && state.embedProgress
      ? `${state.embedProgress.current}/${state.embedProgress.total}`
      : null;

  return (
    <section className="rounded-2xl border border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-slate-900">
      <header className="flex flex-col gap-3 border-b border-slate-200 px-6 py-4 dark:border-slate-800 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-lg font-semibold">Pipeline</h2>
          <p className="text-sm text-slate-500 dark:text-slate-400">{state.status}</p>
        </div>
        <ol className="flex items-center gap-2 text-xs">
          {STAGE_KEYS.map(({ key }) => {
            const stage = state[key] as StageState;
            const tone =
              stage === "done"
                ? "bg-emerald-500 text-white"
                : stage === "active"
                ? "bg-amber-500 text-white animate-pulse"
                : "bg-slate-200 text-slate-500 dark:bg-slate-800 dark:text-slate-400";
            return (
              <li key={key} className={`rounded-full px-3 py-1 font-medium ${tone}`}>
                {labelFor(key, state.mode)}
                {key === "embed" && showProgress ? ` ${showProgress}` : ""}
              </li>
            );
          })}
        </ol>
      </header>

      <div className="px-6 py-6">
        {state.currentImageUrl ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={intermediateUrl(state.currentImageUrl)}
            alt="Pipeline visualization"
            className="mx-auto max-h-[500px] w-full max-w-3xl rounded-xl border border-slate-200 object-contain shadow dark:border-slate-800"
          />
        ) : (
          <div className="flex h-64 items-center justify-center rounded-xl border border-dashed border-slate-300 text-sm text-slate-500 dark:border-slate-700 dark:text-slate-400">
            Awaiting input…
          </div>
        )}

        {typeof state.toothCount === "number" && (
          <p className="mt-4 text-center text-sm text-slate-500 dark:text-slate-400">
            {state.toothCount} teeth processed
          </p>
        )}

        {state.warnings.length > 0 && (
          <ul className="mt-4 space-y-2">
            {state.warnings.map((w, i) => (
              <li
                key={`${i}-${w}`}
                className="rounded-lg bg-amber-50 px-4 py-3 text-sm text-amber-800 dark:bg-amber-900/30 dark:text-amber-200"
              >
                {w}
              </li>
            ))}
          </ul>
        )}
        {state.error && (
          <div className="mt-4 rounded-lg bg-rose-50 px-4 py-3 text-sm text-rose-700 dark:bg-rose-900/30 dark:text-rose-200">
            {state.error}
          </div>
        )}
      </div>
    </section>
  );
}
