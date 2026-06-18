"use client";

import { intermediateUrl, type ToothOverlay } from "@/lib/api";

export type StageState = "idle" | "active" | "done";

// Chronological log of teeth that have been embedded so far. `embedded` is
// the per-progress-event slice (last 1-4 teeth); the parent builds the
// running history by concatenation.
export type EmbeddedTooth = {
  fdi: string;
  fdi_confidence: number;
};

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
  // Running list of embedded teeth (FDI + conf) updated live as the embed
  // stage progresses, so users can see numbering decisions in real time
  // instead of an opaque "11/16" counter.
  embeddedTeeth?: EmbeddedTooth[];
  toothCount?: number;
  mode: "detection" | "segmentation";
  // When true, stageA label becomes "Validate" and the FDI stage is hidden
  // (validate folds FDI assignment + dedup + OOD gate).
  cropsMode?: boolean;
  // Per-tooth FDI label + bbox + (optional) polygon, in image-native pixels.
  // When present, the pipeline panel layers SVG outlines + DOM number chips
  // over the user's uploaded image — no overlay PNG fetched on the hot path.
  toothOverlays?: ToothOverlay[];
  imageWidth?: number;
  imageHeight?: number;
};

// 16-color palette mirroring backend/visualization.py PALETTE so the demo
// looks identical to the old PNG renderer for users who saw the prior build.
const OVERLAY_PALETTE = [
  "rgb(231, 76, 60)", "rgb(46, 204, 113)", "rgb(52, 152, 219)", "rgb(241, 196, 15)",
  "rgb(155, 89, 182)", "rgb(26, 188, 156)", "rgb(230, 126, 34)", "rgb(52, 73, 94)",
  "rgb(192, 57, 43)", "rgb(39, 174, 96)", "rgb(41, 128, 185)", "rgb(243, 156, 18)",
  "rgb(142, 68, 173)", "rgb(22, 160, 133)", "rgb(211, 84, 0)", "rgb(44, 62, 80)",
];

function colorForFdi(fdi: string): string {
  const n = Number(fdi);
  if (Number.isFinite(n)) return OVERLAY_PALETTE[n % OVERLAY_PALETTE.length];
  // Non-numeric FDI (shouldn't happen, but be defensive): hash the string.
  let h = 0;
  for (let i = 0; i < fdi.length; i++) h = (h * 31 + fdi.charCodeAt(i)) & 0xff;
  return OVERLAY_PALETTE[h % OVERLAY_PALETTE.length];
}

type Props = {
  state: PipelineState;
};

const STAGE_KEYS: Array<{ key: keyof PipelineState }> = [
  { key: "stageA" },
  { key: "fdi" },
  { key: "embed" },
  { key: "search" },
];

function labelFor(
  key: string,
  mode: "detection" | "segmentation",
  cropsMode: boolean,
): string {
  if (key === "stageA") {
    if (cropsMode) return "Validate";
    return mode === "segmentation" ? "Segment" : "Detect";
  }
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
          {STAGE_KEYS.filter(({ key }) =>
            // Crops mode folds FDI into validate; hide the FDI pill so the
            // bar reads "Validate ✓ — Embed ✓ — Search ✓".
            !(state.cropsMode && key === "fdi")
          ).map(({ key }) => {
            const stage = state[key] as StageState;
            const tone =
              stage === "done"
                ? "bg-emerald-500 text-white"
                : stage === "active"
                ? "bg-amber-500 text-white animate-pulse"
                : "bg-slate-200 text-slate-500 dark:bg-slate-800 dark:text-slate-400";
            return (
              <li key={key} className={`rounded-full px-3 py-1 font-medium ${tone}`}>
                {labelFor(key, state.mode, !!state.cropsMode)}
                {key === "embed" && showProgress ? ` ${showProgress}` : ""}
              </li>
            );
          })}
        </ol>
      </header>

      <div className="px-6 py-6">
        {state.currentImageUrl ? (
          <ImageWithOverlays
            src={intermediateUrl(state.currentImageUrl)}
            overlays={state.toothOverlays}
            imageWidth={state.imageWidth}
            imageHeight={state.imageHeight}
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

        {/* Live FDI confidence list while embed stage is running. */}
        {state.embed === "active" && state.embeddedTeeth && state.embeddedTeeth.length > 0 && (
          <div className="mt-4">
            <h4 className="mb-2 text-center text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
              Embedded so far · live FDI numbering
            </h4>
            <div className="mx-auto flex max-w-3xl flex-wrap justify-center gap-1.5">
              {state.embeddedTeeth.map((t, i) => {
                const conf = t.fdi_confidence;
                const tone =
                  conf >= 0.7
                    ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-200"
                    : conf >= 0.5
                    ? "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-200"
                    : "bg-rose-100 text-rose-800 dark:bg-rose-900/40 dark:text-rose-200";
                return (
                  <span
                    key={`${t.fdi}-${i}`}
                    className={`inline-flex items-center gap-1 rounded-md px-2 py-0.5 font-mono text-xs ${tone}`}
                    title={`FDI ${t.fdi} · classifier confidence ${(conf * 100).toFixed(0)}%`}
                  >
                    <span className="font-semibold">{t.fdi}</span>
                    <span className="opacity-70">{(conf * 100).toFixed(0)}%</span>
                  </span>
                );
              })}
            </div>
          </div>
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

// Renders the user's panoramic with optional SVG polygon outlines and DOM
// label chips for each tooth. All overlay coords arrive in image-native pixel
// space, so the SVG viewBox matches that and CSS scales everything together.
// Labels are HTML so they stay crisp at any display size — the prior
// implementation baked them into a 2775-px PNG that browsers then averaged
// down to ~700px, making numbers fuzzy.
function ImageWithOverlays({
  src,
  overlays,
  imageWidth,
  imageHeight,
}: {
  src: string;
  overlays?: ToothOverlay[];
  imageWidth?: number;
  imageHeight?: number;
}) {
  const hasOverlays =
    overlays && overlays.length > 0 && imageWidth && imageHeight;

  return (
    <div className="relative mx-auto w-full max-w-3xl">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={src}
        alt="Pipeline visualization"
        className="block max-h-[500px] w-full rounded-xl border border-slate-200 object-contain shadow dark:border-slate-800"
      />
      {hasOverlays && (
        <svg
          viewBox={`0 0 ${imageWidth} ${imageHeight}`}
          preserveAspectRatio="xMidYMid meet"
          className="pointer-events-none absolute inset-0 h-full w-full"
        >
          {overlays!.map((t, i) => {
            const color = colorForFdi(t.fdi);
            const stroke = Math.max(3, Math.min(imageWidth!, imageHeight!) * 0.004);
            if (t.polygon && t.polygon.length >= 3) {
              const d =
                t.polygon
                  .map(([x, y], j) => `${j === 0 ? "M" : "L"}${x},${y}`)
                  .join("") + "Z";
              return (
                <path
                  key={i}
                  d={d}
                  fill="none"
                  stroke={color}
                  strokeWidth={stroke}
                  strokeLinejoin="round"
                />
              );
            }
            const [x1, y1, x2, y2] = t.bbox;
            return (
              <rect
                key={i}
                x={x1}
                y={y1}
                width={x2 - x1}
                height={y2 - y1}
                fill="none"
                stroke={color}
                strokeWidth={stroke}
              />
            );
          })}
        </svg>
      )}
      {hasOverlays && (
        <div className="pointer-events-none absolute inset-0">
          {overlays!.map((t, i) => {
            const color = colorForFdi(t.fdi);
            const [x1, y1] = t.bbox;
            // Position label chip at the top-left of the bbox, slightly above.
            const leftPct = (x1 / imageWidth!) * 100;
            const topPct = (y1 / imageHeight!) * 100;
            return (
              <span
                key={i}
                className="absolute -translate-y-full whitespace-nowrap rounded px-1.5 py-0.5 font-mono text-[11px] font-semibold leading-tight text-white shadow-sm sm:text-xs"
                style={{
                  left: `${leftPct}%`,
                  top: `${topPct}%`,
                  backgroundColor: color,
                }}
              >
                {t.fdi}
              </span>
            );
          })}
        </div>
      )}
    </div>
  );
}
