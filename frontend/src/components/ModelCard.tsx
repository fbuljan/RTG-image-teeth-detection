"use client";

import { useEffect, useState } from "react";

import { InfoHint } from "@/components/InfoHint";
import { API_BASE } from "@/lib/api";

type VerificationMetrics = {
  auc: number;
  eer: number;
  eer_threshold: number;
  num_pairs: number;
  num_positive_pairs: number;
  num_negative_pairs: number;
};

type RetrievalMetrics = {
  rank1_micro: number;
  rank1_macro: number;
  rank5: number;
  rank10: number;
  mAP: number;
};

type SweepEntry = {
  n_query: number;
  method: string;
  n_persons: number;
  n_trials: number;
  rank1_mean: number;
  rank1_std: number | null;
  rank1_ci95_low?: number | null;
  rank1_ci95_high?: number | null;
  rank5_mean: number;
  rank5_std: number | null;
  rank10_mean: number;
  rank10_std: number | null;
  mAP_mean: number | null;
  mAP_std: number | null;
};

type CohortEntry = {
  label: string;
  n: number;
  r1_mean: number;
  r1_ci_low: number;
  r1_ci_high: number;
};

type PersonCohorts = {
  full_test?: CohortEntry;
  all_permanent?: CohortEntry;
  any_deciduous?: CohortEntry;
  age_buckets?: CohortEntry[];
  per_sex?: CohortEntry[];
  honesty_rule_verdict?: string;
};

type RotationSweepRow = {
  n_query: number;
  rank1_mean: number;
  rank1_ci95_low: number | null;
  rank1_ci95_high: number | null;
};

type RotationStress = {
  rotation_deg_max?: number;
  n_persons?: number;
  sweep: RotationSweepRow[];
};

type OpenSetHeadline = {
  auroc_clean?: { point: number; ci95_low: number; ci95_high: number };
  auroc_rotated?: { point: number; ci95_low: number; ci95_high: number };
  threshold_z?: number;
  target_tpr_oos?: number;
  frr_in_registry?: number;
};

type CategoryRow = {
  group: string;
  n_samples: number;
  n_persons: number;
  auc: number;
  rank1_micro: number;
  rank5: number;
  mAP: number;
};

type SubgroupRow = {
  subgroup_type: string;
  group: string;
  n_samples: number;
  n_persons: number;
  auc: number;
  rank1_micro: number;
  rank5: number;
  mAP: number;
};

type YoloMetrics = {
  checkpoint: string;
  task: "detect" | "segment";
  split: string;
  imgsz: number;
  box: { precision: number; recall: number; map50: number; map50_95: number };
  mask?: { precision: number; recall: number; map50: number; map50_95: number };
};

type YoloSummary = {
  detection?: YoloMetrics;
  segmentation?: YoloMetrics;
};

type TrainingFacts = {
  backbone?: string;
  embedding_dim?: number;
  dropout?: number;
  loss?: string;
  loss_margin?: number;
  miner?: string;
  optimizer?: string;
  lr?: number;
  scheduler?: string;
  epochs?: number;
  weight_decay?: number;
  warmup_epochs?: number;
  sampler_p?: number;
  sampler_k?: number;
  crop_mode?: string;
  init_from_classifier?: string | null;
};

type EnsembleSummary = {
  members?: string[];
  weights?: Record<string, number>;
  multi_tooth_sweep?: SweepEntry[];
  forensic_1tooth?: SweepEntry[];
  peak_per_method?: Record<string, SweepEntry>;
};

type ModelCardPayload = {
  checkpoint: string;
  run_name: string;
  registry_size: number;
  default_mode?: "segmentation" | "detection";
  ensemble_available?: boolean;
  eval_test?: { verification: VerificationMetrics; retrieval: RetrievalMetrics };
  multi_tooth_sweep?: SweepEntry[];
  // Legacy GT-crop single-model sweep, kept as the apples-to-apples
  // comparator for the offline GT-crop ensemble block (so the ensemble Δ
  // column compares GT vs GT, not GT-ensemble vs deployed-YOLO-pipeline).
  // Not rendered as a standalone table.
  multi_tooth_sweep_gt_anchor?: SweepEntry[];
  forensic_1tooth?: SweepEntry[];
  per_category?: CategoryRow[];
  subgroups?: SubgroupRow[];
  training?: TrainingFacts;
  yolo?: YoloSummary;
  ensemble?: EnsembleSummary;
  ensemble_yolo?: EnsembleSummary;
  person_cohorts?: PersonCohorts;
  rotation_stress?: RotationStress;
  open_set?: OpenSetHeadline;
};

const PRETTY_SUBGROUP: Record<string, string> = {
  sex: "Sex",
  age_bucket: "Age bucket",
  is_deciduous: "Deciduous dentition",
  erupted: "Erupted",
  root_complete: "Root complete",
};

const PRETTY_CATEGORY: Record<string, string> = {
  incisor: "Incisor",
  canine: "Canine",
  premolar: "Premolar",
  molar: "Molar",
  deciduous_incisor: "Deciduous incisor",
  deciduous_canine: "Deciduous canine",
  deciduous_molar: "Deciduous molar",
};

function fmtPct(x: number | undefined | null, places = 1): string {
  if (x === undefined || x === null || Number.isNaN(x)) return "—";
  return `${(x * 100).toFixed(places)}%`;
}

function fmtNum(x: number | undefined | null, places = 3): string {
  if (x === undefined || x === null || Number.isNaN(x)) return "—";
  return x.toFixed(places);
}

function fmtCount(x: number | undefined | null): string {
  if (x === undefined || x === null || Number.isNaN(x)) return "—";
  return Math.round(x).toLocaleString();
}

export function ModelCard() {
  const [open, setOpen] = useState(false);
  const [data, setData] = useState<ModelCardPayload | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open || data || loading) return;
    setLoading(true);
    fetch(`${API_BASE}/api/model-card`, { cache: "no-store" })
      .then(async (res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return (await res.json()) as ModelCardPayload;
      })
      .then(setData)
      .catch((err) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setLoading(false));
  }, [open, data, loading]);

  return (
    <section className="rounded-2xl border border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-slate-900">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="flex w-full items-center justify-between gap-4 px-6 py-4 text-left"
        aria-expanded={open}
      >
        <div>
          <h2 className="text-lg font-semibold">About the model</h2>
          <p className="text-sm text-slate-500 dark:text-slate-400">
            Evaluation metrics, training setup, demographic and per-anatomy breakdowns.
          </p>
        </div>
        <span className="text-2xl text-slate-400">{open ? "−" : "+"}</span>
      </button>

      {open && (
        <div className="space-y-6 border-t border-slate-200 px-6 py-5 dark:border-slate-800">
          {loading && (
            <p className="text-sm text-slate-500 dark:text-slate-400">Loading metrics…</p>
          )}
          {error && (
            <p className="text-sm text-rose-600 dark:text-rose-400">
              Failed to load: {error}
            </p>
          )}
          {data && <ModelCardBody data={data} />}
        </div>
      )}
    </section>
  );
}

function ModelCardBody({ data }: { data: ModelCardPayload }) {
  return (
    <>
      <Header data={data} />
      {data.yolo && (data.yolo.detection || data.yolo.segmentation) && (
        <YoloBlock yolo={data.yolo} defaultMode={data.default_mode} />
      )}
      {data.eval_test && <SingleToothBlock metrics={data.eval_test} />}
      {data.multi_tooth_sweep && data.multi_tooth_sweep.length > 0 && (
        <MultiToothBlock sweep={data.multi_tooth_sweep} />
      )}
      {data.rotation_stress && data.rotation_stress.sweep.length > 0 && (
        <RotationStressBlock rotation={data.rotation_stress} />
      )}
      {data.open_set && (data.open_set.auroc_clean || data.open_set.auroc_rotated) && (
        <OpenSetBlock openSet={data.open_set} />
      )}
      {data.person_cohorts && (
        <PersonCohortsBlock cohorts={data.person_cohorts} />
      )}
      {data.ensemble && data.ensemble.multi_tooth_sweep && data.ensemble.multi_tooth_sweep.length > 0 && (
        <EnsembleBlock
          ensemble={data.ensemble}
          ensembleYolo={data.ensemble_yolo}
          // GT-crop ensemble vs GT-crop single-model anchor (apples-to-apples).
          // Falls back to the deployed sweep only if the legacy GT anchor is
          // unavailable — but the backend always serves both, so this is a
          // belt-and-braces fallback.
          singleSweep={data.multi_tooth_sweep_gt_anchor ?? data.multi_tooth_sweep}
        />
      )}
      {data.per_category && data.per_category.length > 0 && (
        <CategoryBlock rows={data.per_category} />
      )}
      {data.subgroups && data.subgroups.length > 0 && (
        <SubgroupBlock rows={data.subgroups} />
      )}
      {data.training && <TrainingBlock cfg={data.training} />}
      <Caveats registrySize={data.registry_size} />
    </>
  );
}

function Header({ data }: { data: ModelCardPayload }) {
  return (
    <div className="rounded-xl bg-slate-50 px-4 py-3 text-sm dark:bg-slate-950">
      <div className="font-semibold">{data.run_name}</div>
      <div className="font-mono text-xs text-slate-500 dark:text-slate-400">
        {data.checkpoint}
      </div>
    </div>
  );
}

function SectionTitle({ title, hint }: { title: string; hint?: string }) {
  return (
    <h3 className="mb-2 flex items-center text-sm font-semibold uppercase tracking-wide text-slate-600 dark:text-slate-300">
      {title}
      {hint && <InfoHint text={hint} />}
    </h3>
  );
}

function YoloBlock({
  yolo,
  defaultMode,
}: {
  yolo: YoloSummary;
  defaultMode?: "segmentation" | "detection";
}) {
  type Row = {
    name: string;
    isDefault: boolean;
    task: "detect" | "segment";
    checkpoint: string;
    metrics: { precision: number; recall: number; map50: number; map50_95: number };
    metricsLabel: "Box" | "Mask";
  };
  const rows: Row[] = [];
  if (yolo.detection) {
    rows.push({
      name: "Detection (bbox)",
      isDefault: defaultMode === "detection",
      task: "detect",
      checkpoint: yolo.detection.checkpoint,
      metrics: yolo.detection.box,
      metricsLabel: "Box",
    });
  }
  if (yolo.segmentation) {
    // Show mask metrics for the segmenter — the box numbers are reported as
    // the secondary in a footnote since identification only needs the bbox.
    rows.push({
      name: "Segmentation (instance masks)",
      isDefault: defaultMode === "segmentation",
      task: "segment",
      checkpoint: yolo.segmentation.checkpoint,
      metrics: yolo.segmentation.mask ?? yolo.segmentation.box,
      metricsLabel: yolo.segmentation.mask ? "Mask" : "Box",
    });
  }

  return (
    <div>
      <SectionTitle
        title="Tooth localiser (YOLO)"
        hint="Two YOLO models are deployed. The user picks the backend for each query. Detection returns bounding boxes; segmentation returns instance masks whose tight bbox is used as the crop — closer to the GT-mask crops used during embedder training."
      />
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">
            <tr>
              <th className="py-1 text-left">Model</th>
              <th className="py-1 text-right">Precision</th>
              <th className="py-1 text-right">Recall</th>
              <th className="py-1 text-right">mAP50</th>
              <th className="py-1 text-right">mAP50-95</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.name} className="border-t border-slate-100 dark:border-slate-800">
                <td className="py-2">
                  <div className="font-medium">
                    {row.name}
                    {row.isDefault && (
                      <span className="ml-2 rounded-full bg-amber-500/15 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-amber-700 dark:text-amber-300">
                        Default
                      </span>
                    )}
                  </div>
                  <div className="font-mono text-xs text-slate-500 dark:text-slate-400">
                    {row.metricsLabel} metrics · {row.checkpoint}
                  </div>
                </td>
                <td className="py-2 text-right font-mono tabular-nums">{fmtPct(row.metrics.precision, 1)}</td>
                <td className="py-2 text-right font-mono tabular-nums">{fmtPct(row.metrics.recall, 1)}</td>
                <td className="py-2 text-right font-mono tabular-nums">{fmtPct(row.metrics.map50, 1)}</td>
                <td className="py-2 text-right font-mono tabular-nums">{fmtPct(row.metrics.map50_95, 1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
        Metrics computed on the validation split (179 panoramics). Single class
        (<code>tooth</code>).
      </p>
    </div>
  );
}

function SingleToothBlock({
  metrics,
}: {
  metrics: { verification: VerificationMetrics; retrieval: RetrievalMetrics };
}) {
  const cells: Array<{ label: string; value: string; hint?: string }> = [
    { label: "AUC", value: fmtNum(metrics.verification.auc, 3),
      hint: "Verification AUC — area under the ROC curve over all same-vs-different person pairs from the test split." },
    { label: "EER", value: fmtNum(metrics.verification.eer, 3),
      hint: "Equal Error Rate — false-accept rate at the threshold where it equals the false-reject rate. Lower is better." },
    { label: "Rank-1", value: fmtPct(metrics.retrieval.rank1_micro, 2),
      hint: "Probability that the closest match is the correct person, querying with one held-out tooth against the test-set gallery (178 persons)." },
    { label: "Rank-5", value: fmtPct(metrics.retrieval.rank5, 2) },
    { label: "Rank-10", value: fmtPct(metrics.retrieval.rank10, 2) },
    { label: "mAP", value: fmtPct(metrics.retrieval.mAP, 2),
      hint: "Mean Average Precision over all queries." },
  ];

  return (
    <div>
      <SectionTitle
        title="Single-tooth retrieval — baseline"
        hint="One tooth as query, all other test teeth as gallery (178 unseen persons). This is the hardest setting and the baseline our work improves upon — the headline numbers in the multi-tooth table below are much higher because they aggregate many teeth per query."
      />
      <p className="mb-2 text-xs text-slate-500 dark:text-slate-400">
        How good the embedder is when given <em>only one tooth</em>. The thesis
        contribution comes from aggregating multiple teeth (see the next
        section).
      </p>
      <div className="grid grid-cols-3 gap-3 sm:grid-cols-6">
        {cells.map((c) => (
          <div
            key={c.label}
            className="rounded-lg border border-slate-200 bg-white px-3 py-2 dark:border-slate-800 dark:bg-slate-950"
          >
            <div className="flex items-center text-[10px] font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
              {c.label}
              {c.hint && <InfoHint text={c.hint} />}
            </div>
            <div className="font-mono text-sm tabular-nums">{c.value}</div>
          </div>
        ))}
      </div>
      <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
        {fmtCount(metrics.verification.num_pairs)} pairs evaluated for
        verification ({fmtCount(metrics.verification.num_positive_pairs)} positive,
        threshold = {fmtNum(metrics.verification.eer_threshold, 3)}).
      </p>
    </div>
  );
}

function MultiToothBlock({ sweep }: { sweep: SweepEntry[] }) {
  const maxR1 = Math.max(...sweep.map((s) => s.rank1_mean));
  return (
    <div>
      <SectionTitle
        title="Multi-tooth retrieval — deployed pipeline"
        hint="Deployed embedder on YOLO crops, searched against the 1,178-person registry. Per row: hold out N teeth, mean-pool both sides, average over trials with 95% bootstrap CIs. Headline R1 = 82.6% at N=16."
      />
      <p className="mb-2 text-xs text-slate-500 dark:text-slate-400">
        With multi-tooth aggregation, Rank-1 climbs from ~3% (one tooth) to
        <strong> 82.6%</strong> (16 teeth) and Rank-5 to over 97%. Same pipeline
        the live demo runs.
      </p>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">
            <tr>
              <th className="py-1 text-left">N teeth</th>
              <th className="py-1 text-right">Rank-1</th>
              <th className="py-1 text-right text-[10px] normal-case">95% CI</th>
              <th className="py-1 text-right">Rank-5</th>
              <th className="py-1 text-right">Rank-10</th>
              <th className="py-1 text-right">mAP</th>
              <th className="py-1 text-left">&nbsp;</th>
            </tr>
          </thead>
          <tbody>
            {sweep.map((row) => {
              const widthPct = Math.max(2, Math.round((row.rank1_mean / Math.max(maxR1, 0.01)) * 100));
              const isPeak = row.rank1_mean === maxR1;
              const hasCI =
                row.rank1_ci95_low !== null && row.rank1_ci95_low !== undefined
                && row.rank1_ci95_high !== null && row.rank1_ci95_high !== undefined;
              return (
                <tr
                  key={row.n_query}
                  className={`border-t border-slate-100 dark:border-slate-800 ${
                    isPeak ? "bg-amber-50 dark:bg-amber-900/20" : ""
                  }`}
                >
                  <td className="py-1 font-mono">{row.n_query}</td>
                  <td className="py-1 text-right font-mono tabular-nums">
                    {fmtPct(row.rank1_mean, 1)}
                  </td>
                  <td className="py-1 text-right font-mono text-[10px] tabular-nums text-slate-500 dark:text-slate-400">
                    {hasCI
                      ? `[${(row.rank1_ci95_low! * 100).toFixed(1)}, ${(row.rank1_ci95_high! * 100).toFixed(1)}]`
                      : "—"}
                  </td>
                  <td className="py-1 text-right font-mono tabular-nums">
                    {fmtPct(row.rank5_mean, 1)}
                  </td>
                  <td className="py-1 text-right font-mono tabular-nums">
                    {fmtPct(row.rank10_mean, 1)}
                  </td>
                  <td className="py-1 text-right font-mono tabular-nums">
                    {fmtPct(row.mAP_mean, 1)}
                  </td>
                  <td className="py-1 pl-3">
                    <div className="h-2 w-24 overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
                      <div
                        className={`h-full ${isPeak ? "bg-amber-500" : "bg-slate-400"}`}
                        style={{ width: `${widthPct}%` }}
                      />
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
        Peak is highlighted. Beyond N=16 the eligible gallery shrinks (only
        people with many teeth remain), so larger values aren&apos;t directly
        comparable.
      </p>
    </div>
  );
}

function RotationStressBlock({ rotation }: { rotation: RotationStress }) {
  const deg = rotation.rotation_deg_max ?? 30;
  return (
    <div>
      <SectionTitle
        title={`Rotation stress (±${deg}°)`}
        hint="Same multi-tooth protocol, but the query panoramic is rotated ±30° before YOLO detection. Rotation is not handled by augmentation in training, so this is a stress test of geometric distribution shift."
      />
      <p className="mb-2 text-xs text-slate-500 dark:text-slate-400">
        Rank-1 drops sharply under rotation. The system is not rotation-invariant;
        in deployment, upright panoramics should be assumed.
      </p>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">
            <tr>
              <th className="py-1 text-left">N teeth</th>
              <th className="py-1 text-right">R1 rotated</th>
              <th className="py-1 text-right text-[10px] normal-case">95% CI</th>
            </tr>
          </thead>
          <tbody>
            {rotation.sweep.map((row) => {
              const hasCI = row.rank1_ci95_low !== null && row.rank1_ci95_high !== null;
              return (
                <tr key={row.n_query} className="border-t border-slate-100 dark:border-slate-800">
                  <td className="py-1 font-mono">{row.n_query}</td>
                  <td className="py-1 text-right font-mono tabular-nums">{fmtPct(row.rank1_mean, 1)}</td>
                  <td className="py-1 text-right font-mono text-[10px] tabular-nums text-slate-500 dark:text-slate-400">
                    {hasCI
                      ? `[${(row.rank1_ci95_low! * 100).toFixed(1)}, ${(row.rank1_ci95_high! * 100).toFixed(1)}]`
                      : "—"}
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

function OpenSetBlock({ openSet }: { openSet: OpenSetHeadline }) {
  const clean = openSet.auroc_clean;
  const rotated = openSet.auroc_rotated;
  return (
    <div>
      <SectionTitle
        title="Open-set rejection"
        hint="Calibrated AUROC for the in-registry / out-of-set decision. The threshold is locked from validation; the live demo applies it to every query but does not show the verdict prominently."
      />
      <p className="mb-2 text-xs text-slate-500 dark:text-slate-400">
        Calibrated on held-out enrolment data with the locked z-scored top-1
        threshold. Clean AUROC = {clean ? clean.point.toFixed(3) : "—"}; rotated
        AUROC = {rotated ? rotated.point.toFixed(3) : "—"}. The decision the
        live demo shows in <em>Technical details</em> uses this calibration.
      </p>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">
            <tr>
              <th className="py-1 text-left">Test condition</th>
              <th className="py-1 text-right">AUROC</th>
              <th className="py-1 text-right text-[10px] normal-case">95% CI</th>
            </tr>
          </thead>
          <tbody>
            {clean && (
              <tr className="border-t border-slate-100 dark:border-slate-800">
                <td className="py-1">Clean</td>
                <td className="py-1 text-right font-mono tabular-nums">{clean.point.toFixed(3)}</td>
                <td className="py-1 text-right font-mono text-[10px] tabular-nums text-slate-500 dark:text-slate-400">
                  [{clean.ci95_low.toFixed(3)}, {clean.ci95_high.toFixed(3)}]
                </td>
              </tr>
            )}
            {rotated && (
              <tr className="border-t border-slate-100 dark:border-slate-800">
                <td className="py-1">Rotated</td>
                <td className="py-1 text-right font-mono tabular-nums">{rotated.point.toFixed(3)}</td>
                <td className="py-1 text-right font-mono text-[10px] tabular-nums text-slate-500 dark:text-slate-400">
                  [{rotated.ci95_low.toFixed(3)}, {rotated.ci95_high.toFixed(3)}]
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      {(openSet.threshold_z !== undefined || openSet.frr_in_registry !== undefined) && (
        <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
          {openSet.threshold_z !== undefined && (
            <>Locked threshold z = <span className="font-mono">{openSet.threshold_z.toFixed(3)}</span>. </>
          )}
          {openSet.target_tpr_oos !== undefined && (
            <>Target OOS true-rejection {fmtPct(openSet.target_tpr_oos, 0)}. </>
          )}
          {openSet.frr_in_registry !== undefined && (
            <>Trade-off: in-registry false-rejection {fmtPct(openSet.frr_in_registry, 1)}.</>
          )}
        </p>
      )}
    </div>
  );
}

function PersonCohortsBlock({ cohorts }: { cohorts: PersonCohorts }) {
  const rows: Array<{ label: string; cohort?: CohortEntry; emphasis?: "highlight" | "warn" }> = [
    { label: "Full test", cohort: cohorts.full_test },
    { label: "All-permanent dentition", cohort: cohorts.all_permanent, emphasis: "highlight" },
    { label: "Any deciduous tooth", cohort: cohorts.any_deciduous },
  ];
  const ageRows = (cohorts.age_buckets ?? []).map((c) => ({
    label: `Age ${c.label}y`,
    cohort: c,
    emphasis: (c.label === "6-9" ? "warn" : undefined) as "warn" | undefined,
  }));
  const sexRows = (cohorts.per_sex ?? []).map((c) => ({
    label: c.label === "male" ? "Male" : c.label === "female" ? "Female" : c.label,
    cohort: c,
    emphasis: undefined as "highlight" | "warn" | undefined,
  }));

  return (
    <div>
      <SectionTitle
        title="Person-level cohort retrieval"
        hint="Full-panoramic queries against the 1,178-person deployed registry, stratified by dentition stage, age, and sex. Different protocol from the per-tooth-crop subgroup table below."
      />
      <p className="mb-2 text-xs text-slate-500 dark:text-slate-400">
        Person-level R1 splits sharply by dentition stage. The all-permanent
        subset is the dataset&apos;s adult-proxy ceiling (91.4%); the 6-9y
        mixed-dentition cohort is the floor (68.8%). Person-level sex shows no
        gap.
      </p>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">
            <tr>
              <th className="py-1 text-left">Cohort</th>
              <th className="py-1 text-right">n</th>
              <th className="py-1 text-right">R1</th>
              <th className="py-1 text-right text-[10px] normal-case">95% CI</th>
            </tr>
          </thead>
          <tbody>
            {[...rows, ...ageRows, ...sexRows].map(({ label, cohort, emphasis }, idx) => {
              if (!cohort) return null;
              return (
                <tr
                  key={`${label}-${idx}`}
                  className={`border-t border-slate-100 dark:border-slate-800 ${
                    emphasis === "highlight"
                      ? "bg-emerald-50 dark:bg-emerald-900/20"
                      : emphasis === "warn"
                        ? "bg-amber-50 dark:bg-amber-900/20"
                        : ""
                  }`}
                >
                  <td className="py-1">{label}</td>
                  <td className="py-1 text-right font-mono tabular-nums">{cohort.n}</td>
                  <td className="py-1 text-right font-mono tabular-nums">{fmtPct(cohort.r1_mean, 1)}</td>
                  <td className="py-1 text-right font-mono text-[10px] tabular-nums text-slate-500 dark:text-slate-400">
                    [{(cohort.r1_ci_low * 100).toFixed(1)}, {(cohort.r1_ci_high * 100).toFixed(1)}]
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
        The all-permanent cohort is an upper bound on adult deployment, not a
        deployment-ready number — training contained no patients older than 18,
        no restorations, prostheses, or edentulous regions.
      </p>
    </div>
  );
}

function EnsembleSweepTable({
  sweep,
  singleByN,
  emptyLabel = "—",
}: {
  sweep: SweepEntry[];
  singleByN: Record<number, number>;
  emptyLabel?: string;
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">
          <tr>
            <th className="py-1 text-left">n_query</th>
            <th className="py-1 text-right">Ensemble R-1</th>
            <th className="py-1 text-right">Single R-1</th>
            <th className="py-1 text-right">Δ</th>
            <th className="py-1 text-right">R-5</th>
            <th className="py-1 text-right">R-10</th>
            <th className="py-1 text-right">mAP</th>
          </tr>
        </thead>
        <tbody>
          {sweep.map((row) => {
            const singleR1 = singleByN[row.n_query];
            const delta = singleR1 !== undefined ? row.rank1_mean - singleR1 : null;
            return (
              <tr
                key={row.n_query}
                className="border-t border-slate-100 dark:border-slate-800"
              >
                <td className="py-1 font-mono">{row.n_query}</td>
                <td className="py-1 text-right font-mono tabular-nums">
                  {fmtPct(row.rank1_mean, 1)}
                </td>
                <td className="py-1 text-right font-mono tabular-nums text-slate-500 dark:text-slate-400">
                  {singleR1 !== undefined ? fmtPct(singleR1, 1) : emptyLabel}
                </td>
                <td
                  className={`py-1 text-right font-mono tabular-nums ${
                    delta !== null && delta > 0 ? "text-emerald-600 dark:text-emerald-400" : ""
                  }`}
                >
                  {delta !== null
                    ? `${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(1)}pp`
                    : emptyLabel}
                </td>
                <td className="py-1 text-right font-mono tabular-nums">
                  {fmtPct(row.rank5_mean, 1)}
                </td>
                <td className="py-1 text-right font-mono tabular-nums">
                  {fmtPct(row.rank10_mean, 1)}
                </td>
                <td className="py-1 text-right font-mono tabular-nums">
                  {fmtPct(row.mAP_mean, 1)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function EnsembleBlock({
  ensemble,
  ensembleYolo,
  singleSweep,
}: {
  ensemble: EnsembleSummary;
  ensembleYolo?: EnsembleSummary;
  singleSweep?: SweepEntry[];
}) {
  const sweep = (ensemble.multi_tooth_sweep ?? []).slice().sort((a, b) => a.n_query - b.n_query);
  const yoloSweep = (ensembleYolo?.multi_tooth_sweep ?? [])
    .slice()
    .sort((a, b) => a.n_query - b.n_query);
  const singleByN: Record<number, number> = {};
  for (const s of singleSweep ?? []) singleByN[s.n_query] = s.rank1_mean;

  return (
    <div>
      <SectionTitle
        title="Score-level ensemble (offline experiment)"
        hint="Score-level mean of cosine similarities from all four embedders (baseline / masked / metadata / FDI-init). Same evaluation protocol as the single-model multi-tooth sweep above; the delta column compares against the single-model FDI-init headline. Not used in the live demo — see note below."
      />
      <p className="mb-3 text-xs text-slate-500 dark:text-slate-400">
        Members: {(ensemble.members ?? []).join(", ")}.
      </p>

      <h4 className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-600 dark:text-slate-300">
        Ground-truth-crop eval (controlled conditions)
      </h4>
      <EnsembleSweepTable sweep={sweep} singleByN={singleByN} />
      <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
        Both registry and queries use human-drawn red-mask crops — the
        regime the four embedders were trained on. The ensemble decorrelates
        their failure modes and adds ~+20pp Rank-1 at n_query=16 over single-model FDI-init.
      </p>

      {yoloSweep.length > 0 && (
        <>
          <h4 className="mt-4 mb-1 text-xs font-semibold uppercase tracking-wide text-slate-600 dark:text-slate-300">
            YOLO-crop eval (inference-aligned)
          </h4>
          <EnsembleSweepTable sweep={yoloSweep} singleByN={singleByN} />
          <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
            Both registry and queries are rebuilt with the YOLO segmenter to
            match deployment. Gains shrink because the masked and metadata
            members are sensitive to the small differences between YOLO
            crops and the GT crops they were trained on. Still beats the
            single-model GT result of{" "}
            <span className="font-mono">
              {fmtPct(singleByN[16] ?? 0, 1)}
            </span>{" "}
            at n_query=16.
          </p>
        </>
      )}

      <p className="mt-3 rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-900 dark:bg-amber-950/40 dark:text-amber-200">
        <strong>Why this isn&apos;t in the live demo.</strong> Putting the
        ensemble behind a toggle would expose its biggest weakness: the masked
        member needs a polygon, so it breaks under the detection cropping path
        and degrades the overall result. Forcing both members to share the
        YOLO crop pipeline also requires a registry rebuilt with YOLO crops,
        which trivially matches enrolled images to themselves at similarity 1.0
        and obscures what the model is really doing. The single-model path is
        robust across both cropping modes and exhibits the expected sub-1.0
        similarities that make the demo legible. The ensemble is kept as an
        offline result: a clean +20pp under controlled crop conditions, with a
        documented distribution-shift caveat when crop pipelines drift.
      </p>
    </div>
  );
}

function CategoryBlock({ rows }: { rows: CategoryRow[] }) {
  const sorted = [...rows].sort((a, b) => b.rank1_micro - a.rank1_micro);
  return (
    <div>
      <SectionTitle
        title="Per-anatomical-category retrieval"
        hint="Single-tooth Rank-1 and AUC broken down by tooth category. Useful for spotting which anatomies are most identifying."
      />
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">
            <tr>
              <th className="py-1 text-left">Category</th>
              <th className="py-1 text-right">Samples</th>
              <th className="py-1 text-right">Persons</th>
              <th className="py-1 text-right">Rank-1</th>
              <th className="py-1 text-right">AUC</th>
              <th className="py-1 text-right">mAP</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((row) => (
              <tr key={row.group} className="border-t border-slate-100 dark:border-slate-800">
                <td className="py-1">{PRETTY_CATEGORY[row.group] ?? row.group}</td>
                <td className="py-1 text-right font-mono tabular-nums">{fmtCount(row.n_samples)}</td>
                <td className="py-1 text-right font-mono tabular-nums">{fmtCount(row.n_persons)}</td>
                <td className="py-1 text-right font-mono tabular-nums">{fmtPct(row.rank1_micro, 1)}</td>
                <td className="py-1 text-right font-mono tabular-nums">{fmtNum(row.auc, 3)}</td>
                <td className="py-1 text-right font-mono tabular-nums">{fmtPct(row.mAP, 1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function SubgroupBlock({ rows }: { rows: SubgroupRow[] }) {
  // Group by subgroup_type so we can render each demographic separately.
  const grouped = new Map<string, SubgroupRow[]>();
  for (const row of rows) {
    const key = row.subgroup_type;
    if (!grouped.has(key)) grouped.set(key, []);
    grouped.get(key)!.push(row);
  }

  return (
    <div>
      <SectionTitle
        title="Per-tooth-crop subgroup breakdown"
        hint="Single-tooth crop retrieval (one tooth → registry), stratified. Different protocol from person-level R1 above: this is per-crop, not per-person. Use the Person-level cohort table for the deployed multi-tooth retrieval signal."
      />
      <p className="mb-2 text-xs text-slate-500 dark:text-slate-400">
        These rows are per-tooth-crop retrieval (one tooth against the registry,
        n=5,898 crops). The person-level R1 = 82.6% headline uses a different
        protocol (multi-tooth aggregation per person, n=178). Per-tooth and
        per-person numbers are not directly comparable.
      </p>
      <div className="space-y-4">
        {Array.from(grouped.entries()).map(([type, list]) => (
          <div key={type}>
            <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
              {PRETTY_SUBGROUP[type] ?? type}
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="text-xs text-slate-400 dark:text-slate-500">
                  <tr>
                    <th className="py-1 text-left">Group</th>
                    <th className="py-1 text-right">Samples</th>
                    <th className="py-1 text-right">Persons</th>
                    <th className="py-1 text-right">Rank-1</th>
                    <th className="py-1 text-right">AUC</th>
                  </tr>
                </thead>
                <tbody>
                  {list.map((row) => (
                    <tr key={row.group} className="border-t border-slate-100 dark:border-slate-800">
                      <td className="py-1">{row.group}</td>
                      <td className="py-1 text-right font-mono tabular-nums">{fmtCount(row.n_samples)}</td>
                      <td className="py-1 text-right font-mono tabular-nums">{fmtCount(row.n_persons)}</td>
                      <td className="py-1 text-right font-mono tabular-nums">{fmtPct(row.rank1_micro, 1)}</td>
                      <td className="py-1 text-right font-mono tabular-nums">{fmtNum(row.auc, 3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function TrainingBlock({ cfg }: { cfg: TrainingFacts }) {
  const facts: Array<{ label: string; value: string | undefined }> = [
    { label: "Backbone", value: cfg.backbone },
    { label: "Embedding dim", value: cfg.embedding_dim?.toString() },
    { label: "Dropout", value: cfg.dropout?.toString() },
    { label: "Loss", value: cfg.loss },
    { label: "Loss margin", value: cfg.loss_margin?.toString() },
    { label: "Miner", value: cfg.miner },
    { label: "Optimizer", value: cfg.optimizer },
    { label: "Learning rate", value: cfg.lr?.toString() },
    { label: "Scheduler", value: cfg.scheduler },
    { label: "Epochs", value: cfg.epochs?.toString() },
    { label: "Weight decay", value: cfg.weight_decay?.toString() },
    { label: "Warmup epochs", value: cfg.warmup_epochs?.toString() },
    { label: "PK sampler P×K", value: cfg.sampler_p && cfg.sampler_k ? `${cfg.sampler_p}×${cfg.sampler_k}` : undefined },
    { label: "Crop mode", value: cfg.crop_mode },
    { label: "Initialised from", value: cfg.init_from_classifier ?? "ImageNet" },
  ];

  return (
    <div>
      <SectionTitle
        title="Training setup"
        hint="Hyper-parameters and architecture used to train the deployed embedder. Loaded from the checkpoint's config.yaml."
      />
      <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm sm:grid-cols-3">
        {facts
          .filter((f) => f.value !== undefined && f.value !== null && f.value !== "")
          .map((f) => (
            <div key={f.label}>
              <dt className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">
                {f.label}
              </dt>
              <dd className="font-mono text-sm tabular-nums">{f.value}</dd>
            </div>
          ))}
      </dl>
    </div>
  );
}

function Caveats({ registrySize }: { registrySize: number }) {
  return (
    <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-xs leading-relaxed text-amber-900 dark:border-amber-700 dark:bg-amber-950/40 dark:text-amber-200">
      <strong className="block text-sm">Notes on interpretation</strong>
      <ul className="mt-2 list-disc space-y-1 pl-5">
        <li>
          Evaluation metrics are computed on a held-out test set of 178 persons
          unseen during training.
        </li>
        <li>
          The deployed registry contains {registrySize.toLocaleString()} persons
          (train + val + test). That is the deployment scenario, not the
          evaluation scenario.
        </li>
        <li>
          All similarities are typically &gt; 0.99 — the embedder packs persons
          into a small region of the unit sphere. What matters is the
          <em> relative gap </em>between candidates, not the absolute value.
        </li>
        <li>
          Multi-tooth aggregation peaks at N=16. Larger query sizes shrink the
          eligible gallery to high-tooth-count subjects and aren&apos;t directly
          comparable.
        </li>
        <li>
          Training set is pediatric / adolescent (6-18y). Adult dentition with
          restorations, prostheses, or edentulous regions is untested; the
          91.4% all-permanent-cohort R1 is an upper bound, not a deployment
          figure.
        </li>
        <li>
          One panoramic per person in the dataset — cross-visit re-identification
          (different X-ray of the same person) was never measured.
        </li>
        <li>
          The system is not rotation-invariant. Rank-1 drops from 82.6% upright
          to 43.1% at ±30° rotation, and open-set AUROC from 0.832 to 0.609.
        </li>
      </ul>
    </div>
  );
}
