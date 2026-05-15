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
  rank1_std: number;
  rank5_mean: number;
  rank5_std: number;
  rank10_mean: number;
  rank10_std: number;
  mAP_mean: number;
  mAP_std: number;
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

type ModelCardPayload = {
  checkpoint: string;
  run_name: string;
  registry_size: number;
  eval_test?: { verification: VerificationMetrics; retrieval: RetrievalMetrics };
  multi_tooth_sweep?: SweepEntry[];
  forensic_1tooth?: SweepEntry[];
  per_category?: CategoryRow[];
  subgroups?: SubgroupRow[];
  training?: TrainingFacts;
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
      {data.eval_test && <SingleToothBlock metrics={data.eval_test} />}
      {data.multi_tooth_sweep && data.multi_tooth_sweep.length > 0 && (
        <MultiToothBlock sweep={data.multi_tooth_sweep} />
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
        title="Multi-tooth retrieval — headline result"
        hint="For each n_query, hold out N teeth as query, aggregate the rest into a gallery profile, mean-pool both sides. Averaged over multiple trials per query size. This is the regime the deployed demo actually operates in."
      />
      <p className="mb-2 text-xs text-slate-500 dark:text-slate-400">
        With multi-tooth aggregation, Rank-1 climbs from ~10% (one tooth) to
        55% (16 teeth) and Rank-5 to over 92% — the operating regime of the
        live demo.
      </p>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">
            <tr>
              <th className="py-1 text-left">n_query</th>
              <th className="py-1 text-right">Rank-1</th>
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
        Peak is highlighted. Beyond n_query=16 the eligible gallery shrinks (only
        people with many teeth remain), so larger values aren&apos;t directly
        comparable.
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
        title="Demographic and clinical subgroups"
        hint="Single-tooth Rank-1 / AUC stratified by demographic and clinical attributes. Larger gaps mean the model performs unevenly across that dimension."
      />
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
          Multi-tooth aggregation peaks at n_query = 16. Larger query sizes
          shrink the eligible gallery to high-tooth-count subjects and
          aren&apos;t directly comparable.
        </li>
      </ul>
    </div>
  );
}
