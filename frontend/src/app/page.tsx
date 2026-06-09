"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { ExamplePicks } from "@/components/ExamplePicks";
import { ModelCard } from "@/components/ModelCard";
import { PipelineProgress, type PipelineState } from "@/components/PipelineProgress";
import { RegistryList, type RegistryListHandle } from "@/components/RegistryList";
import { AboutDemo } from "@/components/AboutDemo";
import { EnrolModal } from "@/components/EnrolModal";
import { SessionEnrolments } from "@/components/SessionEnrolments";
import { ResultsCards, type ResultsState } from "@/components/ResultsCards";
import { useToasts } from "@/components/Toaster";
import { UploadZone } from "@/components/UploadZone";
import { getOrMintSessionId, type RegistryPerson, type StageEvent } from "@/lib/api";
import { streamIdentify, type PipelineMode } from "@/lib/identify";

const DEFAULT_MODE: PipelineMode = "segmentation";

const INITIAL_PIPELINE: PipelineState = {
  stageA: "idle",
  fdi: "idle",
  embed: "idle",
  search: "idle",
  status: "Drop an X-ray and click Identify to start.",
  currentImageUrl: null,
  warnings: [],
  mode: DEFAULT_MODE,
};

export default function Page() {
  const [selected, setSelected] = useState<RegistryPerson | undefined>();
  const [busy, setBusy] = useState(false);
  const [mode, setMode] = useState<PipelineMode>(DEFAULT_MODE);
  const [pipeline, setPipeline] = useState<PipelineState>(INITIAL_PIPELINE);
  const [results, setResults] = useState<ResultsState | null>(null);
  // Phase 9.5 — per-tooth metadata captured during embed-stage, consumed when search completes.
  const perToothRef = useRef<import("@/lib/api").PerTooth[]>([]);
  const registryRef = useRef<RegistryListHandle | null>(null);
  const pipelineRef = useRef<HTMLDivElement | null>(null);
  const resultsRef = useRef<HTMLDivElement | null>(null);
  const toasts = useToasts();
  // Phase 9.7 — session enrolment state. session_id is minted lazily on first
  // mount and persisted in localStorage; the parent passes it down to the
  // modal + identify stream so identify-with-session-merge works.
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [showEnrolModal, setShowEnrolModal] = useState(false);
  const [enrolmentsNonce, setEnrolmentsNonce] = useState(0);

  useEffect(() => {
    let cancelled = false;
    getOrMintSessionId().then((sid) => {
      if (!cancelled) setSessionId(sid);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const clearResults = useCallback(() => {
    setPipeline(INITIAL_PIPELINE);
    setResults(null);
  }, []);

  const tryAnother = useCallback(() => {
    clearResults();
    registryRef.current?.scrollIntoView();
  }, [clearResults]);

  const onIdentify = useCallback(
    async (file: File) => {
      setBusy(true);
      setResults(null);
      // Show the just-uploaded image in the pipeline panel right away so the
      // user always sees something (rather than the "Awaiting input"
      // placeholder), even if the first SSE event is still in flight.
      const uploadedPreview = URL.createObjectURL(file);
      setPipeline({
        ...INITIAL_PIPELINE,
        status: "Uploading…",
        currentImageUrl: uploadedPreview,
        mode,
      });

      // Scroll the pipeline panel into view so the user actually sees the
      // overlays as they stream in. Small delay so the panel mounts first.
      requestAnimationFrame(() => {
        pipelineRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      });

      try {
        // Reset the per-tooth cache for this new query (Phase 9.5).
        perToothRef.current = [];
        for await (const evt of streamIdentify(file, {
          mode,
          // Phase 9.7 — when present, backend merges this session's
          // enrolments into the top-K. Calibration stays canonical-only.
          sessionId: sessionId ?? undefined,
        })) {
          // Phase 9.5 — capture per-tooth metadata from embed stage for fragment-search.
          if (
            evt.event === "stage_complete"
            && evt.data.stage === "embed"
            && Array.isArray(evt.data.per_tooth)
          ) {
            perToothRef.current = evt.data.per_tooth;
          }
          applyEvent(evt, setPipeline, setResults, selected, perToothRef.current);
          if (evt.event === "warning") {
            toasts.push({
              level: "warning",
              title: "Pipeline warning",
              message: evt.data.message,
            });
          }
          if (evt.event === "error") {
            toasts.push({
              level: "error",
              title: "Pipeline error",
              message: evt.data.message,
            });
          }
          if (evt.event === "stage_complete" && evt.data.stage === "search") {
            // Once results land, scroll down to them.
            requestAnimationFrame(() => {
              resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
            });
          }
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        setPipeline((prev) => ({ ...prev, error: msg, status: "Pipeline failed" }));
        toasts.push({
          level: "error",
          title: "Pipeline failed",
          message: msg,
        });
      } finally {
        setBusy(false);
        // Release the blob preview once the pipeline has produced its own
        // overlays (which replaced currentImageUrl by then anyway).
        setTimeout(() => URL.revokeObjectURL(uploadedPreview), 5000);
      }
    },
    [selected, toasts, mode, sessionId],
  );

  // Phase 9.7 — "Verify by re-querying" from the EnrolModal triggers a normal
  // identify on the freshly-enrolled panoramic; the session-merged top-K
  // should put the session entry at rank 1 with sim ≈ 1.0.
  const verifyEnrolment = useCallback(
    (file: File) => {
      setShowEnrolModal(false);
      onIdentify(file);
    },
    [onIdentify],
  );

  return (
    <main className="mx-auto flex w-full max-w-5xl flex-col gap-6 px-4 py-10 sm:px-6">
      <header className="space-y-2">
        <div className="flex items-start justify-between gap-4">
          <h1 className="text-3xl font-bold tracking-tight">
            Dental Identification Demo
          </h1>
          <div className="flex items-center gap-2">
            {sessionId && (
              <button
                type="button"
                onClick={() => setShowEnrolModal(true)}
                className="rounded-lg border border-emerald-300 bg-emerald-50 px-3 py-1.5 text-xs font-medium text-emerald-700 hover:bg-emerald-100 dark:border-emerald-800/60 dark:bg-emerald-950/30 dark:text-emerald-300 dark:hover:bg-emerald-900/40"
              >
                + Enrol new person
              </button>
            )}
            <AboutDemo />
          </div>
        </div>
        <p className="max-w-3xl text-sm text-slate-600 dark:text-slate-300">
          Pick a person from the registry, download their panoramic X-ray as
          <code className="mx-1 rounded bg-slate-100 px-1 py-0.5 text-xs dark:bg-slate-800">
            xray.png
          </code>
          , then drop it back into the upload zone. The system will detect
          teeth, number them, embed each one, and search a registry of 1,178
          enrolled persons for the closest match.{" "}
          <span className="text-slate-500 dark:text-slate-400">
            (Example panoramics are themselves enrolled — re-uploading one is a self-match. Use
            the fragment-size chips on the results, or upload a novel X-ray, to see real retrieval.)
          </span>
        </p>
      </header>

      <ExamplePicks
        onSelect={(person) => {
          setSelected(person);
          clearResults();
        }}
      />

      <RegistryList
        ref={registryRef}
        selectedPersonId={selected?.person_id}
        onSelect={(person) => {
          setSelected(person);
          clearResults();
        }}
      />

      {sessionId && (
        <SessionEnrolments
          sessionId={sessionId}
          refreshNonce={enrolmentsNonce}
          onChanged={() => setEnrolmentsNonce((n) => n + 1)}
        />
      )}

      <UploadZone
        busy={busy}
        onIdentify={onIdentify}
        mode={mode}
        onModeChange={setMode}
      />

      {(busy || pipeline.currentImageUrl || pipeline.error) && (
        <div ref={pipelineRef} className="scroll-mt-4">
          <PipelineProgress state={pipeline} />
        </div>
      )}

      {results && (
        <div ref={resultsRef} className="scroll-mt-4">
          <ResultsCards
            state={{
              ...results,
              selectedPersonId: selected?.person_id,
              selectedFakeName: selected?.fake_name,
            }}
            onReset={tryAnother}
            onFragmentResult={(r) => {
              // Phase 9.5 — merge fragment-search payload into the existing
              // ResultsState so the verdict/calibration/list all re-render.
              setResults((prev) =>
                prev
                  ? {
                      ...prev,
                      results: r.results ?? prev.results,
                      confidence: r.confidence ?? prev.confidence,
                      topGap: r.top1_top2_gap ?? prev.topGap,
                      nQueryTeeth: r.n_query_teeth ?? prev.nQueryTeeth,
                      openSetScore: r.open_set_score ?? null,
                      openSetDecision: r.open_set_decision ?? "unknown",
                      openSetThreshold: r.open_set_threshold ?? null,
                      simTop1Percentile: r.sim_top1_percentile ?? null,
                      ageEstimate: r.age_estimate ?? prev.ageEstimate,
                      // Phase 9.5.1 — fragment search recomputes contributions
                      // against the *new* top-1. Use them when present; if the
                      // backend returned [], clear prev's contributions rather
                      // than retain dot products against the previous top-1.
                      toothContributions: r.tooth_contributions ?? [],
                      // Backend explicitly emits timings_ms: {} for fragment
                      // search — replace prev rather than retaining stale
                      // detect/fdi/embed numbers from the original run.
                      timings: r.timings_ms ?? {},
                      // Keep queryId, perTooth, provenance, expectedPersonId from the original run.
                    }
                  : prev,
              );
            }}
          />
        </div>
      )}

      <ModelCard />

      {showEnrolModal && sessionId && (
        <EnrolModal
          sessionId={sessionId}
          onClose={() => setShowEnrolModal(false)}
          onEnrolled={() => setEnrolmentsNonce((n) => n + 1)}
          onVerifyRequest={verifyEnrolment}
        />
      )}
    </main>
  );
}

function mapStage(name: string): keyof PipelineState | null {
  // Backend emits "detect" or "segment" for the first stage depending on mode.
  if (name === "detect" || name === "segment") return "stageA";
  if (name === "fdi" || name === "embed" || name === "search") return name;
  return null;
}

function applyEvent(
  evt: StageEvent,
  setPipeline: React.Dispatch<React.SetStateAction<PipelineState>>,
  setResults: React.Dispatch<React.SetStateAction<ResultsState | null>>,
  selected?: RegistryPerson,
  perTooth?: import("@/lib/api").PerTooth[],
) {
  switch (evt.event) {
    case "stage_start": {
      const stage = mapStage(evt.data.stage);
      if (!stage) return;
      setPipeline((prev) => ({
        ...prev,
        [stage]: "active",
        status: evt.data.message,
        embedProgress:
          stage === "embed" && typeof evt.data.total === "number"
            ? { current: 0, total: evt.data.total }
            : prev.embedProgress,
      }));
      return;
    }
    case "progress": {
      if (evt.data.stage !== "embed") return;
      setPipeline((prev) => ({
        ...prev,
        embedProgress: {
          current: evt.data.current,
          total: evt.data.total,
        },
      }));
      return;
    }
    case "stage_complete": {
      const rawStage = evt.data.stage;
      const stage = mapStage(rawStage);
      if (!stage) return;
      setPipeline((prev) => ({
        ...prev,
        [stage]: "done",
        currentImageUrl: evt.data.annotated_image_url ?? prev.currentImageUrl,
        toothCount:
          typeof evt.data.n_teeth === "number" ? evt.data.n_teeth : prev.toothCount,
      }));
      if (stage === "search" && evt.data.results) {
        setResults({
          results: evt.data.results,
          confidence: evt.data.confidence ?? "uncertain",
          topGap: evt.data.top1_top2_gap ?? 0,
          timings: evt.data.timings_ms ?? {},
          nQueryTeeth: evt.data.n_query_teeth ?? 0,
          nDropped: evt.data.n_dropped ?? 0,
          toothContributions: evt.data.tooth_contributions,
          selectedPersonId: selected?.person_id,
          selectedFakeName: selected?.fake_name,
          // Phase 9.3 — calibrated open-set + provenance from the backend.
          openSetScore: evt.data.open_set_score ?? null,
          openSetDecision: evt.data.open_set_decision ?? "unknown",
          openSetThreshold: evt.data.open_set_threshold ?? null,
          queryProvenance: evt.data.query_provenance ?? "unknown",
          expectedPersonId: evt.data.expected_person_id ?? null,
          simTop1Percentile: evt.data.sim_top1_percentile ?? null,
          ageEstimate: evt.data.age_estimate ?? null,
          // Phase 9.5 — fragment-search support.
          queryId: evt.data.query_id ?? null,
          perTooth: perTooth ?? [],
        });
        setPipeline((prev) => ({ ...prev, status: "Done." }));
      }
      return;
    }
    case "warning":
      setPipeline((prev) => ({
        ...prev,
        warnings: [...prev.warnings, evt.data.message],
      }));
      return;
    case "error":
      setPipeline((prev) => ({ ...prev, error: evt.data.message, status: "Error" }));
      return;
    case "done":
      return;
  }
}
