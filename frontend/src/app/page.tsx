"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { ExamplePicks } from "@/components/ExamplePicks";
import { ModelCard } from "@/components/ModelCard";
import { PipelineProgress, type PipelineState } from "@/components/PipelineProgress";
import { RegistryList, type RegistryListHandle } from "@/components/RegistryList";
import { AboutDemo } from "@/components/AboutDemo";
import { CropsUploadZone } from "@/components/CropsUploadZone";
import { EnrolModal } from "@/components/EnrolModal";
import { SessionEnrolments } from "@/components/SessionEnrolments";
import { ResultsCards, type ResultsState } from "@/components/ResultsCards";
import { useToasts } from "@/components/Toaster";
import { UploadZone } from "@/components/UploadZone";
import { getOrMintSessionId, type RegistryPerson, type StageEvent } from "@/lib/api";
import { streamIdentify, streamIdentifyCrops, type PipelineMode } from "@/lib/identify";

type InputMode = "panoramic" | "crops";

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
  // Input mode switch. "panoramic" routes to /api/identify, "crops" routes
  // to /api/identify-crops. Toggling clears any in-flight results so the
  // user doesn't see stale state from the other path.
  const [inputMode, setInputMode] = useState<InputMode>("panoramic");
  // Per-tooth metadata captured during embed-stage, consumed when search completes.
  const perToothRef = useRef<import("@/lib/api").PerTooth[]>([]);
  const registryRef = useRef<RegistryListHandle | null>(null);
  const pipelineRef = useRef<HTMLDivElement | null>(null);
  const resultsRef = useRef<HTMLDivElement | null>(null);
  const toasts = useToasts();
  // Session enrolment state. session_id is minted lazily on first mount and
  // persisted in localStorage; the parent passes it down to the modal +
  // identify stream so identify-with-session-merge works.
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
        // Explicit false — INITIAL_PIPELINE doesn't carry the key, so a spread
        // alone preserves a stale `true` from a prior crops run. Set it here
        // so the panoramic path always renders the FDI row + correct stage
        // labels even when the prior query was a crops query.
        cropsMode: false,
        // Same shape — undefined here so the prior query's overlay polygons
        // don't render on top of a freshly uploaded panoramic until the new
        // FDI stage event arrives.
        toothOverlays: undefined,
        imageWidth: undefined,
        imageHeight: undefined,
      });

      // Scroll the pipeline panel into view so the user actually sees the
      // overlays as they stream in. Small delay so the panel mounts first.
      requestAnimationFrame(() => {
        pipelineRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      });

      try {
        // Reset the per-tooth cache for this new query.
        perToothRef.current = [];
        for await (const evt of streamIdentify(file, {
          mode,
          // When present, backend merges this session's enrolments into the
          // top-K. Calibration stays canonical-only.
          sessionId: sessionId ?? undefined,
        })) {
          // Capture per-tooth metadata from embed stage for fragment-search.
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

  // "Verify by re-querying" from the EnrolModal triggers a normal identify
  // on the freshly-enrolled panoramic; the session-merged top-K should put
  // the session entry at rank 1 with sim ≈ 1.0.
  const verifyEnrolment = useCallback(
    (file: File) => {
      setShowEnrolModal(false);
      onIdentify(file);
    },
    [onIdentify],
  );

  // Identify-from-crops. Same SSE-driven flow as onIdentify but the first
  // stage is `validate` (OOD gate + FDI assignment + dedup), no panoramic
  // preview, and no detect/fdi stages. Results render via the standard
  // ResultsCards with `crops_mode=true` flipping the header copy.
  const onIdentifyCrops = useCallback(
    async (files: File[], fdiOverrides: (string | null)[]) => {
      setBusy(true);
      setResults(null);
      setPipeline({
        ...INITIAL_PIPELINE,
        status: `Validating ${files.length} crops…`,
        currentImageUrl: null,  // no panoramic in crops mode
        mode,
        cropsMode: true,
        toothOverlays: undefined,
        imageWidth: undefined,
        imageHeight: undefined,
      });
      requestAnimationFrame(() => {
        pipelineRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      });

      try {
        perToothRef.current = [];
        for await (const evt of streamIdentifyCrops(files, {
          fdiOverrides,
          sessionId: sessionId ?? undefined,
        })) {
          if (
            evt.event === "stage_complete"
            && evt.data.stage === "embed"
            && Array.isArray(evt.data.per_tooth)
          ) {
            perToothRef.current = evt.data.per_tooth;
          }
          applyEvent(evt, setPipeline, setResults, selected, perToothRef.current);
          if (evt.event === "warning") {
            toasts.push({ level: "warning", title: "Pipeline warning", message: evt.data.message });
          }
          if (evt.event === "error") {
            toasts.push({ level: "error", title: "Pipeline error", message: evt.data.message });
          }
          if (evt.event === "stage_complete" && evt.data.stage === "search") {
            requestAnimationFrame(() => {
              resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
            });
          }
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        setPipeline((prev) => ({ ...prev, error: msg, status: "Pipeline failed" }));
        toasts.push({ level: "error", title: "Crops pipeline failed", message: msg });
      } finally {
        setBusy(false);
      }
    },
    [selected, toasts, mode, sessionId],
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
          Pick a person, download their panoramic, drop it back into the upload
          zone. The system detects teeth, numbers them, embeds each one, and
          searches a 1,178-person registry for the closest match.
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

      {/* Input-mode tab strip. Switching wipes any in-flight results so the
          user doesn't see panoramic state under a "Matched from N crops"
          header. */}
      <div className="flex flex-wrap items-center gap-1 rounded-full border border-slate-200 bg-slate-100 p-0.5 text-xs dark:border-slate-700 dark:bg-slate-800 self-start">
        {(["panoramic", "crops"] as InputMode[]).map((tab) => {
          const active = inputMode === tab;
          return (
            <button
              key={tab}
              type="button"
              disabled={busy}
              onClick={() => {
                setInputMode(tab);
                clearResults();
              }}
              className={`rounded-full px-3 py-1 font-medium transition disabled:cursor-not-allowed disabled:opacity-50 ${
                active
                  ? "bg-white text-slate-900 shadow dark:bg-slate-950 dark:text-slate-100"
                  : "text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-100"
              }`}
            >
              {tab === "panoramic" ? "Panoramic X-ray" : "Tooth crops"}
            </button>
          );
        })}
      </div>

      {inputMode === "panoramic" ? (
        <UploadZone
          busy={busy}
          onIdentify={onIdentify}
          mode={mode}
          onModeChange={setMode}
        />
      ) : (
        <CropsUploadZone busy={busy} onIdentifyCrops={onIdentifyCrops} />
      )}

      {(busy || pipeline.currentImageUrl || pipeline.error) && (
        <div ref={pipelineRef} className="scroll-mt-4">
          <PipelineProgress state={pipeline} />
        </div>
      )}

      {results && (
        <div ref={resultsRef} className="scroll-mt-4">
          <ResultsCards
            sessionId={sessionId}
            state={{
              ...results,
              selectedPersonId: selected?.person_id,
              selectedFakeName: selected?.fake_name,
            }}
            onReset={tryAnother}
            onFragmentResult={(r) => {
              // Merge fragment-search payload into the existing ResultsState
              // so the verdict/calibration/list all re-render.
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
                      // Fragment search recomputes full-registry rank for the
                      // expected person against the smaller subset's pooled
                      // vector — replace, don't preserve, so the chip always
                      // matches the rendered top-K.
                      expectedMatch: r.expected_match ?? null,
                      ageEstimate: r.age_estimate ?? prev.ageEstimate,
                      // Fragment search recomputes contributions against the
                      // *new* top-1. Use them when present; if the backend
                      // returned [], clear prev's contributions rather than
                      // retain dot products against the previous top-1.
                      toothContributions: r.tooth_contributions ?? [],
                      // Fragment search emits its own dropped:[] (always empty
                      // since user-chosen subsets don't dedup). n_dropped from
                      // the parent /identify run is also stale — those
                      // duplicates were on the full set, not on this user-
                      // chosen subset. Reset both together so the header
                      // doesn't read "Queried with 4 teeth · 2 duplicates
                      // dropped" when the 4 chosen teeth had no dedup at all.
                      nDropped: r.n_dropped ?? 0,
                      dropReasons: r.dropped ?? [],
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
  // Backend emits "detect" / "segment" for panoramic mode and "validate"
  // for crops mode — all three occupy the same first-stage slot in the UI.
  if (name === "detect" || name === "segment" || name === "validate") return "stageA";
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
        // Reset the live FDI list when embed re-starts so a re-run doesn't
        // show stale teeth from the previous query.
        embeddedTeeth: stage === "embed" ? [] : prev.embeddedTeeth,
      }));
      return;
    }
    case "progress": {
      if (evt.data.stage !== "embed") return;
      const newlyEmbedded = evt.data.embedded ?? [];
      setPipeline((prev) => ({
        ...prev,
        embedProgress: {
          current: evt.data.current,
          total: evt.data.total,
        },
        // Append the slice of teeth just embedded. The backend batches ~4
        // per progress event; concatenate to build the running history,
        // slicing to `current` so we never exceed the announced count even
        // if the SSE replays a previously seen batch.
        embeddedTeeth: [
          ...(prev.embeddedTeeth ?? []),
          ...newlyEmbedded,
        ].slice(0, evt.data.current),
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
        // currentImageUrl stays pinned to the user's just-uploaded preview.
        // Overlay outlines + FDI labels arrive as JSON on the FDI stage and
        // get layered on top by ImageWithOverlays — no PNG fetch on the hot
        // path, so visuals arrive instantly when the stage event lands.
        toothOverlays:
          rawStage === "fdi" && Array.isArray(evt.data.tooth_overlays)
            ? evt.data.tooth_overlays
            : prev.toothOverlays,
        imageWidth:
          rawStage === "fdi" && typeof evt.data.image_width === "number"
            ? evt.data.image_width
            : prev.imageWidth,
        imageHeight:
          rawStage === "fdi" && typeof evt.data.image_height === "number"
            ? evt.data.image_height
            : prev.imageHeight,
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
          // Backend search-stage payload carries a structured list. Default
          // to [] so the UI's narrative ("0 dropped" path) doesn't tip into
          // "unknown" just because the payload omits it.
          dropReasons: evt.data.dropped ?? [],
          toothContributions: evt.data.tooth_contributions,
          selectedPersonId: selected?.person_id,
          selectedFakeName: selected?.fake_name,
          // Calibrated open-set + provenance from the backend.
          openSetScore: evt.data.open_set_score ?? null,
          openSetDecision: evt.data.open_set_decision ?? "unknown",
          openSetThreshold: evt.data.open_set_threshold ?? null,
          queryProvenance: evt.data.query_provenance ?? "unknown",
          expectedPersonId: evt.data.expected_person_id ?? null,
          expectedMatch: evt.data.expected_match ?? null,
          ageEstimate: evt.data.age_estimate ?? null,
          queryId: evt.data.query_id ?? null,
          perTooth: perTooth ?? [],
          // Backend sets crops_mode=true on the search event for
          // /api/identify-crops; the results header flips its copy.
          cropsMode: evt.data.crops_mode ?? false,
          // Per-input-crop outcomes (auto-FDI label, OOD, dup). Only
          // populated on the crops path; absent otherwise.
          perCrop: evt.data.per_crop,
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
