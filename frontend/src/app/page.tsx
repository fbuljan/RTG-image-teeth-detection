"use client";

import { useCallback, useRef, useState } from "react";

import { ExamplePicks } from "@/components/ExamplePicks";
import { ModelCard } from "@/components/ModelCard";
import { PipelineProgress, type PipelineState } from "@/components/PipelineProgress";
import { RegistryList, type RegistryListHandle } from "@/components/RegistryList";
import { ResultsCards, type ResultsState } from "@/components/ResultsCards";
import { useToasts } from "@/components/Toaster";
import { UploadZone } from "@/components/UploadZone";
import { type RegistryPerson, type StageEvent } from "@/lib/api";
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
  const registryRef = useRef<RegistryListHandle | null>(null);
  const pipelineRef = useRef<HTMLDivElement | null>(null);
  const resultsRef = useRef<HTMLDivElement | null>(null);
  const toasts = useToasts();

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
        for await (const evt of streamIdentify(file, { mode })) {
          applyEvent(evt, setPipeline, setResults, selected);
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
    [selected, toasts, mode],
  );

  return (
    <main className="mx-auto flex w-full max-w-5xl flex-col gap-6 px-4 py-10 sm:px-6">
      <header className="space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">
          Dental Identification Demo
        </h1>
        <p className="max-w-3xl text-sm text-slate-600 dark:text-slate-300">
          Pick a person from the registry, download their panoramic X-ray as
          <code className="mx-1 rounded bg-slate-100 px-1 py-0.5 text-xs dark:bg-slate-800">
            xray.png
          </code>
          , then drop it back into the upload zone. The system will detect
          teeth, number them, embed each one, and search a registry of 1,178
          enrolled persons for the closest match.
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
          />
        </div>
      )}

      <ModelCard />
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
