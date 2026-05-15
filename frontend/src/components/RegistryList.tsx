"use client";

import { forwardRef, useEffect, useImperativeHandle, useMemo, useRef, useState } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import {
  fetchRegistry,
  panoramicDownloadUrl,
  type RegistryPerson,
} from "@/lib/api";
import { useToasts } from "@/components/Toaster";

type Props = {
  selectedPersonId?: string;
  onSelect: (person: RegistryPerson) => void;
};

export type RegistryListHandle = {
  scrollIntoView: () => void;
};

const ROW_HEIGHT = 64;

export const RegistryList = forwardRef<RegistryListHandle, Props>(function RegistryList(
  { selectedPersonId, onSelect },
  ref,
) {
  const [persons, setPersons] = useState<RegistryPerson[]>([]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const downloadRef = useRef<HTMLAnchorElement | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const sectionRef = useRef<HTMLElement | null>(null);
  const toasts = useToasts();

  useImperativeHandle(ref, () => ({
    scrollIntoView: () => {
      sectionRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    },
  }));

  useEffect(() => {
    let cancelled = false;
    fetchRegistry()
      .then((res) => {
        if (cancelled) return;
        setPersons(res.persons);
      })
      .catch((err) => {
        if (cancelled) return;
        const msg = err instanceof Error ? err.message : String(err);
        setError(msg);
        toasts.push({
          level: "error",
          title: "Couldn't load registry",
          message: `${msg}. Is the backend running on :8000?`,
        });
      })
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [toasts]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return persons;
    return persons.filter((p) => p.fake_name.toLowerCase().includes(q));
  }, [persons, query]);

  const virtualizer = useVirtualizer({
    count: filtered.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => ROW_HEIGHT,
    overscan: 8,
  });

  function handleDownload(person: RegistryPerson) {
    onSelect(person);
    const a = downloadRef.current ?? document.createElement("a");
    a.href = panoramicDownloadUrl(person.person_id);
    a.download = "xray.png";
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  return (
    <section
      ref={sectionRef}
      className="rounded-2xl border border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-slate-900"
    >
      <header className="flex flex-col gap-3 border-b border-slate-200 px-6 py-4 dark:border-slate-800 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-lg font-semibold">Registry</h2>
          <p className="text-sm text-slate-500 dark:text-slate-400">
            {loading
              ? "Loading enrolled persons…"
              : `${persons.length.toLocaleString()} enrolled persons${
                  filtered.length !== persons.length
                    ? ` · ${filtered.length.toLocaleString()} match`
                    : ""
                }`}
          </p>
        </div>
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search by name…"
          className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm shadow-inner focus:border-slate-500 focus:outline-none focus:ring-1 focus:ring-slate-500 dark:border-slate-700 dark:bg-slate-950 sm:w-64"
        />
      </header>

      {error && (
        <div className="px-6 py-4 text-sm text-rose-600 dark:text-rose-400">
          Failed to load registry: {error}
        </div>
      )}

      <div ref={scrollRef} className="h-[420px] overflow-y-auto">
        {loading ? (
          <div className="px-6 py-10 text-center text-sm text-slate-500 dark:text-slate-400">
            Loading…
          </div>
        ) : filtered.length === 0 ? (
          <div className="px-6 py-10 text-center text-sm text-slate-500 dark:text-slate-400">
            No persons match &quot;{query}&quot;.
          </div>
        ) : (
          <div
            style={{ height: virtualizer.getTotalSize(), position: "relative" }}
          >
            {virtualizer.getVirtualItems().map((vrow) => {
              const person = filtered[vrow.index];
              const isSelected = person.person_id === selectedPersonId;
              return (
                <div
                  key={person.person_id}
                  ref={virtualizer.measureElement}
                  data-index={vrow.index}
                  style={{
                    position: "absolute",
                    top: 0,
                    left: 0,
                    right: 0,
                    transform: `translateY(${vrow.start}px)`,
                  }}
                  className={`flex items-center justify-between gap-4 border-b border-slate-100 px-6 py-3 dark:border-slate-800 ${
                    isSelected
                      ? "bg-amber-50 dark:bg-amber-900/30"
                      : "hover:bg-slate-50 dark:hover:bg-slate-800/40"
                  }`}
                >
                  <div className="flex min-w-0 flex-col">
                    <span className="truncate text-base font-medium">{person.fake_name}</span>
                    <span className="text-xs text-slate-500 dark:text-slate-400">
                      {person.n_teeth} teeth · ID {person.person_id.slice(-8)}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    {isSelected && (
                      <span className="rounded-full bg-amber-500/20 px-2 py-0.5 text-xs font-medium uppercase tracking-wide text-amber-800 dark:text-amber-200">
                        Selected
                      </span>
                    )}
                    <button
                      type="button"
                      onClick={() => handleDownload(person)}
                      className="shrink-0 rounded-lg bg-slate-900 px-3 py-1.5 text-sm font-medium text-white shadow hover:bg-slate-800 dark:bg-slate-100 dark:text-slate-900 dark:hover:bg-white"
                    >
                      Download X-ray
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      <a ref={downloadRef} className="hidden" aria-hidden="true" />
    </section>
  );
});
