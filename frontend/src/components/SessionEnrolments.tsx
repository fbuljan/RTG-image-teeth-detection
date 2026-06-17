"use client";

import { useCallback, useEffect, useState } from "react";

import {
  clearSessionEnrolments,
  deleteEnrolment,
  fetchEnrolments,
  type SessionEnrolment,
} from "@/lib/api";

// Phase 9.7 — "Your enrolments (this session)" panel.
//
// Lists the caller's session enrolments and exposes per-row delete + a
// "Clear all" button. Polled on mount and whenever the parent bumps the
// `refreshNonce`. Shows the 24h-scope disclaimer prominently so a thesis
// viewer doesn't mistake this for persistent storage.

export function SessionEnrolments({
  sessionId,
  refreshNonce,
  onChanged,
}: {
  sessionId: string;
  refreshNonce: number;
  onChanged: () => void;
}) {
  const [items, setItems] = useState<SessionEnrolment[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await fetchEnrolments(sessionId);
      setItems(r.persons);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [sessionId]);

  useEffect(() => {
    refresh();
  }, [refresh, refreshNonce]);

  const onDelete = useCallback(
    async (personId: string) => {
      try {
        await deleteEnrolment(sessionId, personId);
        onChanged();
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    },
    [sessionId, onChanged],
  );

  const onClear = useCallback(async () => {
    if (items.length === 0) return;
    if (!confirm(`Remove all ${items.length} session enrolment(s)?`)) return;
    try {
      await clearSessionEnrolments(sessionId);
      onChanged();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [items.length, sessionId, onChanged]);

  if (items.length === 0 && !loading && !error) {
    // No enrolments yet — don't render anything; the modal trigger is in the header.
    return null;
  }

  return (
    <section className="rounded-2xl border border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-slate-900">
      <header className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-200 px-6 py-3 dark:border-slate-800">
        <div>
          <h2 className="text-base font-semibold">
            Your enrolments
            <span className="ml-2 inline-flex items-center rounded-full bg-amber-500/15 px-2 py-0.5 text-xs font-medium text-amber-700 ring-1 ring-inset ring-amber-500/30 dark:text-amber-300">
              session · 24h
            </span>
          </h2>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            Scoped to this browser. Canonical registry is read-only.
          </p>
        </div>
        {items.length > 0 && (
          <button
            type="button"
            onClick={onClear}
            className="rounded-md border border-rose-300 px-2.5 py-1 text-xs font-medium text-rose-700 hover:bg-rose-50 dark:border-rose-800/60 dark:text-rose-300 dark:hover:bg-rose-950/30"
          >
            Clear all
          </button>
        )}
      </header>

      {error && (
        <p className="px-6 py-3 text-sm text-rose-700 dark:text-rose-300">{error}</p>
      )}

      <ul className="divide-y divide-slate-100 dark:divide-slate-800">
        {items.map((p) => (
          <li key={p.person_id} className="flex items-center justify-between gap-4 px-6 py-2.5 text-sm">
            <div>
              <p className="font-medium">{p.fake_name}</p>
              <p className="text-xs text-slate-500">
                {p.n_teeth} teeth · {new Date(p.enrolled_at * 1000).toLocaleTimeString()}
              </p>
            </div>
            <button
              type="button"
              onClick={() => onDelete(p.person_id)}
              aria-label={`Delete ${p.fake_name}`}
              className="rounded-md border border-slate-300 px-2 py-1 text-xs text-slate-600 hover:bg-slate-50 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800"
            >
              Delete
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
