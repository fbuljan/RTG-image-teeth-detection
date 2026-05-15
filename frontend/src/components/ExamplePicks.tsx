"use client";

import { useEffect, useState } from "react";

import { API_BASE, panoramicDownloadUrl, type RegistryPerson } from "@/lib/api";

type Props = {
  onSelect: (person: RegistryPerson) => void;
};

export function ExamplePicks({ onSelect }: Props) {
  const [examples, setExamples] = useState<RegistryPerson[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch(`${API_BASE}/api/registry/examples`, { cache: "no-store" })
      .then(async (res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return (await res.json()) as { examples: RegistryPerson[] };
      })
      .then((data) => {
        if (!cancelled) setExamples(data.examples);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (error || examples.length === 0) return null;

  function handlePick(person: RegistryPerson) {
    onSelect(person);
    const a = document.createElement("a");
    a.href = panoramicDownloadUrl(person.person_id);
    a.download = "xray.png";
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  return (
    <section className="rounded-2xl border border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-slate-900">
      <header className="border-b border-slate-200 px-6 py-4 dark:border-slate-800">
        <h2 className="text-lg font-semibold">Quick picks</h2>
        <p className="text-sm text-slate-500 dark:text-slate-400">
          Demo-ready X-rays — click any chip to download and start a query.
        </p>
      </header>
      <div className="flex flex-wrap gap-2 px-6 py-4">
        {examples.map((person) => (
          <button
            key={person.person_id}
            type="button"
            onClick={() => handlePick(person)}
            className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1.5 text-sm font-medium text-slate-700 hover:border-amber-400 hover:bg-amber-50 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-200 dark:hover:border-amber-500 dark:hover:bg-amber-900/30"
            title={`${person.n_teeth} teeth`}
          >
            {person.fake_name}
          </button>
        ))}
      </div>
    </section>
  );
}
