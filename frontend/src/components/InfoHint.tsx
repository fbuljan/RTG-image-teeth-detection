"use client";

import { useState } from "react";

type Props = {
  text: string;
  className?: string;
};

/**
 * Inline "?" icon that shows a small tooltip on hover/focus. Replaces native
 * browser tooltips which are slow to appear and visually inconsistent.
 */
export function InfoHint({ text, className = "" }: Props) {
  const [open, setOpen] = useState(false);
  return (
    <span className={`relative inline-flex items-center ${className}`}>
      <button
        type="button"
        aria-label="More info"
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        className="ml-1 inline-flex h-4 w-4 items-center justify-center rounded-full border border-slate-300 text-[10px] font-semibold text-slate-500 hover:border-slate-500 hover:text-slate-700 dark:border-slate-600 dark:text-slate-400 dark:hover:border-slate-400 dark:hover:text-slate-200"
      >
        ?
      </button>
      {open && (
        <span
          role="tooltip"
          className="absolute bottom-full left-1/2 z-30 mb-2 w-72 -translate-x-1/2 whitespace-pre-line rounded-md border border-slate-200 bg-white px-3 py-2 text-left text-xs font-normal normal-case leading-relaxed tracking-normal text-slate-700 shadow-lg dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200"
        >
          {text}
        </span>
      )}
    </span>
  );
}
