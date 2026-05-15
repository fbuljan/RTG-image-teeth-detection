"use client";

import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";

export type ToastLevel = "info" | "warning" | "error";

export type Toast = {
  id: number;
  level: ToastLevel;
  title?: string;
  message: string;
};

type ToastInput = Omit<Toast, "id">;

type ToastContextValue = {
  push: (toast: ToastInput) => void;
  clear: () => void;
};

const ToastContext = createContext<ToastContextValue | null>(null);

export function useToasts(): ToastContextValue {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error("useToasts must be used within <ToastProvider>");
  return ctx;
}

const AUTO_DISMISS_MS: Record<ToastLevel, number | null> = {
  info: 5000,
  warning: 8000,
  error: null, // sticky — user must close
};

const LEVEL_CLASS: Record<ToastLevel, string> = {
  info: "border-slate-300 bg-white text-slate-800 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100",
  warning: "border-amber-300 bg-amber-50 text-amber-900 dark:border-amber-700 dark:bg-amber-950 dark:text-amber-100",
  error: "border-rose-300 bg-rose-50 text-rose-900 dark:border-rose-700 dark:bg-rose-950 dark:text-rose-100",
};

const LEVEL_LABEL: Record<ToastLevel, string> = {
  info: "Info",
  warning: "Warning",
  error: "Error",
};

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const nextId = useRef(1);

  const dismiss = useCallback((id: number) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const push = useCallback((toast: ToastInput) => {
    const id = nextId.current++;
    setToasts((prev) => [...prev, { ...toast, id }]);
    const ttl = AUTO_DISMISS_MS[toast.level];
    if (ttl !== null) {
      setTimeout(() => dismiss(id), ttl);
    }
  }, [dismiss]);

  const clear = useCallback(() => setToasts([]), []);

  return (
    <ToastContext.Provider value={{ push, clear }}>
      {children}
      <ToastViewport toasts={toasts} onDismiss={dismiss} />
    </ToastContext.Provider>
  );
}

function ToastViewport({ toasts, onDismiss }: { toasts: Toast[]; onDismiss: (id: number) => void }) {
  return (
    <div
      aria-live="polite"
      aria-atomic="true"
      className="pointer-events-none fixed inset-x-0 top-4 z-50 flex flex-col items-center gap-2 px-4 sm:items-end sm:right-4 sm:left-auto sm:max-w-md"
    >
      {toasts.map((t) => (
        <ToastCard key={t.id} toast={t} onDismiss={() => onDismiss(t.id)} />
      ))}
    </div>
  );
}

function ToastCard({ toast, onDismiss }: { toast: Toast; onDismiss: () => void }) {
  // Re-mount animation hook — slide in on appear.
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const t = requestAnimationFrame(() => setVisible(true));
    return () => cancelAnimationFrame(t);
  }, []);

  return (
    <div
      role={toast.level === "error" ? "alert" : "status"}
      className={`pointer-events-auto w-full rounded-xl border px-4 py-3 shadow-lg transition-all duration-200 ${
        LEVEL_CLASS[toast.level]
      } ${visible ? "translate-y-0 opacity-100" : "-translate-y-2 opacity-0"}`}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="text-xs font-semibold uppercase tracking-wide opacity-70">
            {toast.title ?? LEVEL_LABEL[toast.level]}
          </div>
          <p className="mt-1 text-sm leading-snug">{toast.message}</p>
        </div>
        <button
          type="button"
          onClick={onDismiss}
          className="ml-2 shrink-0 rounded-md px-2 py-1 text-xs font-medium hover:bg-black/5 dark:hover:bg-white/10"
          aria-label="Dismiss"
        >
          ✕
        </button>
      </div>
    </div>
  );
}
