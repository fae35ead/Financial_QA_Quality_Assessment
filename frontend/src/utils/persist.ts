const STORAGE_PREFIX = "qa.frontend.v1";

export const STORAGE_KEYS = {
  analysisPage: `${STORAGE_PREFIX}.analysis_page`,
  batchPage: `${STORAGE_PREFIX}.batch_page`,
} as const;

function canUseStorage() {
  return typeof window !== "undefined" && typeof window.localStorage !== "undefined";
}

export function readLocalState<T>(key: string, fallback: T): T {
  if (!canUseStorage()) {
    return fallback;
  }
  try {
    const raw = window.localStorage.getItem(key);
    if (!raw) {
      return fallback;
    }
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

export function writeLocalState<T>(key: string, value: T) {
  if (!canUseStorage()) {
    return;
  }
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // ignore storage quota/serialization issues
  }
}

export function removeLocalState(key: string) {
  if (!canUseStorage()) {
    return;
  }
  try {
    window.localStorage.removeItem(key);
  } catch {
    // ignore
  }
}
