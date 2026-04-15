const USER_KEY = "iota_user_id";
const SESSION_KEY = "iota_session_id";

export async function getOrCreateUserId(): Promise<string> {
  if (typeof window === "undefined") return "server";

  const cached = localStorage.getItem(USER_KEY);
  if (cached) return cached;

  try {
    const FingerprintJS = await import("@fingerprintjs/fingerprintjs");
    const fp = await FingerprintJS.load();
    const result = await fp.get();
    const id = result.visitorId;
    localStorage.setItem(USER_KEY, id);
    return id;
  } catch {
    const fallback = crypto.randomUUID();
    localStorage.setItem(USER_KEY, fallback);
    return fallback;
  }
}

export function getOrCreateSessionId(): string {
  if (typeof window === "undefined") return crypto.randomUUID();

  const cached = localStorage.getItem(SESSION_KEY);
  if (cached) return cached;

  const generated = crypto.randomUUID();
  localStorage.setItem(SESSION_KEY, generated);
  return generated;
}

export function newSessionId(): string {
  const id = crypto.randomUUID();
  if (typeof window !== "undefined") {
    localStorage.setItem(SESSION_KEY, id);
  }
  return id;
}

export function resetIdentity(): void {
  if (typeof window === "undefined") return;
  localStorage.removeItem(USER_KEY);
  localStorage.removeItem(SESSION_KEY);
}
