/**
 * identity.ts
 * Generates a stable user_id via FingerprintJS and persists it in localStorage.
 * Also manages session_id lifecycle (new session per page load, persisted across refreshes).
 */
import { v4 as uuidv4 } from 'uuid'

const USER_KEY    = 'nora_user_id'
const SESSION_KEY = 'nora_session_id'

/**
 * Get or create a stable user_id.
 * Uses FingerprintJS visitorId as the source of truth,
 * falls back to a localStorage UUID if FingerprintJS fails.
 */
export async function getUserId(): Promise<string> {
  // Return cached value immediately if available
  if (typeof window === 'undefined') return 'server'
  const cached = localStorage.getItem(USER_KEY)
  if (cached) return cached

  try {
    const FingerprintJS = await import('@fingerprintjs/fingerprintjs')
    const fp     = await FingerprintJS.default.load()
    const result = await fp.get()
    const id     = result.visitorId
    localStorage.setItem(USER_KEY, id)
    return id
  } catch {
    // Fallback: stable UUID in localStorage
    const fallback = uuidv4()
    localStorage.setItem(USER_KEY, fallback)
    return fallback
  }
}

/**
 * Get or create the current session_id.
 * Sessions persist across page refreshes (stored in sessionStorage).
 * A new session is created if sessionStorage is cleared or this is a fresh tab.
 */
export function getSessionId(): string {
  if (typeof window === 'undefined') return uuidv4()

  // Use sessionStorage so session resets on new tab but persists on refresh
  let id = sessionStorage.getItem(SESSION_KEY)
  if (!id) {
    id = uuidv4()
    sessionStorage.setItem(SESSION_KEY, id)
  }
  return id
}

/** Force a new session (called by "New conversation" button) */
export function newSession(): string {
  const id = uuidv4()
  if (typeof window !== 'undefined') {
    sessionStorage.setItem(SESSION_KEY, id)
  }
  return id
}

/** Clear stored identity — next call to getUserId() will generate a new one */
export function resetIdentity(): void {
  if (typeof window === 'undefined') return
  localStorage.removeItem('nora_user_id')
  sessionStorage.removeItem('nora_session_id')
}