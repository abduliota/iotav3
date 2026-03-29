/**
 * api.ts — typed wrappers around the FastAPI backend
 */
import { StreamChunk, Source, SystemStats, Document, Conversation } from '@/types'

const BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

// ── Stream query ──────────────────────────────────────────────────────────────
export async function* streamQuery(params: {
  query:     string
  userId:    string
  sessionId: string
  topK?:     number
}): AsyncGenerator<StreamChunk> {
  const url = `${BASE}/api/query-stream`
  const res = await fetch(url, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query:      params.query,
      user_id:    params.userId,
      session_id: params.sessionId,
      top_k:      params.topK,
    }),
  })
  if (!res.ok) {
    throw new Error(`API error ${res.status}: ${await res.text()}`)
  }
  const reader  = res.body!.getReader()
  const decoder = new TextDecoder()
  let   buffer  = ''
  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''
    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed) continue
      try {
        yield JSON.parse(trimmed) as StreamChunk
      } catch {
        // malformed line — skip
      }
    }
  }
  if (buffer.trim()) {
    try { yield JSON.parse(buffer.trim()) as StreamChunk } catch {}
  }
}

// ── Feedback ──────────────────────────────────────────────────────────────────
export async function submitFeedback(params: {
  sessionId:         string
  userId:            string
  messageId:         string
  feedback:          1 | 0
  comments?:         string
  userMessage?:      string
  assistantMessage?: string
}): Promise<void> {
  await fetch(`${BASE}/api/feedback`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id:        params.sessionId,
      user_id:           params.userId,
      message_id:        params.messageId,
      feedback:          params.feedback,
      comments:          params.comments,
      user_message:      params.userMessage,
      assistant_message: params.assistantMessage,
    }),
  })
}

// ── Session history ───────────────────────────────────────────────────────────
export async function fetchSessionMessages(sessionId: string): Promise<Array<{
  message_id:        string
  user_message:      string
  assistant_message: string
  timestamp:         string
}>> {
  const res = await fetch(`${BASE}/api/session/${sessionId}/messages`)
  if (!res.ok) return []
  const data = await res.json()
  return data.messages ?? []
}

// ── System stats ──────────────────────────────────────────────────────────────
export async function fetchStats(): Promise<SystemStats | null> {
  try {
    const res = await fetch(`${BASE}/admin/stats`)
    if (!res.ok) return null
    return await res.json()
  } catch {
    return null
  }
}

// ── Documents list ────────────────────────────────────────────────────────────
export async function fetchDocuments(search = '', limit = 20): Promise<{
  documents: Document[]
  total:     number
}> {
  try {
    const params = new URLSearchParams()
    if (search) params.set('search', search)
    params.set('limit', String(limit))
    const res = await fetch(`${BASE}/api/documents?${params}`)
    if (!res.ok) return { documents: [], total: 0 }
    return await res.json()
  } catch {
    return { documents: [], total: 0 }
  }
}

// ── Conversations list ────────────────────────────────────────────────────────
export async function fetchConversations(userId: string): Promise<Conversation[]> {
  if (!userId) return []
  try {
    const res = await fetch(`${BASE}/api/conversations?user_id=${encodeURIComponent(userId)}`)
    if (!res.ok) return []
    const data = await res.json()
    return data.conversations ?? []
  } catch {
    return []
  }
}