export interface Source {
  document_name: string
  page_start:    number | null
  page_end:      number | null
  section_title: string | null
  similarity:    number
  snippet:       string | null
}

export interface Message {
  id:          string
  role:        'user' | 'assistant'
  content:     string
  sources:     Source[]
  timestamp:   Date
  isStreaming?: boolean
  method?:     string
  cached?:     boolean
  feedback?:   1 | 0 | null
}

export interface StreamChunk {
  type:        'token' | 'sources' | 'done' | 'error'
  text?:       string
  sources?:    Source[]
  message_id?: string
  cached?:     boolean
  method?:     string
  message?:    string
}

// ── New types for dashboard ───────────────────────────────────────────────────

export interface SystemStats {
  api_status:         string
  docs_ingested:      number
  total_chunks:       number
  cached_answers:     number
  cache_hit_rate_pct: number
  avg_response_ms:    number
  model:              string
}

export interface Document {
  document_name: string
  source_type:   string
  total_pages:   string | number
  chunk_count:   number
}

export interface Conversation {
  session_id:      string
  title:           string
  last_message_at: string
  message_count:   number
}