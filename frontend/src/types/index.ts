export interface Source {
  document_name: string
  page_start:    number | null
  page_end:      number | null
  section_title: string | null
  similarity:    number
  snippet:       string | null
}

export interface Message {
  id:         string
  role:       'user' | 'assistant'
  content:    string
  sources:    Source[]
  timestamp:  Date
  isStreaming?: boolean
  method?:    string
  cached?:    boolean
  feedback?:  1 | 0 | null   // 1=like, 0=dislike, null=no vote
}

export interface StreamChunk {
  type:       'token' | 'sources' | 'done' | 'error'
  text?:      string
  sources?:   Source[]
  message_id?: string
  cached?:    boolean
  method?:    string
  message?:   string
}
