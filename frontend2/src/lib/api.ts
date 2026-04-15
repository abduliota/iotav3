export type BackendSource = {
  document_name: string;
  page_start?: number | null;
  page_end?: number | null;
  section_title?: string | null;
  similarity?: number;
  snippet?: string | null;
};

export type QueryRequestPayload = {
  query: string;
  top_k?: number;
  user_id?: string;
  session_id?: string;
  debug?: boolean;
};

export type QueryResponsePayload = {
  answer: string;
  sources: BackendSource[];
  cached: boolean;
  method?: string;
  message_id?: string;
  candidate_count?: number;
  reranker_top_score?: number;
};

export type DocumentInfo = {
  document_name: string;
  source_type: string;
  total_pages: number | string;
  chunk_count: number;
};

export type DocumentsResponsePayload = {
  documents: DocumentInfo[];
  total: number;
};

export type ConversationSummary = {
  session_id: string;
  title: string;
  last_message_at: string;
  message_count: number;
};

export type SessionMessage = {
  message_id: string;
  user_message: string;
  assistant_message: string;
  timestamp: string;
};

type ConversationsResponsePayload = {
  conversations: ConversationSummary[];
};

type SessionMessagesResponsePayload = {
  session_id: string;
  messages: SessionMessage[];
};

export type FeedbackPayload = {
  session_id: string;
  user_id: string;
  message_id: string;
  feedback: 0 | 1;
  comments?: string;
  user_message?: string;
  assistant_message?: string;
};

export type SystemStats = {
  api_status: string;
  docs_ingested: number;
  total_chunks: number;
  cached_answers: number;
  cache_hit_rate_pct: number;
  avg_response_ms: number;
  model: string;
};

export type CacheStatus = {
  backend?: string;
  connected?: boolean;
  cached_entries?: number;
  ttl_seconds?: number;
  ttl_days?: number | string;
  note?: string;
  error?: string;
};

export type FetchDocumentsOptions = {
  search?: string;
  limit?: number;
};

const parseJsonSafe = async <T>(res: Response): Promise<T | null> => {
  try {
    return (await res.json()) as T;
  } catch {
    return null;
  }
};

const readError = async (res: Response, fallback: string) => {
  const body = await parseJsonSafe<{ detail?: string }>(res);
  return body?.detail || fallback;
};

export const createApiClient = (apiBase: string) => {
  const base = apiBase.replace(/\/$/, "");
  const withBase = (path: string) => `${base}${path}`;

  return {
    async fetchHealth() {
      const res = await fetch(withBase("/health"), { cache: "no-store" });
      if (!res.ok) throw new Error(await readError(res, "Health check failed"));
      const body = await parseJsonSafe<{ status?: string; version?: string }>(res);
      return body ?? {};
    },

    async fetchSystemStats() {
      const res = await fetch(withBase("/admin/stats"), { cache: "no-store" });
      if (!res.ok) throw new Error(await readError(res, "Failed to load system stats"));
      const body = await parseJsonSafe<SystemStats>(res);
      if (!body) throw new Error("Invalid stats response");
      return body;
    },

    async fetchCacheStatus() {
      const res = await fetch(withBase("/admin/cache/status"), {
        cache: "no-store",
      });
      if (!res.ok) throw new Error(await readError(res, "Failed to load cache status"));
      const body = await parseJsonSafe<CacheStatus>(res);
      return body ?? {};
    },

    async fetchDocuments(options?: FetchDocumentsOptions) {
      const params = new URLSearchParams();
      if (options?.search) params.set("search", options.search);
      if (options?.limit) params.set("limit", String(options.limit));
      const query = params.toString();
      const endpoint = query
        ? `${withBase("/api/documents")}?${query}`
        : withBase("/api/documents");
      const res = await fetch(endpoint);
      if (!res.ok) throw new Error(await readError(res, "Failed to load documents"));
      const body = await parseJsonSafe<DocumentsResponsePayload>(res);
      return body?.documents ?? [];
    },

    async queryAnswer(payload: QueryRequestPayload) {
      const res = await fetch(withBase("/api/query"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(await readError(res, "Query failed"));
      const body = await parseJsonSafe<QueryResponsePayload>(res);
      if (!body) throw new Error("Invalid query response");
      return body;
    },

    async fetchConversations(userId: string) {
      if (!userId) return [];
      const endpoint = `${withBase("/api/conversations")}?user_id=${encodeURIComponent(userId)}`;
      const res = await fetch(endpoint);
      if (!res.ok) throw new Error(await readError(res, "Failed to load conversations"));
      const body = await parseJsonSafe<ConversationsResponsePayload>(res);
      return body?.conversations ?? [];
    },

    async fetchSessionMessages(sessionId: string) {
      const res = await fetch(
        withBase(`/api/session/${encodeURIComponent(sessionId)}/messages`)
      );
      if (!res.ok) throw new Error(await readError(res, "Failed to load session messages"));
      const body = await parseJsonSafe<SessionMessagesResponsePayload>(res);
      return body?.messages ?? [];
    },

    async submitFeedback(payload: FeedbackPayload) {
      const res = await fetch(withBase("/api/feedback"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(await readError(res, "Failed to submit feedback"));
      return parseJsonSafe<{ status?: string; feedback?: number }>(res);
    },
  };
};
