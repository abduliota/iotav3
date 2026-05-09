import type { FormEvent } from "react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./App.css";
import {
  type CacheStatus,
  createApiClient,
  type BackendSource,
  type ConversationSummary,
  type DocumentInfo,
  type SystemStats,
} from "./lib/api";
import {
  getOrCreateSessionId,
  getOrCreateUserId,
  newSessionId,
  resetIdentity,
} from "./lib/identity";

type Role = "user" | "assistant";

type ChatMessage = {
  id: string;
  role: Role;
  content: string;
  sources?: string[];
  contexts?: SourceContext[];
  question?: string;
  backendMessageId?: string;
  feedback?: 0 | 1;
};

type SourceContext = {
  source: string;
  page?: number | null;
  content: string;
  similarity?: number;
  sectionTitle?: string | null;
};

type ApiStatus = "live" | "building";

const API_BASE =
  (import.meta.env.VITE_API_BASE as string | undefined) ??
  "/backend";
const DEFAULT_TOP_K = 5;
const formatModelLabel = (model?: string) => {
  if (!model) return "Model unavailable";
  const normalized = model.toLowerCase();
  if (normalized === "gpt-4o-mini") return "GPT-4o-Mini";
  return model
    .replace(/[-_]/g, " ")
    .split(" ")
    .filter(Boolean)
    .map((part) =>
      part.length <= 3
        ? part.toUpperCase()
        : part.charAt(0).toUpperCase() + part.slice(1)
    )
    .join("-");
};
const formatMetric = (value: number | null | undefined) =>
  typeof value === "number" && Number.isFinite(value)
    ? value.toLocaleString()
    : "--";

function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState("");
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [documentsLoading, setDocumentsLoading] = useState(false);
  const [documentsError, setDocumentsError] = useState("");
  const [documentSearch, setDocumentSearch] = useState("");
  const [debouncedDocumentSearch, setDebouncedDocumentSearch] = useState("");
  const [systemStats, setSystemStats] = useState<SystemStats | null>(null);
  const [cacheStatus, setCacheStatus] = useState<CacheStatus | null>(null);
  const [statsLoading, setStatsLoading] = useState(false);
  const [statsError, setStatsError] = useState("");
  const [apiStatus, setApiStatus] = useState<ApiStatus>("building");
  const [conversationLoading, setConversationLoading] = useState(false);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string>("");
  const [userId, setUserId] = useState("");
  const [identityLoading, setIdentityLoading] = useState(true);
  const [showIdentity, setShowIdentity] = useState(false);
  const [feedbackError, setFeedbackError] = useState("");
  const [feedbackLoadingByMessage, setFeedbackLoadingByMessage] = useState<
    Record<string, boolean>
  >({});
  const [messageTabs, setMessageTabs] = useState<
    Record<string, "answer" | "sources">
  >({});
  const [selectedContexts, setSelectedContexts] = useState<
    Record<string, number>
  >({});
  const [selectedLatestContextIndex, setSelectedLatestContextIndex] = useState(0);
  const [documentFilter, setDocumentFilter] = useState<string>("All");

  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const chatThreadRef = useRef<HTMLDivElement | null>(null);
  const identityRef = useRef<HTMLDivElement | null>(null);

  const apiBase = useMemo(() => API_BASE.replace(/\/$/, ""), []);
  const api = useMemo(() => createApiClient(apiBase), [apiBase]);
  const shortUserId = useMemo(() => {
    if (!userId) return "Recognizing...";
    if (userId.length <= 12) return userId;
    return `${userId.slice(0, 8)}...`;
  }, [userId]);

  const escapeRegExp = (value: string) =>
    value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

  const sourceLabel = (ctx?: SourceContext, fallbackSource?: string) => {
    if (ctx?.page) return `Page ${ctx.page}`;
    if (fallbackSource) {
      const match = fallbackSource.match(/page=(\d+)/i);
      if (match?.[1]) return `Page ${match[1]}`;
    }
    return "Source";
  };

  const highlightText = (text: string, query: string) => {
    if (!query.trim()) return text;
    const terms = query
      .split(/\s+/)
      .map((t) => t.trim())
      .filter((t) => t.length > 3);
    if (terms.length === 0) return text;
    const regex = new RegExp(`(${terms.map(escapeRegExp).join("|")})`, "gi");
    return text.split(regex).map((part, idx) => {
      const isMatch = regex.test(part);
      regex.lastIndex = 0;
      return isMatch ? (
        <mark key={`${part}-${idx}`}>{part}</mark>
      ) : (
        <span key={`${part}-${idx}`}>{part}</span>
      );
    });
  };

  const mapSourceToContext = (source: BackendSource): SourceContext => ({
    source: source.document_name || "Unknown source",
    page: source.page_start ?? null,
    content:
      source.snippet?.trim() ||
      "No snippet returned for this source. Re-run query to refresh retrieval output.",
    similarity: source.similarity,
    sectionTitle: source.section_title ?? null,
  });

  const createSession = useCallback(() => {
    const nextSession = newSessionId();
    setActiveSessionId(nextSession);
    setMessages([]);
    setChatError("");
    setFeedbackError("");
  }, []);

  useEffect(() => {
    setActiveSessionId(getOrCreateSessionId());
  }, []);

  const loadConversations = useCallback(async () => {
    if (!userId) return;
    try {
      setConversationLoading(true);
      const data = await api.fetchConversations(userId);
      setConversations(data);
    } catch (err) {
      console.warn("Failed to load conversations", err);
    } finally {
      setConversationLoading(false);
    }
  }, [api, userId]);

  const loadSessionMessages = useCallback(
    async (sessionId: string) => {
      if (!sessionId) return;
      try {
        const data = await api.fetchSessionMessages(sessionId);
        if (data.length === 0) {
          setMessages([]);
          return;
        }
        const mapped: ChatMessage[] = [];
        data.forEach((item) => {
          mapped.push({
            id: `${item.message_id}-user`,
            role: "user",
            content: item.user_message,
          });
          mapped.push({
            id: `${item.message_id}-assistant`,
            role: "assistant",
            content: item.assistant_message,
            backendMessageId: item.message_id,
            question: item.user_message,
          });
        });
        setMessages(mapped);
      } catch (err) {
        console.warn("Failed to load session messages", err);
      }
    },
    [api]
  );

  const loadDocuments = useCallback(async () => {
    try {
      setDocumentsError("");
      setDocumentsLoading(true);
      const data = await api.fetchDocuments({
        search: debouncedDocumentSearch || undefined,
        limit: 50,
      });
      setDocuments(data);
    } catch (err) {
      console.warn("Failed to load documents", err);
      // FIX 2: Show friendly message instead of raw exception string
      setDocumentsError("Unable to load documents. Retrying shortly.");
    } finally {
      setDocumentsLoading(false);
    }
  }, [api, debouncedDocumentSearch]);

  const loadSystemStats = useCallback(async () => {
    try {
      setStatsError("");
      setStatsLoading(true);
      const [stats, cache] = await Promise.all([
        api.fetchSystemStats(),
        api.fetchCacheStatus().catch(() => null),
      ]);
      setSystemStats(stats);
      if (cache) setCacheStatus(cache);
    } catch (err) {
      setStatsError(
        err instanceof Error ? err.message : "System stats unavailable"
      );
    } finally {
      setStatsLoading(false);
    }
  }, [api]);

  useEffect(() => {
    let cancelled = false;

    const initializeIdentity = async () => {
      try {
        setIdentityLoading(true);
        const id = await getOrCreateUserId();
        if (!cancelled) setUserId(id);
      } catch (err) {
        console.warn("Failed to initialize fingerprint identity", err);
        if (!cancelled) setUserId(crypto.randomUUID());
      } finally {
        if (!cancelled) setIdentityLoading(false);
      }
    };

    initializeIdentity();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const id = window.setTimeout(
      () => setDebouncedDocumentSearch(documentSearch.trim()),
      300
    );
    return () => window.clearTimeout(id);
  }, [documentSearch]);

  useEffect(() => {
    loadDocuments();
    loadConversations();
    loadSystemStats();
  }, [loadConversations, loadDocuments, loadSystemStats]);

  // FIX 1: Changed polling interval from 30s (30000) to 5 minutes (300000)
  // to reduce Supabase connection pressure and prevent ConnectionTerminated errors
  useEffect(() => {
    const id = window.setInterval(() => {
      loadDocuments();
      loadConversations();
      loadSystemStats();
    }, 300000);
    return () => window.clearInterval(id);
  }, [loadConversations, loadDocuments, loadSystemStats]);

  useEffect(() => {
    if (!showIdentity) return;

    const closeOnOutsideClick = (event: MouseEvent) => {
      if (
        identityRef.current &&
        !identityRef.current.contains(event.target as Node)
      ) {
        setShowIdentity(false);
      }
    };

    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setShowIdentity(false);
    };

    document.addEventListener("mousedown", closeOnOutsideClick);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("mousedown", closeOnOutsideClick);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [showIdentity]);

  useEffect(() => {
    if (!activeSessionId) return;
    localStorage.setItem("iota_session_id", activeSessionId);
    loadSessionMessages(activeSessionId);
  }, [activeSessionId, loadSessionMessages]);

  useEffect(() => {
    let cancelled = false;

    const checkHealth = async () => {
      try {
        let timeoutId: number | undefined;
        const healthRequest = api.fetchHealth();
        const res = await Promise.race([
          healthRequest,
          new Promise((_, reject) =>
            (timeoutId = window.setTimeout(
              () => reject(new Error("Health timeout")),
              6000
            ))
          ),
        ]);
        if (timeoutId) window.clearTimeout(timeoutId);
        const body = res as { status?: string };
        const isOk =
          typeof body?.status === "string"
            ? body.status.toLowerCase() === "ok"
            : true;
        if (!cancelled) setApiStatus(isOk ? "live" : "building");
      } catch (err) {
        console.error(err);
        if (!cancelled) setApiStatus("building");
      }
    };

    checkHealth();
    const id = window.setInterval(checkHealth, 10000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [api]);

  const scrollToBottom = () => {
    if (chatThreadRef.current) {
      chatThreadRef.current.scrollTop = chatThreadRef.current.scrollHeight;
    }
  };

  const KNOWN_SOURCE_TYPES = ["SAMA", "NCA", "ISO", "SDAIA"];

  const filteredDocuments = useMemo(() => {
    if (documentFilter === "All") return documents;
    if (documentFilter === "Other") {
      return documents.filter(
        (d) => !d.source_type || !KNOWN_SOURCE_TYPES.includes(d.source_type)
      );
    }
    return documents.filter((d) => d.source_type === documentFilter);
  }, [documents, documentFilter]);

  const latestAssistant = useMemo(
    () => [...messages].reverse().find((msg) => msg.role === "assistant"),
    [messages]
  );
  const activeLatestContext = useMemo(() => {
    if (!latestAssistant?.contexts || latestAssistant.contexts.length === 0) {
      return null;
    }
    return (
      latestAssistant.contexts[selectedLatestContextIndex] ??
      latestAssistant.contexts[0]
    );
  }, [latestAssistant, selectedLatestContextIndex]);

  useEffect(() => {
    setSelectedLatestContextIndex(0);
  }, [latestAssistant?.id]);

  const sendMessage = async (event?: FormEvent) => {
    event?.preventDefault();
    setChatError("");
    setFeedbackError("");

    const trimmed = input.trim();
    if (!trimmed) {
      setChatError("Ask a question first.");
      return;
    }
    if (!userId) {
      setChatError("Identity is still initializing. Please try again.");
      return;
    }

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: trimmed,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setChatLoading(true);

    try {
      const data = await api.queryAnswer({
        query: trimmed,
        top_k: DEFAULT_TOP_K,
        user_id: userId,
        session_id: activeSessionId,
      });
      const contexts = (data.sources ?? []).map(mapSourceToContext);
      const sourceLabels = contexts.map((ctx) =>
        ctx.page ? `${ctx.source} (p.${ctx.page})` : ctx.source
      );
      const assistantMessage: ChatMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: data.answer,
        sources: sourceLabels,
        contexts,
        question: trimmed,
        backendMessageId: data.message_id,
      };
      setMessages((prev) => [...prev, assistantMessage]);
      setMessageTabs((prev) => ({ ...prev, [assistantMessage.id]: "answer" }));
      if (contexts.length > 0) {
        setSelectedContexts((prev) => ({ ...prev, [assistantMessage.id]: 0 }));
      }
      setTimeout(scrollToBottom, 50);
      loadConversations();
    } catch (err) {
      setChatError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setChatLoading(false);
    }
  };

  const submitFeedback = async (message: ChatMessage, feedback: 0 | 1) => {
    if (!message.backendMessageId) return;
    setFeedbackError("");
    setFeedbackLoadingByMessage((prev) => ({ ...prev, [message.id]: true }));
    try {
      await api.submitFeedback({
        session_id: activeSessionId,
        user_id: userId,
        message_id: message.backendMessageId,
        feedback,
        user_message: message.question,
        assistant_message: message.content,
      });
      setMessages((prev) =>
        prev.map((m) => (m.id === message.id ? { ...m, feedback } : m))
      );
    } catch (err) {
      setFeedbackError(
        err instanceof Error ? err.message : "Failed to submit feedback"
      );
    } finally {
      setFeedbackLoadingByMessage((prev) => ({ ...prev, [message.id]: false }));
    }
  };

  const handleResetIdentity = useCallback(async () => {
    try {
      resetIdentity();
      setShowIdentity(false);
      setMessages([]);
      setChatError("");
      setFeedbackError("");
      setConversations([]);
      setIdentityLoading(true);

      const regeneratedUserId = await getOrCreateUserId();
      const regeneratedSession = getOrCreateSessionId();
      setUserId(regeneratedUserId);
      setActiveSessionId(regeneratedSession);
      loadConversations();
    } catch (err) {
      console.warn("Failed to reset identity", err);
      setChatError("Unable to reset identity. Please retry.");
    } finally {
      setIdentityLoading(false);
    }
  }, [loadConversations]);

  return (
    <div className="app-bg">
      <div className="app-shell">
        <div className="left-stack">
        <aside className="rail rail-brand">
          <div className="rail-card brand-card">
            <div className="brand-card-head">
              <div className="identity-wrap" ref={identityRef}>
                <button
                  type="button"
                  title="Fingerprint identity"
                  className={`identity-btn ${identityLoading ? "identity-loading" : ""}`}
                  onClick={() => setShowIdentity((prev) => !prev)}
                >
                  <svg
                    className="identity-icon"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth={1.8}
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M12 11c0 3.517-1.009 6.799-2.753 9.571m-3.44-2.04.054-.09A13.916 13.916 0 0 0 8 11a4 4 0 1 1 8 0c0 1.017-.07 2.019-.203 3m-2.118 6.844A21.88 21.88 0 0 0 15.171 17m3.839 1.132c.645-2.266.99-4.659.99-7.132A8 8 0 0 0 8 4.07M3 15.364c.64-1.319 1-2.8 1-4.364 0-1.457.39-2.823 1.07-4"
                    />
                  </svg>
                  <span>{shortUserId}</span>
                </button>

                {showIdentity && (
                  <div className="identity-popover">
                    <div className="identity-status">
                      <span className="identity-dot" />
                      Recognized via Fingerprint
                    </div>
                    <div className="identity-card">
                      <p className="identity-card-label">Your device ID</p>
                      <p className="identity-card-value">
                        {userId || "Loading identity..."}
                      </p>
                    </div>
                    <p className="identity-note">
                      Your browser fingerprint identifies your device - no login
                      required.
                    </p>
                    <button
                      type="button"
                      className="identity-reset-btn"
                      onClick={handleResetIdentity}
                    >
                      Reset identity
                    </button>
                  </div>
                )}
              </div>
            </div>
            <div className="brand-row">
              <img
                src="/iota-logo.png"
                alt="IOTA logo"
                className="brand-logo"
              />
              <div>
                <p className="eyebrow">IOTA KSA</p>
                <h1 className="brand-title">Regulation AI</h1>
                <p className="lede">
                  AI answers with citations from SAMA rulebooks and schemes.
                </p>
              </div>
            </div>
            <div className="pill-row">
              <span
                className={`pill ${
                  apiStatus === "live" ? "pill-live" : "pill-warn"
                }`}
              >
                {apiStatus === "live"
                  ? "Live · API healthy"
                  : "Building · Rechecking"}
              </span>
              <span className="pill pill-muted">
                {formatModelLabel(systemStats?.model)}
              </span>
              <span className="pill pill-muted">PGVector</span>
              <span className="pill pill-muted">Hybrid RAG</span>
              <span
                className={`pill ${
                  cacheStatus?.connected === false ? "pill-warn" : "pill-muted"
                }`}
              >
                Redis Cache
              </span>
            </div>
            {statsError && (
              <div className="status status-error compact-status">
                {statsError}
              </div>
            )}
            <div className="stat-grid">
              <div>
                <p className="stat-label">Docs ingested</p>
                <p className="stat-value">
                  {statsLoading
                    ? "--"
                    : formatMetric(systemStats?.docs_ingested)}
                </p>
              </div>
              <div>
                <p className="stat-label">Chunks</p>
                <p className="stat-value">
                  {statsLoading
                    ? "--"
                    : formatMetric(systemStats?.total_chunks)}
                </p>
              </div>
              <div>
                <p className="stat-label">Cached answers</p>
                <p className="stat-value">
                  {statsLoading
                    ? "--"
                    : formatMetric(systemStats?.cached_answers)}
                </p>
              </div>
              <div>
                <p className="stat-label">Cache hit rate</p>
                <p className="stat-value">
                  {statsLoading || systemStats?.cache_hit_rate_pct == null
                    ? "--"
                    : `${systemStats.cache_hit_rate_pct.toLocaleString()}%`}
                </p>
              </div>
            </div>
          </div>
        </aside>

        <aside className="rail rail-secondary">
          <div className="rail-card docs-card">
            <div className="card-head">
              <div>
                <h3>Indexed documents</h3>
              </div>
            </div>
            <div className="doc-search-wrap">
              <input
                type="text"
                className="input"
                placeholder="Search indexed documents..."
                value={documentSearch}
                onChange={(e) => setDocumentSearch(e.target.value)}
              />
              {documentSearch && (
                <button
                  type="button"
                  className="btn btn-ghost btn-small"
                  onClick={() => setDocumentSearch("")}
                >
                  Clear
                </button>
              )}
            </div>
            <div className="doc-filter-row">
              {["All", "SAMA", "NCA", "ISO", "SDAIA", "Other"].map((f) => (
                <button
                  key={f}
                  type="button"
                  className={`doc-filter-btn ${documentFilter === f ? "doc-filter-active" : ""}`}
                  onClick={() => setDocumentFilter(f)}
                >
                  {f}
                </button>
              ))}
            </div>
            {/* FIX 2: Show friendly message instead of raw ConnectionTerminated exception */}
            {documentsError && (
              <div className="status status-error compact-status">
                Unable to load documents. Retrying shortly.
              </div>
            )}
            {documentsLoading ? (
              <div className="empty rail-empty">Loading documents...</div>
            ) : filteredDocuments.length === 0 ? (
              <div className="empty rail-empty">
                {debouncedDocumentSearch
                  ? "No documents match your search."
                  : documentFilter !== "All"
                  ? `No ${documentFilter} documents found.`
                  : "No indexed documents available right now."}
              </div>
            ) : (
              <div className="doc-list">
                <ul>
                  {filteredDocuments.map((d) => (
                    <li key={d.document_name}>
                      <span className="doc-name">{d.document_name}</span>
                      <span className="doc-meta">
                        {d.chunk_count} chunks
                        {d.total_pages ? ` · ${d.total_pages} pages` : ""}
                        {d.source_type ? ` · ${d.source_type}` : ""}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </aside>
        </div>

        <main className="main-stack">
          <section className="chat-surface">
            <div className="chat-head">
              <div>
                <p className="label">Conversation</p>
                <h3>Assistant</h3>
                <p className="muted">Responses cite the retrieved passages.</p>
              </div>
              <div className="pill-row">
                <span
                  className={`pill ${chatLoading ? "pill-warn" : "pill-muted"}`}
                >
                  {chatLoading ? "Generating..." : "Idle"}
                </span>
                <span className="pill pill-muted">
                  {messages.length} messages
                </span>
              </div>
            </div>

            <div className="chat-body">
              <div className="chat-thread" ref={chatThreadRef}>
                {messages.length === 0 && (
                  <div className="empty">
                    No messages yet. Ask your first question.
                  </div>
                )}
                {messages.map((msg) => {
                  const activeTab = messageTabs[msg.id] ?? "answer";
                  const selectedContextIndex = selectedContexts[msg.id] ?? 0;
                  const selectedContext = msg.contexts?.[selectedContextIndex];
                  const hasContexts = !!(
                    msg.contexts && msg.contexts.length > 0
                  );
                  const hasSources = !!(msg.sources && msg.sources.length > 0);
                  const sourceCount = hasContexts
                    ? msg.contexts!.length
                    : msg.sources?.length ?? 0;

                  return (
                    <div key={msg.id} className={`message message-${msg.role}`}>
                      <div className="message-meta">
                        {msg.role === "user" ? "You" : "Assistant"}
                      </div>

                      {msg.role === "assistant" ? (
                        <>
                          <div className="message-tabs">
                            <button
                              type="button"
                              className={`tab ${
                                activeTab === "answer" ? "tab-active" : ""
                              }`}
                              onClick={() =>
                                setMessageTabs((prev) => ({
                                  ...prev,
                                  [msg.id]: "answer",
                                }))
                              }
                            >
                              Answer
                            </button>
                            <button
                              type="button"
                              className={`tab ${
                                activeTab === "sources" ? "tab-active" : ""
                              }`}
                              onClick={() =>
                                setMessageTabs((prev) => ({
                                  ...prev,
                                  [msg.id]: "sources",
                                }))
                              }
                              disabled={!hasContexts && !hasSources}
                            >
                              Sources ({sourceCount})
                            </button>
                          </div>

                          {activeTab === "answer" && (
                            <div className="message-body">
                              {msg.content}
                              {msg.contexts && msg.contexts.length > 0 && (
                                <div className="context-note">
                                  Context pulled from {msg.contexts.length}{" "}
                                  chunk(s).
                                </div>
                              )}
                              {msg.role === "assistant" && msg.backendMessageId && (
                                <div className="feedback-row">
                                  <button
                                    type="button"
                                    className={`feedback-btn ${
                                      msg.feedback === 1 ? "feedback-btn-active" : ""
                                    }`}
                                    disabled={feedbackLoadingByMessage[msg.id]}
                                    onClick={() => submitFeedback(msg, 1)}
                                  >
                                    Helpful
                                  </button>
                                  <button
                                    type="button"
                                    className={`feedback-btn ${
                                      msg.feedback === 0 ? "feedback-btn-active" : ""
                                    }`}
                                    disabled={feedbackLoadingByMessage[msg.id]}
                                    onClick={() => submitFeedback(msg, 0)}
                                  >
                                    Not helpful
                                  </button>
                                </div>
                              )}
                            </div>
                          )}

                          {activeTab === "sources" && (
                            <div className="sources-grid">
                              <div className="sources-list">
                                {hasContexts ? (
                                  msg.contexts!.map((ctx, idx) => (
                                    <button
                                      key={`${ctx.source}-${idx}`}
                                      type="button"
                                      className={`source-chip ${
                                        selectedContextIndex === idx
                                          ? "source-chip-active"
                                          : ""
                                      }`}
                                      onClick={() =>
                                        setSelectedContexts((prev) => ({
                                          ...prev,
                                          [msg.id]: idx,
                                        }))
                                      }
                                    >
                                      {sourceLabel(ctx, ctx.source)}
                                    </button>
                                  ))
                                ) : hasSources ? (
                                  msg.sources!.map((src) => (
                                    <span
                                      key={src}
                                      className="source-chip source-chip-muted"
                                    >
                                      {sourceLabel(undefined, src)}
                                    </span>
                                  ))
                                ) : (
                                  <div className="empty">
                                    No sources returned for this answer.
                                  </div>
                                )}
                              </div>

                              <div className="context-preview">
                                {hasContexts && selectedContext ? (
                                  <>
                                    <div className="context-meta">
                                      <strong>
                                        {sourceLabel(
                                          selectedContext,
                                          selectedContext.source
                                        )}
                                      </strong>
                                      {selectedContext.page ? (
                                        <span>
                                          {" "}
                                          · Page {selectedContext.page}
                                        </span>
                                      ) : null}
                                    </div>
                                    <div className="context-text">
                                      {highlightText(
                                        selectedContext.content,
                                        msg.question ?? ""
                                      )}
                                    </div>
                                  </>
                                ) : hasSources ? (
                                  <div className="empty">
                                    Context was not returned for these sources.
                                    Rerun after backend restart or reingest.
                                  </div>
                                ) : (
                                  <div className="empty">
                                    Select a source to view its context.
                                  </div>
                                )}
                              </div>
                            </div>
                          )}
                        </>
                      ) : (
                        <div className="message-body">{msg.content}</div>
                      )}
                    </div>
                  );
                })}
                <div ref={messagesEndRef} />
              </div>
            </div>

            <form className="chat-input-row" onSubmit={sendMessage}>
              <input
                type="text"
                className="input"
                placeholder="Ask about the uploaded PDF..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                disabled={chatLoading}
              />
              <button
                className="btn btn-primary"
                type="submit"
                disabled={chatLoading}
              >
                {chatLoading ? "Thinking..." : "Send"}
              </button>
            </form>
            {chatError && (
              <div className="status status-error">{chatError}</div>
            )}
            {feedbackError && (
              <div className="status status-error">{feedbackError}</div>
            )}
          </section>
        </main>

        <div className="right-stack">
          <aside className="sources-rail">
            <div className="sources-rail-head">
              <p className="label">Latest sources</p>
              <h4>Context spotlight</h4>
            </div>
            {latestAssistant?.contexts && latestAssistant.contexts.length > 0 && activeLatestContext ? (
              <div className="sources-rail-body">
                <div className="sources-list">
                  {latestAssistant.contexts.map((ctx, idx) => (
                    <button
                      key={`${ctx.source}-${idx}`}
                      type="button"
                      className={`source-chip ${
                        selectedLatestContextIndex === idx
                          ? "source-chip-active"
                          : ""
                      }`}
                      onClick={() => setSelectedLatestContextIndex(idx)}
                    >
                      {sourceLabel(ctx, ctx.source)}
                    </button>
                  ))}
                </div>
                <div className="context-preview">
                  <div className="context-meta">
                    <strong>
                      {sourceLabel(activeLatestContext, activeLatestContext.source)}
                    </strong>
                    {activeLatestContext.page ? (
                      <span>
                        {" "}
                        · Page {activeLatestContext.page}
                      </span>
                    ) : null}
                  </div>
                  <div className="context-text">
                    {highlightText(activeLatestContext.content, latestAssistant.question ?? "")}
                  </div>
                </div>
              </div>
            ) : (
              <div className="sources-rail-empty">
                Ask a question to see cited passages.
              </div>
            )}
          </aside>

          <aside className="rail conversations-rail">
            <div className="rail-card conversations-card">
              <div className="card-head">
                <div>
                  <h3>Conversations</h3>
                  <p className="muted">Session history from backend endpoints.</p>
                </div>
              </div>
              <div className="action-row">
                <button className="btn btn-primary" type="button" onClick={createSession}>
                  New conversation
                </button>
              </div>
              <div className="conversation-list">
                {conversationLoading && (
                  <p className="muted">Loading conversations...</p>
                )}
                {!conversationLoading && conversations.length === 0 && (
                  <p className="muted">No saved conversations yet.</p>
                )}
                {conversations.map((conv) => (
                  <button
                    key={conv.session_id}
                    type="button"
                    className={`conversation-item ${
                      activeSessionId === conv.session_id
                        ? "conversation-item-active"
                        : ""
                    }`}
                    onClick={() => setActiveSessionId(conv.session_id)}
                  >
                    <strong>{conv.title || "Untitled"}</strong>
                    <span>
                      {conv.message_count} msgs ·{" "}
                      {new Date(conv.last_message_at).toLocaleString()}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          </aside>
        </div>
      </div>

      <footer className="footer">
        The chats and tool responses are experimental and generated by RAG
        models that can make mistakes. Please review the underlying
        documentation via the listed sources before taking action. The tool and
        organization accept no responsibility or liability for legal outcomes.
      </footer>
    </div>
  );
}

export default App;