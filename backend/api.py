"""
api.py — FastAPI server for the SAMA NORA Chatbot
Endpoints:
  GET  /health
  POST /api/query               — standard JSON response
  POST /api/query-stream        — structured NDJSON streaming
  POST /api/feedback            — save like/dislike feedback
  GET  /api/session/{id}/messages — fetch past messages for a session
  GET  /api/conversations       — list all sessions for a user
  GET  /api/documents           — list ingested documents
  GET  /admin/stats             — system metrics
  GET  /admin/cache/status
  POST /admin/cache/clear
  POST /admin/keys              — generate a new API key
  GET  /admin/keys              — list all API keys
  DELETE /admin/keys/{id}       — revoke an API key

Changes in this version:
  - [FIX] Streaming endpoint clears sources when LLM returns not-found answer.
  - [FIX] Added _sb_with_retry() wrapper to handle Supabase ConnectionTerminated
    errors by recreating the client and retrying once automatically.
  - [FIX] list_documents, list_conversations, get_session_messages now use retry wrapper.
  - [AUTH] Added API key authentication on all /api/* endpoints.
    Keys are stored in the api_keys Supabase table and cached in memory for 60s.
    Admin endpoints (create/list/revoke keys) require ADMIN_API_KEY env var.
"""

from __future__ import annotations

import json
import uuid
import time as _time
import logging
import secrets
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Generator

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("api")

from simple_rag import answer_query

# ── Rolling stats trackers ────────────────────────────────────────────────────
_request_times: list[float] = []   # response times in ms, last 50
_cache_hits:    int          = 0
_cache_total:   int          = 0

# ── Supabase client ───────────────────────────────────────────────────────────
_sb = None

def get_sb():
    global _sb
    if _sb is None:
        from supabase import create_client
        _sb = create_client(
            os.environ["SUPABASE_URL"],
            os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or "",
        )
    return _sb


def _sb_with_retry(fn):
    """
    Retry a Supabase call once with a fresh client if a ConnectionTerminated
    error is encountered (HTTP/2 error_code:1 or error_code:9).
    """
    global _sb
    try:
        return fn(get_sb())
    except Exception as e:
        err_str = str(e)
        if "ConnectionTerminated" in err_str or "error_code:" in err_str:
            log.warning(f"[db] ConnectionTerminated detected — recreating client and retrying: {e}")
            _sb = None
            return fn(get_sb())
        raise


# ── API Key authentication ────────────────────────────────────────────────────
# Keys are stored in the `api_keys` Supabase table and cached in memory for
# 60 seconds so we don't hit the DB on every single request.

_api_key_cache: set[str] = set()
_api_key_cache_ts: float = 0.0
_API_KEY_CACHE_TTL = 60.0  # seconds

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def _load_api_keys() -> set[str]:
    """Load active API keys from Supabase, with 60-second in-memory cache."""
    global _api_key_cache, _api_key_cache_ts
    now = _time.monotonic()
    if now - _api_key_cache_ts < _API_KEY_CACHE_TTL and _api_key_cache:
        return _api_key_cache
    try:
        resp = _sb_with_retry(
            lambda sb: sb.table("api_keys")
            .select("api_key")
            .eq("is_active", True)
            .execute()
        )
        keys = {row["api_key"] for row in (resp.data or [])}
        _api_key_cache = keys
        _api_key_cache_ts = now
        log.info(f"[auth] Loaded {len(keys)} active API key(s) from Supabase")
        return keys
    except Exception as e:
        log.warning(f"[auth] Failed to load API keys from Supabase: {e}. Using stale cache.")
        return _api_key_cache  # return stale cache on DB failure


def _require_api_key(key: str = Security(_API_KEY_HEADER)) -> None:
    """
    FastAPI dependency — validates X-API-Key header against the api_keys table.
    If no keys exist in the table yet (initial setup), auth is skipped so you
    can call /admin/keys to create the first key without being locked out.
    """
    valid_keys = _load_api_keys()
    if not valid_keys:
        # No keys configured yet — skip auth during initial setup
        return
    if not key or key not in valid_keys:
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing API key. Include X-API-Key header."
        )


def _invalidate_key_cache() -> None:
    """Force the next request to reload keys from Supabase."""
    global _api_key_cache_ts
    _api_key_cache_ts = 0.0


# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="SAMA NORA Chatbot", version="3.3.0")

_raw_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:3001,http://127.0.0.1:3000")
CORS_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# ── Request / response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query:      str
    user_id:    str | None = None
    session_id: str | None = None
    top_k:      int | None = None
    debug:      bool = False


class Source(BaseModel):
    document_name: str
    page_start:    int | None = None
    page_end:      int | None = None
    section_title: str | None = None
    similarity:    float = 0.0
    snippet:       str | None = None


class QueryResponse(BaseModel):
    answer:             str
    sources:            list[Source]
    cached:             bool
    method:             str | None = None
    message_id:         str | None = None
    candidate_count:    int | None = None
    reranker_top_score: float | None = None


class FeedbackRequest(BaseModel):
    session_id:        str
    user_id:           str
    message_id:        str
    feedback:          int
    comments:          str | None = None
    user_message:      str | None = None
    assistant_message: str | None = None


class CreateKeyRequest(BaseModel):
    label: str  # e.g. "ZetaLabs Internal", "Client A"


# ── Helper: detect not-found answers ─────────────────────────────────────────

def _answer_is_not_found(answer: str) -> bool:
    """Return True if the LLM answered that no information was found."""
    a = answer.lower()
    return any(p in a for p in [
        "does not contain", "cannot find", "not found in",
        "لا تتوفر", "لم أجد",
    ])


# ── DB helpers ────────────────────────────────────────────────────────────────

def _ensure_user(user_id: str) -> None:
    try:
        _sb_with_retry(
            lambda sb: sb.table("user").upsert(
                {"user_id": user_id},
                on_conflict="user_id",
            ).execute()
        )
    except Exception as e:
        log.error(f"[db] ensure_user FAILED for {user_id[:12]}: {e}")


def _ensure_session(session_id: str, user_id: str) -> None:
    try:
        _sb_with_retry(
            lambda sb: sb.table("session").upsert(
                {"session_id": session_id, "user_id": user_id},
                on_conflict="session_id",
            ).execute()
        )
    except Exception as e:
        log.error(f"[db] ensure_session FAILED for {session_id[:12]}: {e}")


def _save_message(
    message_id: str,
    session_id: str,
    user_id: str,
    user_message: str,
    assistant_message: str,
) -> None:
    try:
        _sb_with_retry(
            lambda sb: sb.table("session_messages").insert({
                "message_id":        message_id,
                "session_id":        session_id,
                "user_id":           user_id,
                "user_message":      user_message,
                "assistant_message": assistant_message,
            }).execute()
        )
    except Exception as e:
        log.error(f"[db] save_message FAILED for {message_id[:12]}: {e}")


def _persist_interaction(
    user_id: str | None,
    session_id: str | None,
    user_message: str,
    assistant_message: str,
    message_id: str,
) -> None:
    if not user_id or not session_id:
        return
    _ensure_user(user_id)
    _ensure_session(session_id, user_id)
    _save_message(message_id, session_id, user_id, user_message, assistant_message)
    _maybe_update_summary(session_id, user_id)


# ── Rolling summary helpers ───────────────────────────────────────────────────

SUMMARY_EVERY_N = 6

def _get_message_count(session_id: str) -> int:
    try:
        resp = _sb_with_retry(
            lambda sb: sb.table("session_messages")
            .select("message_id", count="exact")
            .eq("session_id", session_id)
            .execute()
        )
        return resp.count or 0
    except Exception as e:
        log.warning(f"[summary] count failed: {e}")
        return 0


def _fetch_last_n_messages(session_id: str, n: int = 6) -> list[dict]:
    try:
        resp = _sb_with_retry(
            lambda sb: sb.table("session_messages")
            .select("user_message, assistant_message")
            .eq("session_id", session_id)
            .order("timestamp", desc=True)
            .limit(n)
            .execute()
        )
        return list(reversed(resp.data or []))
    except Exception as e:
        log.warning(f"[summary] fetch_last_n failed: {e}")
        return []


def _generate_summary(messages: list[dict], existing_summary: str = "") -> str:
    try:
        import openai
        history_text = "\n".join(
            f"User: {m['user_message']}\nAssistant: {m['assistant_message']}"
            for m in messages
        )
        prior = f"Prior summary: {existing_summary}\n\n" if existing_summary else ""
        prompt = (
            f"{prior}Recent conversation:\n{history_text}\n\n"
            "Summarise what the user has been asking about in 2-3 concise sentences. "
            "Focus on the regulatory topics covered."
        )
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        log.error(f"[summary] generate failed: {e}")
        return existing_summary


def _upsert_summary(session_id: str, user_id: str, summary: str, count: int) -> None:
    try:
        _sb_with_retry(
            lambda sb: sb.table("session_summary").upsert({
                "session_id":    session_id,
                "user_id":       user_id,
                "summary_text":  summary,
                "summary_json":  "{}",
                "message_count": count,
            }, on_conflict="session_id").execute()
        )
    except Exception as e:
        log.error(f"[summary] upsert failed: {e}")


def _maybe_update_summary(session_id: str, user_id: str) -> None:
    try:
        count = _get_message_count(session_id)
        if count % SUMMARY_EVERY_N != 0:
            return
        existing = ""
        try:
            ex = _sb_with_retry(
                lambda sb: sb.table("session_summary")
                .select("summary_text")
                .eq("session_id", session_id)
                .limit(1)
                .execute()
            )
            if ex.data:
                existing = ex.data[0].get("summary_text", "")
        except Exception:
            pass
        messages = _fetch_last_n_messages(session_id, n=SUMMARY_EVERY_N)
        if not messages:
            return
        summary = _generate_summary(messages, existing)
        _upsert_summary(session_id, user_id, summary, count)
    except Exception as e:
        log.error(f"[summary] maybe_update failed: {e}")


def _get_session_summary(session_id: str) -> str:
    if not session_id:
        return ""
    try:
        resp = _sb_with_retry(
            lambda sb: sb.table("session_summary")
            .select("summary_text")
            .eq("session_id", session_id)
            .limit(1)
            .execute()
        )
        if resp.data:
            return resp.data[0].get("summary_text", "")
    except Exception as e:
        log.warning(f"[summary] fetch failed: {e}")
    return ""


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "3.3.0"}


@app.post("/api/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest, _: None = Depends(_require_api_key)):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty")

    message_id      = str(uuid.uuid4())
    session_summary = _get_session_summary(req.session_id)

    result = answer_query(
        req.query,
        top_k=req.top_k,
        debug=req.debug,
        session_summary=session_summary,
    )

    _persist_interaction(
        req.user_id, req.session_id,
        req.query, result.get("answer", ""), message_id,
    )

    return QueryResponse(
        answer=result["answer"],
        sources=[Source(**s) for s in result.get("sources", [])],
        cached=result.get("cached", False),
        method=result.get("method"),
        message_id=message_id,
        candidate_count=result.get("candidate_count"),
        reranker_top_score=result.get("reranker_top_score"),
    )


@app.post("/api/query-stream")
def query_stream_endpoint(req: QueryRequest, _: None = Depends(_require_api_key)):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty")

    def generate() -> Generator[str, None, None]:
        global _cache_total, _cache_hits
        message_id = str(uuid.uuid4())

        try:
            t_start         = _time.perf_counter()
            session_summary = _get_session_summary(req.session_id)

            result = answer_query(
                req.query,
                top_k=req.top_k,
                debug=req.debug,
                session_summary=session_summary,
            )

            elapsed_ms = (_time.perf_counter() - t_start) * 1000
            _request_times.append(elapsed_ms)
            if len(_request_times) > 50:
                _request_times.pop(0)
            _cache_total += 1
            if result.get("cached"):
                _cache_hits += 1

            answer = result.get("answer", "")

            words = answer.split(" ")
            for i, word in enumerate(words):
                token = word + (" " if i < len(words) - 1 else "")
                yield json.dumps({"type": "token", "text": token}) + "\n"

            if _answer_is_not_found(answer):
                sources_to_send = []
                log.info(f"[stream] not-found answer — clearing sources for clean UX")
            else:
                sources_to_send = result.get("sources", [])

            sources_payload = []
            for s in sources_to_send:
                sources_payload.append({
                    "document_name": s.get("document_name", ""),
                    "page_start":    s.get("page_start"),
                    "page_end":      s.get("page_end"),
                    "section_title": s.get("section_title"),
                    "similarity":    round(float(s.get("similarity", 0)), 4),
                    "snippet":       (s.get("snippet") or s.get("content") or "")[:300],
                })

            yield json.dumps({
                "type":       "sources",
                "sources":    sources_payload,
                "message_id": message_id,
                "cached":     result.get("cached", False),
                "method":     result.get("method"),
            }) + "\n"

            yield json.dumps({"type": "done"}) + "\n"

            _persist_interaction(
                req.user_id, req.session_id,
                req.query, answer, message_id,
            )

        except Exception as e:
            log.error(f"[stream] error: {e}", exc_info=True)
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={
            "X-Accel-Buffering":  "no",
            "Cache-Control":      "no-cache",
            "Transfer-Encoding":  "chunked",
        },
    )


@app.post("/api/feedback")
def feedback_endpoint(req: FeedbackRequest, _: None = Depends(_require_api_key)):
    if req.feedback not in (0, 1):
        raise HTTPException(status_code=400, detail="feedback must be 0 or 1")

    try:
        payload: dict = {
            "session_id": req.session_id,
            "user_id":    req.user_id,
            "message_id": req.message_id,
            "feedback":   req.feedback,
        }
        if req.comments          is not None: payload["comments"]          = req.comments
        if req.user_message      is not None: payload["user_message"]      = req.user_message
        if req.assistant_message is not None: payload["assistant_message"] = req.assistant_message

        _sb_with_retry(
            lambda sb: sb.table("session_feedback").insert(payload).execute()
        )
        return {"status": "ok", "feedback": req.feedback}
    except Exception as e:
        log.error(f"[feedback] DB insert FAILED: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/session/{session_id}/messages")
def get_session_messages(session_id: str, limit: int = 20, _: None = Depends(_require_api_key)):
    try:
        resp = _sb_with_retry(
            lambda sb: sb.table("session_messages")
            .select("message_id, user_message, assistant_message, timestamp")
            .eq("session_id", session_id)
            .order("timestamp", desc=False)
            .limit(limit)
            .execute()
        )
        return {"session_id": session_id, "messages": resp.data or []}
    except Exception as e:
        log.error(f"[session] fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/conversations")
def list_conversations(user_id: str = "", limit: int = 50, _: None = Depends(_require_api_key)):
    """Return all sessions for a user, sorted newest first, with title = first message."""
    if not user_id:
        return {"conversations": []}
    try:
        resp = _sb_with_retry(
            lambda sb: sb.table("session_messages")
            .select("session_id, user_message, timestamp")
            .eq("user_id", user_id)
            .order("timestamp", desc=False)
            .execute()
        )
        rows = resp.data or []

        seen: dict = {}
        for row in rows:
            sid = row["session_id"]
            if sid not in seen:
                title = (row.get("user_message") or "New conversation")[:40]
                seen[sid] = {
                    "session_id":      sid,
                    "title":           title,
                    "last_message_at": row["timestamp"],
                    "message_count":   1,
                }
            else:
                seen[sid]["last_message_at"] = row["timestamp"]
                seen[sid]["message_count"]  += 1

        conversations = sorted(
            seen.values(),
            key=lambda x: x["last_message_at"],
            reverse=True,
        )
        return {"conversations": conversations[:limit]}
    except Exception as e:
        log.error(f"[conversations] failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents")
def list_documents(search: str = "", limit: int = 20, _: None = Depends(_require_api_key)):
    """
    Return all documents from the documents table as the source of truth,
    with chunk counts joined from sama_nora_chunks.
    """
    try:
        doc_resp = _sb_with_retry(
            lambda sb: sb.table("documents")
            .select("document_name, source_type, total_pages")
            .execute()
        )
        all_docs = doc_resp.data or []

        chunks_resp = _sb_with_retry(
            lambda sb: sb.table("sama_nora_chunks")
            .select("document_name")
            .execute()
        )
        chunk_counts = Counter(r["document_name"] for r in (chunks_resp.data or []))

        seen: dict[str, dict] = {}
        for doc in all_docs:
            name = doc.get("document_name", "").strip()
            if not name or len(name) < 4:
                continue
            if name not in seen:
                seen[name] = doc
            else:
                if seen[name].get("source_type") is None and doc.get("source_type"):
                    seen[name] = doc

        results = []
        for name, doc in seen.items():
            if search and search.lower() not in name.lower():
                continue
            results.append({
                "document_name": name,
                "source_type":   doc.get("source_type") or "SAMA",
                "total_pages":   doc.get("total_pages") or "?",
                "chunk_count":   chunk_counts.get(name, 0),
            })

        results.sort(key=lambda x: x["chunk_count"], reverse=True)
        return {"documents": results[:limit], "total": len(results)}
    except Exception as e:
        log.error(f"[documents] list failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/stats")
def admin_stats():
    """Return system metrics for the dashboard info card."""
    docs_count = 0
    try:
        d = _sb_with_retry(
            lambda sb: sb.table("documents").select("id", count="exact").execute()
        )
        docs_count = d.count or 0
    except Exception as e:
        log.warning(f"[stats] docs count failed: {e}")

    chunks_count = 0
    try:
        c = _sb_with_retry(
            lambda sb: sb.table("sama_nora_chunks").select("id", count="exact").execute()
        )
        chunks_count = c.count or 0
    except Exception as e:
        log.warning(f"[stats] chunks count failed: {e}")

    cached_answers = 0
    try:
        redis_url = os.getenv("REDIS_URL", "")
        if redis_url:
            import redis as redis_lib
            r = redis_lib.from_url(
                redis_url, socket_timeout=2,
                socket_connect_timeout=2, decode_responses=False,
            )
            cached_answers = r.llen("sama:cache:embeddings")
    except Exception:
        pass

    hit_rate = 0.0
    if _cache_total > 0:
        hit_rate = round(_cache_hits / _cache_total * 100, 1)

    avg_ms = 0
    if _request_times:
        avg_ms = round(sum(_request_times) / len(_request_times))

    return {
        "api_status":         "ok",
        "docs_ingested":      docs_count,
        "total_chunks":       chunks_count,
        "cached_answers":     cached_answers,
        "cache_hit_rate_pct": hit_rate,
        "avg_response_ms":    avg_ms,
        "model":              os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    }


@app.get("/admin/cache/status")
def cache_status():
    backend   = os.getenv("CACHE_BACKEND", "memory")
    redis_url = os.getenv("REDIS_URL", "")
    if backend != "redis" or not redis_url:
        return {"backend": "memory", "note": "Set CACHE_BACKEND=redis and REDIS_URL for persistent cache"}
    try:
        import redis as redis_lib
        r = redis_lib.from_url(redis_url, socket_timeout=3, socket_connect_timeout=3, decode_responses=False)
        r.ping()
        count = r.llen("sama:cache:embeddings")
        ttl   = r.ttl("sama:cache:embeddings")
        return {
            "backend":        "redis",
            "connected":      True,
            "cached_entries": count,
            "ttl_seconds":    ttl,
            "ttl_days":       round(ttl / 86400, 1) if ttl > 0 else "no expiry set",
        }
    except Exception as e:
        return {"backend": "redis", "connected": False, "error": str(e)}


@app.post("/admin/cache/clear")
def cache_clear(api_key: str = ""):
    expected_key = os.getenv("ADMIN_API_KEY", "")
    if expected_key and api_key != expected_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    backend   = os.getenv("CACHE_BACKEND", "memory")
    redis_url = os.getenv("REDIS_URL", "")
    if backend != "redis" or not redis_url:
        from simple_rag import _mem_cache
        count = len(_mem_cache)
        _mem_cache.clear()
        return {"cleared": count, "backend": "memory"}
    try:
        import redis as redis_lib
        r = redis_lib.from_url(redis_url, socket_timeout=3, socket_connect_timeout=3, decode_responses=False)
        keys = r.keys("sama:cache:*")
        if keys:
            r.delete(*keys)
        return {"cleared": len(keys), "backend": "redis"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── API Key management endpoints ──────────────────────────────────────────────

@app.post("/admin/keys")
def create_api_key(req: CreateKeyRequest, admin_key: str = ""):
    """
    Generate a new API key for a client/user.
    Requires ADMIN_API_KEY in the admin_key query param (if set in env).
    Example: POST /admin/keys?admin_key=xxx  {"label": "Client A"}
    Returns the raw key ONCE — store it securely, it cannot be retrieved again.
    """
    expected = os.getenv("ADMIN_API_KEY", "")
    if expected and admin_key != expected:
        raise HTTPException(status_code=403, detail="Invalid admin key")
    if not req.label or not req.label.strip():
        raise HTTPException(status_code=400, detail="label cannot be empty")

    new_key = secrets.token_hex(32)  # 64-char hex string
    try:
        result = _sb_with_retry(
            lambda sb: sb.table("api_keys").insert({
                "api_key": new_key,
                "label":   req.label.strip(),
            }).execute()
        )
        row = (result.data or [{}])[0]
        _invalidate_key_cache()
        log.info(f"[auth] Created new API key for label='{req.label}'")
        return {
            "api_key":    new_key,
            "id":         row.get("id"),
            "label":      req.label.strip(),
            "created_at": row.get("created_at"),
            "warning":    "Store this key securely. It will not be shown again.",
        }
    except Exception as e:
        log.error(f"[auth] create_api_key failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/keys")
def list_api_keys(admin_key: str = ""):
    """
    List all API keys (labels + status only, raw key values are never returned).
    Requires ADMIN_API_KEY in the admin_key query param (if set in env).
    """
    expected = os.getenv("ADMIN_API_KEY", "")
    if expected and admin_key != expected:
        raise HTTPException(status_code=403, detail="Invalid admin key")
    try:
        resp = _sb_with_retry(
            lambda sb: sb.table("api_keys")
            .select("id, label, is_active, created_at")
            .order("created_at", desc=True)
            .execute()
        )
        return {"keys": resp.data or []}
    except Exception as e:
        log.error(f"[auth] list_api_keys failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/admin/keys/{key_id}")
def revoke_api_key(key_id: str, admin_key: str = ""):
    """
    Revoke an API key by its UUID (get UUID from GET /admin/keys).
    Requires ADMIN_API_KEY in the admin_key query param (if set in env).
    The key is deactivated, not deleted — audit trail is preserved.
    """
    expected = os.getenv("ADMIN_API_KEY", "")
    if expected and admin_key != expected:
        raise HTTPException(status_code=403, detail="Invalid admin key")
    try:
        _sb_with_retry(
            lambda sb: sb.table("api_keys")
            .update({"is_active": False})
            .eq("id", key_id)
            .execute()
        )
        _invalidate_key_cache()
        log.info(f"[auth] Revoked API key id={key_id}")
        return {"status": "revoked", "id": key_id}
    except Exception as e:
        log.error(f"[auth] revoke_api_key failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)