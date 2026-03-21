"""
api.py — FastAPI server for the SAMA NORA Chatbot
Endpoints:
  GET  /health
  POST /api/query               — standard JSON response
  POST /api/query-stream        — structured NDJSON streaming
  POST /api/feedback            — save like/dislike feedback
  GET  /api/session/{id}/messages — fetch past messages for a session
  GET  /admin/cache/status
  POST /admin/cache/clear

[FIX] session_summary is now passed as a dedicated kwarg to answer_query()
      instead of being prepended to the raw query string. This prevents the
      summary from corrupting the embedding / retrieval path on every 6th,
      12th, 18th message in a session.
"""

from __future__ import annotations

import json
import uuid
import logging
from datetime import datetime, timezone
from typing import Any, Generator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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

# ── Supabase client (reuse from simple_rag or create new) ────────────────────
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

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="SAMA NORA Chatbot", version="3.1.0")

# Allow Next.js dev (3000) + prod domain + any configured origins
_raw_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:3001,http://127.0.0.1:3000")
CORS_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
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
    feedback:          int          # 1 = like, 0 = dislike
    comments:          str | None = None
    user_message:      str | None = None
    assistant_message: str | None = None


# ── DB helpers ────────────────────────────────────────────────────────────────

def _ensure_user(user_id: str) -> None:
    """Insert user row if not exists.
    NOTE: 'user' is a reserved word in PostgreSQL. The Supabase Python client
    handles this correctly when you pass the string "user" to .table(), but
    we add explicit error logging to catch any silent failures.
    """
    try:
        resp = get_sb().table("user").upsert(
            {"user_id": user_id},
            on_conflict="user_id",
        ).execute()
        log.info(f"[db] user upserted: {user_id[:12]}...")
    except Exception as e:
        log.error(f"[db] ensure_user FAILED for {user_id[:12]}: {e}")


def _ensure_session(session_id: str, user_id: str) -> None:
    """Insert session row if not exists."""
    try:
        resp = get_sb().table("session").upsert(
            {"session_id": session_id, "user_id": user_id},
            on_conflict="session_id",
        ).execute()
        log.info(f"[db] session upserted: {session_id[:12]}...")
    except Exception as e:
        log.error(f"[db] ensure_session FAILED for {session_id[:12]}: {e}")


def _save_message(
    message_id: str,
    session_id: str,
    user_id: str,
    user_message: str,
    assistant_message: str,
) -> None:
    """Insert Q&A pair into session_messages."""
    try:
        resp = get_sb().table("session_messages").insert({
            "message_id":        message_id,
            "session_id":        session_id,
            "user_id":           user_id,
            "user_message":      user_message,
            "assistant_message": assistant_message,
        }).execute()
        log.info(f"[db] message saved: {message_id[:12]}...")
    except Exception as e:
        log.error(f"[db] save_message FAILED for {message_id[:12]}: {e}")


def _persist_interaction(
    user_id: str | None,
    session_id: str | None,
    user_message: str,
    assistant_message: str,
    message_id: str,
) -> None:
    """Full persistence flow: user → session → message."""
    if not user_id or not session_id:
        log.warning(f"[db] persist_interaction skipped — missing user_id={user_id!r} session_id={session_id!r}")
        return
    log.info(f"[db] persisting interaction user={user_id[:12]} session={session_id[:12]} msg={message_id[:12]}")
    _ensure_user(user_id)
    _ensure_session(session_id, user_id)
    _save_message(message_id, session_id, user_id, user_message, assistant_message)
    # After saving, check if it's time to update the rolling summary
    _maybe_update_summary(session_id, user_id)


# ── Rolling summary helpers ───────────────────────────────────────────────────

SUMMARY_EVERY_N = 6   # regenerate summary every N messages

def _get_message_count(session_id: str) -> int:
    """Return how many messages exist for this session."""
    try:
        resp = (
            get_sb()
            .table("session_messages")
            .select("message_id", count="exact")
            .eq("session_id", session_id)
            .execute()
        )
        return resp.count or 0
    except Exception as e:
        log.warning(f"[summary] count failed: {e}")
        return 0


def _fetch_last_n_messages(session_id: str, n: int = 6) -> list[dict]:
    """Fetch the last N Q&A pairs for summarisation."""
    try:
        resp = (
            get_sb()
            .table("session_messages")
            .select("user_message, assistant_message")
            .eq("session_id", session_id)
            .order("timestamp", desc=True)
            .limit(n)
            .execute()
        )
        # Reverse so chronological order (oldest first)
        return list(reversed(resp.data or []))
    except Exception as e:
        log.warning(f"[summary] fetch_last_n failed: {e}")
        return []


def _generate_summary(messages: list[dict], existing_summary: str = "") -> str:
    """Call GPT-4o-mini to produce a 2-3 sentence rolling summary."""
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
            "Focus on the regulatory topics covered. Be specific (mention regulation names, "
            "frameworks, specific questions). This will be used as context for future answers."
        )
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.2,
        )
        summary = resp.choices[0].message.content.strip()
        log.info(f"[summary] generated: {summary[:80]}...")
        return summary
    except Exception as e:
        log.error(f"[summary] generate failed: {e}")
        return existing_summary  # fall back to existing summary on error


def _upsert_summary(session_id: str, user_id: str, summary: str, count: int) -> None:
    try:
        get_sb().table("session_summary").upsert({
            "session_id":    session_id,
            "user_id":       user_id,
            "summary_text":  summary,
            "summary_json":  "{}",
            "message_count": count,
        }, on_conflict="session_id").execute()
        log.info(f"[summary] upserted for session {session_id[:12]}")
    except Exception as e:
        log.error(f"[summary] upsert failed: {e}")


def _maybe_update_summary(session_id: str, user_id: str) -> None:
    """Regenerate rolling summary every SUMMARY_EVERY_N messages."""
    try:
        count = _get_message_count(session_id)
        if count % SUMMARY_EVERY_N != 0:
            return   # not time yet
        log.info(f"[summary] triggering update at message count={count}")
        # Fetch existing summary for continuity
        existing = ""
        try:
            ex = (
                get_sb()
                .table("session_summary")
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
    """Fetch the current rolling summary for a session. Returns empty string if none."""
    if not session_id:
        return ""
    try:
        resp = (
            get_sb()
            .table("session_summary")
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
    return {"status": "ok", "version": "3.1.0"}


@app.post("/api/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty")

    message_id = str(uuid.uuid4())

    # [FIX] Fetch the rolling summary but pass it as a dedicated kwarg to
    # answer_query() — do NOT prepend it to the query string. Prepending
    # it corrupted the embedding vector (pulling retrieval toward the summary
    # topic rather than the actual question).
    session_summary = _get_session_summary(req.session_id)

    result = answer_query(
        req.query,                      # clean query — no summary mixed in
        top_k=req.top_k,
        debug=req.debug,
        session_summary=session_summary, # goes to LLM prompt only
    )

    # Persist to DB (non-blocking — we don't fail the request if this errors)
    _persist_interaction(
        req.user_id,
        req.session_id,
        req.query,
        result.get("answer", ""),
        message_id,
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
def query_stream_endpoint(req: QueryRequest):
    """
    Structured NDJSON stream. Each line is a JSON object:

      {"type": "token",   "text": "word "}          — one token/word at a time
      {"type": "sources", "sources": [...],
       "message_id": "uuid", "cached": bool,
       "method": "generative|cached|not_found"}      — sent once, after all tokens
      {"type": "done"}                               — stream ended
      {"type": "error",  "message": "..."}           — on failure
    """
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty")

    def generate() -> Generator[str, None, None]:
        message_id = str(uuid.uuid4())

        try:
            # [FIX] Same fix as query_endpoint: summary passed as kwarg,
            # not prepended to the query string.
            session_summary = _get_session_summary(req.session_id)

            result = answer_query(
                req.query,                       # clean query — no summary mixed in
                top_k=req.top_k,
                debug=req.debug,
                session_summary=session_summary,  # goes to LLM prompt only
            )

            answer = result.get("answer", "")

            # ── Stream tokens word-by-word ─────────────────────────────────
            words = answer.split(" ")
            for i, word in enumerate(words):
                token = word + (" " if i < len(words) - 1 else "")
                yield json.dumps({"type": "token", "text": token}) + "\n"

            # ── Send sources + metadata ────────────────────────────────────
            sources_payload = []
            for s in result.get("sources", []):
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

            # ── Persist to DB after stream completes ───────────────────────
            _persist_interaction(
                req.user_id,
                req.session_id,
                req.query,
                answer,
                message_id,
            )

        except Exception as e:
            log.error(f"[stream] error: {e}", exc_info=True)
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={
            # Required for streaming to work through proxies/browsers
            "X-Accel-Buffering":  "no",
            "Cache-Control":      "no-cache",
            "Transfer-Encoding":  "chunked",
        },
    )


@app.post("/api/feedback")
def feedback_endpoint(req: FeedbackRequest):
    """
    Save a like (1) or dislike (0) for a message.
    Also stores the Q&A text for future LLM fine-tuning.
    """
    if req.feedback not in (0, 1):
        raise HTTPException(status_code=400, detail="feedback must be 0 (dislike) or 1 (like)")

    log.info(f"[feedback] saving vote={req.feedback} msg={req.message_id[:12]} user={req.user_id[:12]} comments={req.comments!r}")
    try:
        # Build insert payload — only include optional fields if they have values
        # (avoids PGRST204 if columns don't exist yet in the schema cache)
        payload: dict = {
            "session_id":  req.session_id,
            "user_id":     req.user_id,
            "message_id":  req.message_id,
            "feedback":    req.feedback,
        }
        if req.comments          is not None: payload["comments"]          = req.comments
        if req.user_message      is not None: payload["user_message"]      = req.user_message
        if req.assistant_message is not None: payload["assistant_message"] = req.assistant_message

        resp = get_sb().table("session_feedback").insert(payload).execute()
        log.info(f"[feedback] saved OK: {resp.data}")
        return {"status": "ok", "feedback": req.feedback}
    except Exception as e:
        log.error(f"[feedback] DB insert FAILED: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/session/{session_id}/messages")
def get_session_messages(session_id: str, limit: int = 20):
    """
    Return the last `limit` messages for a session (for conversation reload on refresh).
    Returns list of {message_id, user_message, assistant_message, timestamp}
    """
    try:
        resp = (
            get_sb()
            .table("session_messages")
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


# ── Cache admin endpoints ─────────────────────────────────────────────────────

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
            "backend": "redis", "connected": True,
            "cached_entries": count,
            "ttl_seconds": ttl,
            "ttl_days": round(ttl / 86400, 1) if ttl > 0 else "no expiry set",
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)