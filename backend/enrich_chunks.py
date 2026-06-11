"""
enrich_chunks.py — Add contextual headers to existing SAMA NORA chunks
ZetaLabs / IOTA Technologies

Run ONCE to enrich all 32,655 existing chunks in Supabase with
contextual headers — Step 1 of the Advanced RAG Improvement Plan.

What it does:
  1. Loads document_name → source_type mapping from documents table
  2. Reads all existing chunks from Supabase in batches
  3. For each chunk, calls GPT-4o-mini to generate a 1-sentence
     context header using document name + source_type (SAMA/NCA/PDPL)
     + section title + content preview
  4. Prepends "[Context: ...]\n\n" to the chunk content
  5. Re-embeds the enriched content with multilingual-e5-small
  6. Updates the record in Supabase (content + embedding only)

Fixes in v2:
  - Loads source_type (SAMA / NCA / PDPL / ISO) per document so the
    LLM generates headers with the correct regulatory body name
  - Added --force flag to re-enrich already-enriched chunks
    (use this to fix the first 10 chunks that got wrong SAMA labels)

Idempotent:
  Skips chunks that already start with "[Context:" unless --force is used.

Cost estimate : ~$3-4 total for all 32,655 chunks (GPT-4o-mini)
Time estimate : ~11 hours per terminal (run two terminals in parallel)
Risk          : Zero — only updates content and embedding fields.

Usage:
    python enrich_chunks.py                     # enrich all
    python enrich_chunks.py --dry-run           # test, no DB writes
    python enrich_chunks.py --limit 10          # test first 10
    python enrich_chunks.py --force --limit 10  # re-do first 10 (fix wrong headers)
    python enrich_chunks.py --source SAMA       # SAMA only (Terminal 1)
    python enrich_chunks.py --source NCA        # NCA only  (Terminal 2)
    python enrich_chunks.py --skip-embedding    # content only, no re-embed
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import logging
from datetime import timedelta
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
SUPABASE_URL     = os.environ["SUPABASE_URL"]
SUPABASE_KEY     = (os.environ.get("SUPABASE_KEY") or
                    os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or "")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
AZURE_KEY        = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_ENDPOINT   = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT", "gpt-4o-mini")
LLM_BACKEND      = os.getenv("LLM_BACKEND", "openai")
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")

CONTEXT_PREFIX      = "[Context:"
BATCH_SIZE          = 50
LLM_DELAY           = 0.25
DB_DELAY            = 0.1
MAX_CONTENT_PREVIEW = 400

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("enrich_chunks.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Lazy singletons ───────────────────────────────────────────────────────────
_supabase   = None
_embedder   = None
_llm_client = None
_llm_model  = None


def get_supabase():
    global _supabase
    if _supabase is None:
        from supabase import create_client
        _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        log.info(f"[embedder] Loading {EMBEDDING_MODEL} ...")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
        log.info("[embedder] Ready.")
    return _embedder


def get_llm_client():
    global _llm_client, _llm_model
    if _llm_client is None:
        import openai
        if LLM_BACKEND == "azure" and AZURE_KEY and AZURE_ENDPOINT:
            _llm_client = openai.AzureOpenAI(
                api_key=AZURE_KEY,
                azure_endpoint=AZURE_ENDPOINT,
                api_version="2024-02-01",
            )
            _llm_model = AZURE_DEPLOYMENT
            log.info(f"[llm] Using Azure OpenAI — deployment: {AZURE_DEPLOYMENT}")
        else:
            _llm_client = openai.OpenAI(api_key=OPENAI_API_KEY)
            _llm_model  = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            log.info(f"[llm] Using OpenAI — model: {_llm_model}")
    return _llm_client, _llm_model


def embed_text(text: str) -> list[float]:
    model = get_embedder()
    prefix = "passage: " if "e5" in EMBEDDING_MODEL.lower() else ""
    return model.encode(prefix + text, normalize_embeddings=True).tolist()


# ── Source type map ───────────────────────────────────────────────────────────

# Human-readable names for each source type used in the context prompt
SOURCE_TYPE_NAMES = {
    "SAMA":  "SAMA (Saudi Arabian Monetary Authority)",
    "NCA":   "NCA (National Cybersecurity Authority)",
    "PDPL":  "PDPL (Personal Data Protection Law / SDAIA)",
    "ISO":   "ISO International Standard",
    "SDAIA": "SDAIA (Saudi Data and AI Authority)",
}

def infer_source_from_name(document_name: str) -> Optional[str]:
    """
    Infer regulatory body from document name.
    More reliable than DB source_type which may be incorrectly set to SAMA.
    This runs FIRST — DB lookup is only a fallback.
    """
    name = document_name.upper()

    NCA_SIGNALS  = ["ECC", "CCC", "OTCC", "NCA", "CYBERSECURITY CONTROL",
                    "CYBER SECURITY CONTROL", "ESSENTIAL CYBERSECURITY",
                    "CLOUD CYBERSECURITY", "OPERATIONAL TECHNOLOGY",
                    "CYBERSECURITY STEERING"]
    PDPL_SIGNALS = ["PDPL", "PERSONAL DATA", "DATA PROTECTION", "SDAIA", "NDMO",
                    "PRIVACY LAW", "PERSONAL INFORMATION"]
    ISO_SIGNALS  = ["ISO 27001", "ISO 27701", "ISO 22301", "ISO 20000",
                    "ISO 42001", "ISO 23200", "ISO 27400", "IEC 27"]

    for s in NCA_SIGNALS:
        if s in name:
            return "NCA"
    for s in PDPL_SIGNALS:
        if s in name:
            return "PDPL"
    for s in ISO_SIGNALS:
        if s in name:
            return "ISO"

    return None  # fall back to DB source_type or default SAMA


def load_source_map() -> dict[str, str]:
    """
    Load document_id to source_type from the documents table.
    Used as FALLBACK when document name inference does not match.
    Note: DB values may be incorrectly set to SAMA for NCA/PDPL docs.
    That is why infer_source_from_name() runs first in the chunk loop.
    """
    try:
        resp = get_supabase().table("documents").select("id, source_type").execute()
        rows = resp.data or []
        id_map: dict[str, str] = {}
        for row in rows:
            doc_id      = row.get("id", "")
            source_type = (row.get("source_type") or "SAMA").strip().upper()
            if doc_id:
                id_map[doc_id] = source_type

        log.info(f"[init] Loaded DB source types for {len(id_map)} documents")

        from collections import Counter
        dist = Counter(id_map.values())
        for src, count in sorted(dist.items()):
            log.info(f"[init]   DB {src}: {count} documents")

        return id_map
    except Exception as e:
        log.warning(f"[init] Could not load source map: {e} — will use name inference only")
        return {}


# ── Context header generation ─────────────────────────────────────────────────

CONTEXT_PROMPT = """\
You are a Saudi banking and cybersecurity regulation expert.

Regulatory body : {source_type_name}
Document name   : {document_name}
Section title   : {section_title}
Content preview : {content_preview}

Write ONE sentence (maximum 70 words) that precisely describes:
1. Which specific regulation or framework this text belongs to — use the EXACT regulatory body shown above ({source_type_name}), not a different one
2. What regulatory topic it covers (e.g. account opening, AML, capital adequacy, cybersecurity controls, data protection)
3. What entity type or scenario it applies to (e.g. SME, individual, bank, corporate, fintech, government entity)

Return ONLY the sentence. No preamble, no labels, no explanation."""


def generate_context_header(
    document_name: str,
    section_title: Optional[str],
    content: str,
    source_type: str = "SAMA",
) -> Optional[str]:
    """
    Call GPT-4o-mini to generate a 1-sentence context header for a chunk.
    source_type is used to tell the LLM which regulatory body owns this document.
    Returns None on failure — chunk is skipped gracefully.
    """
    client, model = get_llm_client()

    source_type_name = SOURCE_TYPE_NAMES.get(source_type, source_type)

    prompt = CONTEXT_PROMPT.format(
        source_type_name = source_type_name,
        document_name    = document_name or "Unknown regulatory document",
        section_title    = section_title or "N/A",
        content_preview  = content[:MAX_CONTENT_PREVIEW].replace("\n", " "),
    )

    try:
        resp = client.chat.completions.create(
            model       = model,
            messages    = [{"role": "user", "content": prompt}],
            max_tokens  = 100,
            temperature = 0.1,
        )
        header = resp.choices[0].message.content.strip().strip('"\'')
        if not header or len(header) < 10:
            return None
        return header
    except Exception as e:
        log.warning(f"[llm] Header generation failed: {e}")
        return None


def enrich_content(original_content: str, header: str) -> str:
    return f"[Context: {header}]\n\n{original_content}"


def strip_old_header(content: str) -> str:
    """Remove existing [Context: ...] header so we can re-enrich with correct one."""
    if not content.startswith(CONTEXT_PREFIX):
        return content
    # Find end of the context line(s) — header ends at first \n\n
    end = content.find("\n\n")
    if end == -1:
        return content
    return content[end + 2:].strip()


# ── Supabase helpers ──────────────────────────────────────────────────────────

def get_total_chunk_count() -> int:
    try:
        resp = get_supabase().table("sama_nora_chunks").select("id", count="exact").execute()
        return resp.count or 0
    except Exception as e:
        log.error(f"[db] Failed to get chunk count: {e}")
        return 0


def fetch_chunk_batch(offset: int, batch_size: int) -> list[dict]:
    """Fetch a batch of chunks including document_id for source_type lookup."""
    try:
        resp = (
            get_supabase()
            .table("sama_nora_chunks")
            .select("id, document_id, document_name, section_title, content")
            .range(offset, offset + batch_size - 1)
            .order("id")
            .execute()
        )
        return resp.data or []
    except Exception as e:
        log.error(f"[db] Batch fetch failed at offset {offset}: {e}")
        return []


def update_chunk(
    chunk_id: str,
    enriched_content: str,
    new_embedding: list[float],
    dry_run: bool = False,
) -> bool:
    if dry_run:
        return True
    try:
        get_supabase().table("sama_nora_chunks").update({
            "content":   enriched_content,
            "embedding": new_embedding,
        }).eq("id", chunk_id).execute()
        return True
    except Exception as e:
        log.error(f"[db] Update failed for chunk {chunk_id[:12]}: {e}")
        return False


def update_chunk_content_only(
    chunk_id: str,
    enriched_content: str,
    dry_run: bool = False,
) -> bool:
    if dry_run:
        return True
    try:
        get_supabase().table("sama_nora_chunks").update({
            "content": enriched_content,
        }).eq("id", chunk_id).execute()
        return True
    except Exception as e:
        log.error(f"[db] Content update failed for chunk {chunk_id[:12]}: {e}")
        return False


# ── Progress tracker ──────────────────────────────────────────────────────────

class Progress:
    def __init__(self, total: int):
        self.total      = total
        self.enriched   = 0
        self.skipped    = 0
        self.failed     = 0
        self.start_time = time.time()

    @property
    def done(self) -> int:
        return self.enriched + self.skipped + self.failed

    def eta(self) -> str:
        if self.done == 0:
            return "calculating..."
        elapsed   = time.time() - self.start_time
        rate      = self.done / elapsed
        remaining = (self.total - self.done) / rate if rate > 0 else 0
        return str(timedelta(seconds=int(remaining)))

    def log_progress(self):
        pct = (self.done / self.total * 100) if self.total > 0 else 0
        log.info(
            f"[progress] {self.done:,}/{self.total:,} ({pct:.1f}%) — "
            f"enriched={self.enriched:,} skipped={self.skipped:,} "
            f"failed={self.failed} ETA={self.eta()}"
        )

    def log_summary(self):
        elapsed  = timedelta(seconds=int(time.time() - self.start_time))
        cost_est = self.enriched * 0.0001
        log.info("")
        log.info("=" * 55)
        log.info("  ENRICHMENT COMPLETE")
        log.info("=" * 55)
        log.info(f"  Total chunks     : {self.total:,}")
        log.info(f"  Enriched         : {self.enriched:,}")
        log.info(f"  Skipped (already): {self.skipped:,}")
        log.info(f"  Failed           : {self.failed}")
        log.info(f"  Time taken       : {elapsed}")
        log.info(f"  Est. API cost    : ~${cost_est:.2f}")
        log.info("=" * 55)
        if self.failed > 0:
            log.warning(f"  {self.failed} chunks failed — check enrich_chunks.log")
        log.info("")
        log.info("  NEXT STEPS:")
        log.info("  1. Clear Redis cache: POST /admin/cache/clear")
        log.info("  2. Test the SME question to verify retrieval improved")
        log.info("=" * 55)


# ── Main enrichment loop ──────────────────────────────────────────────────────

def run_enrichment(
    dry_run:        bool          = False,
    limit:          Optional[int] = None,
    batch_size:     int           = BATCH_SIZE,
    source_filter:  Optional[str] = None,
    skip_embedding: bool          = False,
    force:          bool          = False,
) -> None:

    log.info("=" * 55)
    log.info("  SAMA NORA — Chunk Enrichment v2 (Contextual Headers)")
    log.info("=" * 55)
    log.info(f"  Embedding model : {EMBEDDING_MODEL}")
    log.info(f"  LLM backend     : {LLM_BACKEND}")
    log.info(f"  Dry run         : {dry_run}")
    log.info(f"  Source filter   : {source_filter or 'all'}")
    log.info(f"  Limit           : {limit or 'none (all chunks)'}")
    log.info(f"  Skip embedding  : {skip_embedding}")
    log.info(f"  Force re-enrich : {force}")
    log.info("=" * 55)

    # ── Warm up ───────────────────────────────────────────────────────────────
    log.info("[init] Warming up LLM client...")
    get_llm_client()

    if not skip_embedding:
        log.info("[init] Loading embedding model...")
        get_embedder()

    # ── Load source type map ──────────────────────────────────────────────────
    log.info("[init] Loading document source types from Supabase...")
    source_map = load_source_map()   # document_id → source_type

    # ── Get total count ───────────────────────────────────────────────────────
    total_in_db = get_total_chunk_count()
    total = min(total_in_db, limit) if limit else total_in_db
    log.info(f"[init] Chunks to process: {total:,}")

    if total == 0:
        log.info("[init] No chunks found. Check Supabase connection.")
        return

    # ── Confirm ───────────────────────────────────────────────────────────────
    if not dry_run:
        est_cost = total * 0.0001
        est_hrs  = (total * 2.5) / 3600
        log.info(f"[init] Estimated cost: ~${est_cost:.2f}  |  Time: ~{est_hrs:.1f} hours")
        if source_filter:
            log.info(f"[init] Filtering to source_type = {source_filter}")
        log.info("[init] Starting in 5 seconds... Ctrl+C to cancel")
        time.sleep(5)

    prog      = Progress(total)
    offset    = 0
    processed = 0

    while processed < total:
        remaining_quota    = total - processed
        current_batch_size = min(batch_size, remaining_quota)
        chunks             = fetch_chunk_batch(offset, current_batch_size)

        if not chunks:
            log.info(f"[loop] No more chunks at offset {offset}. Done.")
            break

        for chunk in chunks:
            if processed >= total:
                break

            chunk_id  = chunk["id"]
            doc_id    = chunk.get("document_id", "")
            doc_name  = chunk.get("document_name", "")
            sec_title = chunk.get("section_title")
            content   = chunk.get("content", "")

            # Look up source_type for this chunk's document
            # Infer from document name first (most reliable)
            # DB source_type may be wrong (e.g. NCA docs stored as SAMA)
            inferred    = infer_source_from_name(doc_name)
            source_type = inferred or source_map.get(doc_id, "SAMA")

            processed += 1

            # ── Source filter (applied in Python) ────────────────────────────
            if source_filter and source_type.upper() != source_filter.upper():
                prog.skipped += 1
                continue

            # ── Already enriched? ─────────────────────────────────────────────
            if content.startswith(CONTEXT_PREFIX) and not force:
                prog.skipped += 1
                log.debug(f"[skip] Already enriched: {chunk_id[:12]}")
                continue

            if not content.strip():
                prog.failed += 1
                log.warning(f"[skip] Empty content: {chunk_id[:12]}")
                continue

            # ── Strip old header if force re-enriching ────────────────────────
            raw_content = strip_old_header(content) if force and content.startswith(CONTEXT_PREFIX) else content

            # ── Generate context header ───────────────────────────────────────
            header = generate_context_header(doc_name, sec_title, raw_content, source_type)
            time.sleep(LLM_DELAY)

            if not header:
                prog.failed += 1
                log.warning(f"[fail] No header: {chunk_id[:12]} ({doc_name[:40]})")
                continue

            enriched = enrich_content(raw_content, header)

            # ── Embed ─────────────────────────────────────────────────────────
            if not skip_embedding:
                try:
                    new_embedding = embed_text(enriched)
                except Exception as e:
                    log.warning(f"[fail] Embedding failed for {chunk_id[:12]}: {e}")
                    prog.failed += 1
                    continue
                success = update_chunk(chunk_id, enriched, new_embedding, dry_run)
            else:
                success = update_chunk_content_only(chunk_id, enriched, dry_run)

            if success:
                prog.enriched += 1
                log.debug(
                    f"[ok] {chunk_id[:12]} | [{source_type}] {doc_name[:30]} | "
                    f"{header[:70]}..."
                )
            else:
                prog.failed += 1

            time.sleep(DB_DELAY)

        prog.log_progress()
        offset += len(chunks)
        time.sleep(0.3)

    prog.log_summary()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SAMA NORA — Add contextual headers to chunks (v2)"
    )
    parser.add_argument("--dry-run",        action="store_true",
                        help="Test without writing to DB")
    parser.add_argument("--limit",          type=int, default=None,
                        help="Only process this many chunks")
    parser.add_argument("--batch-size",     type=int, default=BATCH_SIZE,
                        help=f"Chunks per DB fetch (default: {BATCH_SIZE})")
    parser.add_argument("--source",         type=str, default=None,
                        choices=["SAMA", "NCA", "PDPL", "ISO", "SDAIA"],
                        help="Only enrich chunks from this regulatory body")
    parser.add_argument("--skip-embedding", action="store_true",
                        help="Update content only, skip re-embedding")
    parser.add_argument("--force",          action="store_true",
                        help="Re-enrich chunks that already have headers (fixes wrong labels)")
    args = parser.parse_args()

    try:
        run_enrichment(
            dry_run        = args.dry_run,
            limit          = args.limit,
            batch_size     = args.batch_size,
            source_filter  = args.source,
            skip_embedding = args.skip_embedding,
            force          = args.force,
        )
    except KeyboardInterrupt:
        log.info("\n[interrupted] Enrichment stopped. Safe to restart — already-enriched chunks will be skipped.")
        sys.exit(0)


if __name__ == "__main__":
    main()