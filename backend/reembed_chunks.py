"""
reembed_chunks.py — BGE-M3 Re-embedding (Step 5, Parallel Column)
ZetaLabs / IOTA Technologies — SAMA NORA

What it does:
  1. Reads all chunks from sama_nora_chunks where embedding_bge IS NULL
  2. Embeds the (already-enriched, [Context: ...]) content with BAAI/bge-m3
  3. Writes the 1024-dim dense vector to embedding_bge column only
  4. Never touches the existing 384-dim `embedding` column

Idempotent:
  Skips any chunk where embedding_bge is already populated.
  Safe to restart after network drops — picks up where it left off.

Cost: $0 — BGE-M3 runs locally, no API calls.
Time: ~4-5 hours on CPU for 32,655 chunks (slower than e5-small due to
       larger 1024-dim model — BGE-M3 is ~2.2GB).

Usage:
    python reembed_chunks.py                 # re-embed all
    python reembed_chunks.py --limit 10       # test on first 10
    python reembed_chunks.py --dry-run        # no DB writes
    python reembed_chunks.py --batch-size 25  # smaller batches (less memory)

Requires:
    pip install -U FlagEmbedding
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
SUPABASE_URL    = os.environ["SUPABASE_URL"]
SUPABASE_KEY    = (os.environ.get("SUPABASE_KEY") or
                   os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or "")
BGE_MODEL       = os.getenv("BGE_MODEL", "BAAI/bge-m3")
BATCH_SIZE      = 50
DB_DELAY        = 0.1
USE_FP16        = os.getenv("BGE_USE_FP16", "true").lower() == "true"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("reembed_chunks.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Lazy singletons ───────────────────────────────────────────────────────────
_supabase = None
_bge_model = None


def get_supabase():
    global _supabase
    if _supabase is None:
        from supabase import create_client
        _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase


def get_bge_model():
    global _bge_model
    if _bge_model is None:
        from FlagEmbedding import BGEM3FlagModel
        log.info(f"[bge] Loading {BGE_MODEL} (fp16={USE_FP16}) ...")
        _bge_model = BGEM3FlagModel(BGE_MODEL, use_fp16=USE_FP16)
        log.info("[bge] Ready.")
    return _bge_model


def embed_bge(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of texts with BGE-M3, return dense vectors (1024-dim).
    BGE-M3 also produces sparse + multi-vector outputs, but we only
    use the dense vector for the embedding_bge column.
    """
    model = get_bge_model()
    out = model.encode(
        texts,
        batch_size=8,
        max_length=1024,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    dense = out["dense_vecs"]
    return [vec.tolist() for vec in dense]


# ── Supabase helpers ──────────────────────────────────────────────────────────

def get_total_pending_count() -> int:
    """Count chunks where embedding_bge is still NULL."""
    try:
        resp = (
            get_supabase()
            .table("sama_nora_chunks")
            .select("id", count="exact")
            .is_("embedding_bge", "null")
            .execute()
        )
        return resp.count or 0
    except Exception as e:
        log.error(f"[db] Failed to get pending count: {e}")
        return 0


def fetch_pending_batch(limit: int) -> list[dict]:
    """
    Fetch a batch of chunks needing embedding_bge.
    Always fetches from offset 0 since completed rows drop out of the
    WHERE embedding_bge IS NULL filter — no offset drift on restart.
    """
    try:
        resp = (
            get_supabase()
            .table("sama_nora_chunks")
            .select("id, content")
            .is_("embedding_bge", "null")
            .order("id")
            .limit(limit)
            .execute()
        )
        return resp.data or []
    except Exception as e:
        log.error(f"[db] Batch fetch failed: {e}")
        return []


def update_embedding_bge(chunk_id: str, vec: list[float], dry_run: bool = False) -> bool:
    if dry_run:
        return True
    try:
        get_supabase().table("sama_nora_chunks").update({
            "embedding_bge": vec,
        }).eq("id", chunk_id).execute()
        return True
    except Exception as e:
        log.error(f"[db] Update failed for {chunk_id[:12]}: {e}")
        return False


# ── Progress tracker ──────────────────────────────────────────────────────────

class Progress:
    def __init__(self, total: int):
        self.total      = total
        self.done_count = 0
        self.failed     = 0
        self.start_time = time.time()

    def eta(self) -> str:
        if self.done_count == 0:
            return "calculating..."
        elapsed   = time.time() - self.start_time
        rate      = self.done_count / elapsed
        remaining = (self.total - self.done_count) / rate if rate > 0 else 0
        return str(timedelta(seconds=int(remaining)))

    def log_progress(self):
        pct = (self.done_count / self.total * 100) if self.total > 0 else 0
        log.info(
            f"[progress] {self.done_count:,}/{self.total:,} ({pct:.1f}%) — "
            f"failed={self.failed} ETA={self.eta()}"
        )

    def log_summary(self):
        elapsed = timedelta(seconds=int(time.time() - self.start_time))
        log.info("")
        log.info("=" * 55)
        log.info("  BGE-M3 RE-EMBEDDING COMPLETE")
        log.info("=" * 55)
        log.info(f"  Total processed  : {self.done_count:,}")
        log.info(f"  Failed           : {self.failed}")
        log.info(f"  Time taken       : {elapsed}")
        log.info("=" * 55)
        if self.failed > 0:
            log.warning(f"  {self.failed} chunks failed — check reembed_chunks.log")
        log.info("")
        log.info("  NEXT STEPS:")
        log.info("  1. Verify a few rows: SELECT id, embedding_bge FROM")
        log.info("     sama_nora_chunks WHERE embedding_bge IS NOT NULL LIMIT 5;")
        log.info("  2. Test with USE_BGE_COLUMN=true on a staging deployment")
        log.info("  3. Compare benchmark results before promoting the column")
        log.info("=" * 55)


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_reembed(
    dry_run:    bool          = False,
    limit:      Optional[int] = None,
    batch_size: int           = BATCH_SIZE,
) -> None:

    log.info("=" * 55)
    log.info("  SAMA NORA — BGE-M3 Re-embedding (Step 5, parallel column)")
    log.info("=" * 55)
    log.info(f"  Model        : {BGE_MODEL}")
    log.info(f"  Dry run      : {dry_run}")
    log.info(f"  Limit        : {limit or 'none (all pending)'}")
    log.info(f"  Batch size   : {batch_size}")
    log.info("=" * 55)

    log.info("[init] Loading BGE-M3 model (first run downloads ~2.2GB)...")
    get_bge_model()

    total_pending = get_total_pending_count()
    total = min(total_pending, limit) if limit else total_pending
    log.info(f"[init] Chunks pending embedding_bge: {total_pending:,}")
    log.info(f"[init] Will process: {total:,}")

    if total == 0:
        log.info("[init] Nothing to do — all chunks already have embedding_bge.")
        return

    if not dry_run:
        est_hrs = (total * 0.5) / 3600  # rough: ~0.5s/chunk on CPU
        log.info(f"[init] Estimated time: ~{est_hrs:.1f} hours")
        log.info("[init] Starting in 5 seconds... Ctrl+C to cancel")
        time.sleep(5)

    prog = Progress(total)

    while prog.done_count < total:
        remaining = total - prog.done_count
        this_batch = min(batch_size, remaining)
        chunks = fetch_pending_batch(this_batch)

        if not chunks:
            log.info("[loop] No more pending chunks. Done.")
            break

        texts = [c.get("content", "") or "" for c in chunks]
        # Filter out empty content (shouldn't happen but be safe)
        valid = [(c, t) for c, t in zip(chunks, texts) if t.strip()]
        if not valid:
            prog.failed += len(chunks)
            continue

        try:
            vecs = embed_bge([t for _, t in valid])
        except Exception as e:
            log.error(f"[bge] Embedding batch failed: {e}")
            prog.failed += len(valid)
            prog.done_count += len(chunks)
            prog.log_progress()
            continue

        for (chunk, _), vec in zip(valid, vecs):
            success = update_embedding_bge(chunk["id"], vec, dry_run)
            if not success:
                prog.failed += 1
            time.sleep(DB_DELAY)

        prog.done_count += len(chunks)
        prog.log_progress()
        time.sleep(0.2)

    prog.log_summary()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SAMA NORA — BGE-M3 re-embedding into embedding_bge column"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Test without writing to DB")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process this many chunks")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Chunks per DB fetch (default: {BATCH_SIZE})")
    args = parser.parse_args()

    try:
        run_reembed(
            dry_run    = args.dry_run,
            limit      = args.limit,
            batch_size = args.batch_size,
        )
    except KeyboardInterrupt:
        log.info("\n[interrupted] Stopped. Safe to restart — pending rows are unchanged.")
        sys.exit(0)


if __name__ == "__main__":
    main()