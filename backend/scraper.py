"""
scraper.py — SAMA & NCA PDF Scraper & Ingestion Pipeline
ZetaLabs / IOTA Technologies — SAMA NORA RAG Chatbot

Focused on SAMA and NCA only — 6 workers total (3 per site).
Each PDF is processed immediately on discovery:
  find → download → extract → chunk → embed → push to DB

Extraction — ALL 6 extractors run, best result wins (scored 0-100):
  1. pdfplumber         accurate char positions, word-level fallback
  2. pdfminer-strict    2-column layouts, tight LAParams (line_margin=0.3)
  3. pdfminer-loose     single-column docs, relaxed LAParams (line_margin=0.7)
  4. pymupdf-blocks     fast, paragraph-block mode, widest PDF version support
  5. pymupdf-dict       span-level, richer structure for flowing text
  6. pypdf              newer PDF spec, different cross-reference parser

  Winner chosen by 4-dimension quality score:
    Coverage 40% + Lexical richness 25% + Paragraph structure 20% + Printable 15%

  Section titles detected and stored in chunk["section_title"] for citations.

Install:
    pip install pdfplumber pdfminer.six pymupdf pypdf sentence-transformers
    pip install supabase "scrapling[fetchers]" python-dotenv requests
    scrapling install

Run:
    python scraper.py                         # 6 workers (3 SAMA + 3 NCA)
    python scraper.py --dry-run               # no DB writes
    python scraper.py --validate-only         # list PDFs only, no download
    python scraper.py --file doc.pdf --name "My Doc" --source SAMA
    python scraper.py --chunk-size 350        # override chunk size
    python scraper.py --chunk-overlap 3       # override overlap sentences
"""

from __future__ import annotations

import os
import re
import sys
import time
import uuid
import hashlib
import logging
import argparse
import threading
from pathlib import Path
from urllib.parse import urljoin, urlparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
SUPABASE_URL    = os.environ["SUPABASE_URL"]
SUPABASE_KEY    = (os.environ.get("SUPABASE_KEY") or
                   os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")

def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key, str(default)).split("#")[0].strip()
    try:    return int(raw)
    except: return default

def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key, str(default)).split("#")[0].strip()
    try:    return float(raw)
    except: return default

def _env_bool(key: str, default: bool) -> bool:
    return os.getenv(key, str(default)).split("#")[0].strip().lower() in ("true","1","yes")

CHUNK_SIZE         = _env_int("CHUNK_SIZE",        400)
MIN_CHUNK_CHARS    = _env_int("MIN_CHUNK_CHARS",    80)
CHUNK_OVERLAP      = _env_int("CHUNK_OVERLAP",      3)
BATCH_SIZE         = _env_int("BATCH_SIZE",         25)
REQUEST_DELAY      = _env_float("REQUEST_DELAY",    1.5)
MAX_DEPTH          = _env_int("MAX_DEPTH",           3)
MAX_PAGES_PER_SITE = _env_int("MAX_PAGES_PER_SITE", 300)
BASE_DOWNLOAD_DIR  = Path(os.getenv("DOWNLOAD_DIR", "./downloaded_pdfs").split("#")[0].strip())
DRY_RUN            = _env_bool("DRY_RUN",           False)
STEALTH_MODE       = _env_bool("STEALTH_MODE",       True)
WORKERS_PER_SITE   = 3   # 3 SAMA + 3 NCA = 6 total

BASE_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("scraper.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Source catalogue — SAMA and NCA only
# ─────────────────────────────────────────────────────────────────────────────
SOURCES: list[dict] = [
    {
        "root_url":         "https://www.sama.gov.sa/en-US/RulesInstructions/Pages/default.aspx",
        "label":            "SAMA",
        "source_type":      "SAMA",
        "stay_on_domain":   True,
        "url_must_contain": [
            "RulesInstructions", "regulation", "circular", "guideline",
            "framework", "instruction", "policy", "cybersecurity",
            "Prudential", "Basel", "Capital", "Liquidity", "Risk",
            "Payment", "Consumer", "Disclosure", "IFRS",
        ],
    },
    {
        "root_url":         "https://nca.gov.sa/en/",
        "label":            "NCA",
        "source_type":      "NCA",
        "stay_on_domain":   True,
        "url_must_contain": [
            "controls", "framework", "cybersecurity", "standard",
            "ecc", "ccc", "otcc", "cloud", "guideline", "policy",
            "regulation", "document", "publication", "resource",
        ],
    },
]

# 3 entry points per site — each worker starts from a different sub-section
# so they crawl the site in parallel without duplicating work
SAMA_ENTRY_POINTS = [
    "https://www.sama.gov.sa/en-US/RulesInstructions/Pages/default.aspx",
    "https://www.sama.gov.sa/en-US/RulesInstructions/BankingRules/Pages/default.aspx",
    "https://www.sama.gov.sa/en-US/RulesInstructions/CyberSecurity/",
]

NCA_ENTRY_POINTS = [
    "https://nca.gov.sa/en/",
    "https://nca.gov.sa/en/cybersecurity-documents",
    "https://nca.gov.sa/en/regulations-standards",
]

# ─────────────────────────────────────────────────────────────────────────────
# Thread-safe shared state
# ─────────────────────────────────────────────────────────────────────────────
_supabase       = None
_embedder       = None
_embed_lock     = threading.Lock()
_stealthy_lock  = threading.Lock()
_db_lock        = threading.Lock()
_db_sem         = threading.Semaphore(4)
DB_RETRY_DELAY  = 2.0
DB_MAX_RETRIES  = 3

# Per-site seen_pdfs sets shared across workers for the same site so that
# SAMA-1, SAMA-2, SAMA-3 never ingest the same PDF twice
_seen_pdfs_lock = threading.Lock()
_seen_pdfs: dict[str, set[str]] = {}

def _site_seen(label: str, url: str) -> bool:
    """Return True if url was already seen for this site label; register if not."""
    with _seen_pdfs_lock:
        if label not in _seen_pdfs:
            _seen_pdfs[label] = set()
        if url in _seen_pdfs[label]:
            return True
        _seen_pdfs[label].add(url)
        return False

def _db_call(fn, *args, **kwargs):
    for attempt in range(1, DB_MAX_RETRIES + 1):
        with _db_sem:
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                err = str(e).lower()
                is_transient = any(x in err for x in [
                    "server disconnected", "connection", "timeout",
                    "remotedisconnected", "reset", "eof",
                ])
                if is_transient and attempt < DB_MAX_RETRIES:
                    wait = DB_RETRY_DELAY * attempt
                    log.warning(f"[DB] transient error attempt {attempt}/{DB_MAX_RETRIES}, retry in {wait:.1f}s: {e}")
                    time.sleep(wait)
                    continue
                raise

def get_supabase():
    global _supabase
    if _supabase is None:
        from supabase import create_client
        _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase

def get_embedder():
    global _embedder
    with _embed_lock:
        if _embedder is None:
            from sentence_transformers import SentenceTransformer
            log.info(f"[embedder] Loading {EMBEDDING_MODEL} ...")
            _embedder = SentenceTransformer(EMBEDDING_MODEL)
            log.info("[embedder] Ready.")
    return _embedder

def embed_text(text: str) -> list[float]:
    model = get_embedder()
    prefix = "passage: " if "e5" in EMBEDDING_MODEL.lower() else ""
    with _embed_lock:
        vec = model.encode(prefix + text, normalize_embeddings=True)
    return vec.tolist()

# ─────────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ─────────────────────────────────────────────────────────────────────────────
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36"
    )
}

def _make_download_headers(url: str) -> dict:
    parsed = urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    return {
        "User-Agent":                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept":                    "application/pdf,application/octet-stream,*/*;q=0.8",
        "Accept-Language":           "en-US,en;q=0.9,ar;q=0.8",
        "Accept-Encoding":           "gzip, deflate, br",
        "Referer":                   origin + "/",
        "Origin":                    origin,
        "Connection":                "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest":            "document",
        "Sec-Fetch-Mode":            "navigate",
        "Sec-Fetch-Site":            "same-origin",
        "Sec-Fetch-User":            "?1",
        "Cache-Control":             "no-cache",
        "Pragma":                    "no-cache",
        "DNT":                       "1",
    }

def fetch_page_html(url: str, site: str) -> Optional[str]:
    url = url.replace(" ", "%20")
    try:
        if STEALTH_MODE:
            from scrapling.fetchers import StealthyFetcher
            with _stealthy_lock:
                page = StealthyFetcher.fetch(
                    url, headless=True, network_idle=True,
                    timeout=30000, wait_for_idle=True,
                )
            return page.html_content if page else None
        else:
            from scrapling.fetchers import Fetcher
            page = Fetcher.get(url, timeout=20, stealthy_headers=True)
            return page.html_content if page else None
    except Exception as e:
        log.warning(f"[{site}] scrapling failed: {e}")
        try:
            r = requests.get(url, headers=_HEADERS, timeout=20, allow_redirects=True)
            r.raise_for_status()
            return r.text
        except Exception as e2:
            log.warning(f"[{site}] requests fallback failed: {e2}")
            return None

def download_pdf_bytes(url: str, site: str) -> Optional[bytes]:
    url     = url.replace(" ", "%20")
    headers = _make_download_headers(url)
    parsed  = urlparse(url)
    origin  = f"{parsed.scheme}://{parsed.netloc}"

    for attempt in range(1, 4):
        try:
            session = requests.Session()
            session.headers.update(headers)
            if attempt == 1:
                try:
                    session.get(origin, timeout=15, allow_redirects=True)
                    time.sleep(1.0)
                except Exception:
                    pass
            r = session.get(url, timeout=240, stream=True, allow_redirects=True)
            r.raise_for_status()
            ct = r.headers.get("Content-Type", "").lower()
            if "text/html" in ct and not url.lower().endswith(".pdf"):
                log.warning(f"[{site}] got HTML instead of PDF: {url}")
                return None
            data = b"".join(r.iter_content(chunk_size=65536))
            if len(data) > 1024:
                return data
            log.warning(f"[{site}] download too small ({len(data)} bytes)")
        except Exception as e:
            log.warning(f"[{site}] download attempt {attempt}/3 failed: {e}")
            if attempt < 3:
                time.sleep(15 * attempt)
    return None

# ─────────────────────────────────────────────────────────────────────────────
# URL / name helpers
# ─────────────────────────────────────────────────────────────────────────────

def is_pdf_url(url: str) -> bool:
    return url.lower().split("?")[0].rstrip("/").endswith(".pdf")

def same_domain(url: str, root: str) -> bool:
    return urlparse(url).netloc == urlparse(root).netloc

def site_download_dir(source_label: str) -> Path:
    d = BASE_DOWNLOAD_DIR / re.sub(r"[^\w]", "_", source_label)
    d.mkdir(parents=True, exist_ok=True)
    return d

def url_to_local_path(url: str, source_label: str) -> Path:
    h    = hashlib.md5(url.encode()).hexdigest()[:8]
    stem = Path(urlparse(url.replace(" ", "%20")).path).name or "document"
    stem = re.sub(r"[^\w\-.]", "_", stem)[:80]
    return site_download_dir(source_label) / f"{stem}_{h}.pdf"

_GENERIC_LABELS = {
    "download", "view", "click here", "here", "pdf", "file",
    "to view", "english", "arabic", "read more", "more",
    "open", "get", "access", "link", "document", "attachment",
    "download instructions", "download pdf", "view pdf",
    "click to download", "click to view", "download file",
    "download document", "download here", "view document",
    "عربي", "تحميل", "عرض", "اضغط هنا", "هنا",
}

_GENERIC_PATTERNS = [
    r"^click here",
    r"^part \d+",
    r"^\- ",
    r"\(pdf,?\s*[\d.]+\s*[km]b\)",
    r"^\d+\s*[km]b$",
]

def clean_name(label: str, url: str, fallback_prefix: str = "") -> str:
    stripped = label.strip() if label else ""
    stripped = re.sub(r"\s*\(pdf,?\s*[\d.]+\s*[km]b\)\s*$", "", stripped, flags=re.I).strip()
    is_generic = (
        not stripped
        or len(stripped) <= 4
        or stripped.lower() in _GENERIC_LABELS
        or re.fullmatch(r"[\W\d]+", stripped)
        or any(re.search(p, stripped, re.I) for p in _GENERIC_PATTERNS)
    )
    if not is_generic:
        return re.sub(r"\s+", " ", stripped)[:200]
    stem = Path(urlparse(url.replace("%20", "_")).path).stem
    stem = re.sub(r"[_%\-]+", " ", stem).strip()
    stem = re.sub(r"\s+", " ", stem).strip()
    if not stem or len(stem) < 3:
        stem = url.split("/")[-1].split("?")[0][:80]
    prefix = f"{fallback_prefix} - " if fallback_prefix else ""
    return (prefix + stem)[:200]

# ─────────────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────────────

def document_exists(document_name: str) -> Optional[str]:
    try:
        resp = _db_call(
            lambda: get_supabase().table("documents")
            .select("id").eq("document_name", document_name).limit(1).execute()
        )
        return resp.data[0]["id"] if resp.data else None
    except Exception as e:
        log.warning(f"[DB] document_exists failed: {e}")
    return None

def chunks_exist(document_id: str) -> bool:
    try:
        resp = _db_call(
            lambda: get_supabase().table("sama_nora_chunks")
            .select("id").eq("document_id", document_id).limit(1).execute()
        )
        return bool(resp.data)
    except Exception:
        return False

def upsert_document(document_name: str, source_type: str, total_pages: int) -> str:
    with _db_lock:
        existing = document_exists(document_name)
        if existing:
            return existing
        doc_id = str(uuid.uuid4())
        if not DRY_RUN:
            _db_call(
                lambda: get_supabase().table("documents").insert({
                    "id":            doc_id,
                    "document_name": document_name,
                    "document_code": None,
                    "source_type":   source_type,
                    "total_pages":   str(total_pages),
                }).execute()
            )
        return doc_id

def insert_chunks(document_id: str, chunks: list[dict], site: str) -> int:
    """Upsert chunks — silently skips UUIDs already in DB (no duplicate key crash)."""
    if not chunks:
        return 0
    inserted = 0
    batch: list[dict] = []

    for chunk in chunks:
        try:
            embedding = embed_text(chunk["content"])
        except Exception as e:
            log.warning(f"[{site}] embedding failed: {e}")
            continue
        batch.append({
            "id":            str(uuid.uuid4()),
            "document_id":   document_id,
            "document_name": chunk["document_name"],
            "page_start":    chunk["page_start"],
            "page_end":      chunk["page_end"],
            "section_title": chunk["section_title"],
            "content":       chunk["content"],
            "embedding":     embedding,
            "token_count":   chunk["token_count"],
            "language":      chunk["language"],
        })
        if len(batch) >= BATCH_SIZE:
            if not DRY_RUN:
                _db_call(
                    lambda b=batch: get_supabase()
                    .table("sama_nora_chunks")
                    .upsert(b, on_conflict="id", ignore_duplicates=True)
                    .execute()
                )
            inserted += len(batch)
            batch = []
            time.sleep(0.3)

    if batch:
        if not DRY_RUN:
            _db_call(
                lambda b=batch: get_supabase()
                .table("sama_nora_chunks")
                .upsert(b, on_conflict="id", ignore_duplicates=True)
                .execute()
            )
        inserted += len(batch)

    return inserted

# ─────────────────────────────────────────────────────────────────────────────
# PDF Extraction — layered pipeline
#
# 1. pdfplumber  — primary. Uses PDF rendering pipeline for accurate character
#    positions. Best for clean digital PDFs. Also tries word-level extraction
#    when page-level text is empty.
#
# 2. pdfminer    — first fallback. LAParams tuned for regulatory docs:
#    line_margin=0.3 prevents column bleeding, char_margin=2.0 handles wider
#    character spacing. Better reading order on 2-column layouts.
#
# 3. pymupdf     — second fallback. Widest PDF version support. Uses "blocks"
#    mode which preserves paragraph structure.
#
# Section title detection scans the first 15 lines of each page for:
#    - ALL CAPS lines (≤80 chars) — classic regulatory heading
#    - Numbered headings: "1.1", "Article 3", "Section 2"
#    - Short title-case lines (≤60 chars, ≥70% capitalised words)
# ─────────────────────────────────────────────────────────────────────────────

def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)

def _detect_language(text: str) -> str:
    arabic = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
    return "ar" if arabic > len(text) * 0.25 else "en"

def _detect_section_title(lines: list[str]) -> Optional[str]:
    candidates = []
    for line in lines[:15]:
        line = line.strip()
        if not line or len(line) < 3:
            continue
        if line.isupper() and 5 < len(line) <= 80:
            candidates.append((0, line))
            continue
        if re.match(r"^(\d+\.)+\s+\w|^Article\s+\d+|^Section\s+\d+|^Chapter\s+\d+", line, re.I):
            candidates.append((1, line))
            continue
        if len(line) <= 60 and line[0].isupper() and not line.endswith("."):
            words = line.split()
            if len(words) >= 2 and sum(1 for w in words if w[0].isupper()) / len(words) >= 0.7:
                candidates.append((2, line))
    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1][:120]
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Individual extractors — each returns list[{page, text, total_pages}] or []
#
# Extractor  | Strengths                                    | Weaknesses
# -----------|----------------------------------------------|----------------------
# pdfplumber | Accurate char positions, good for tables     | Slow on large files,
#            | word-level fallback for sparse pages          | misorders 2-col layouts
# pdfminer   | Best reading-order on 2-column layouts,      | Slow, verbose output
#            | LAParams tuned for regulatory docs            | on some Arabic PDFs
# pymupdf    | Fastest, widest version support,             | Loses some formatting
#  (blocks)  | paragraph-aware block extraction             | in complex layouts
# pymupdf    | Better for flowing text — respects line       | Less paragraph structure
#  (dict)    | spans and font sizes for heading detection    |
# pypdf2     | Good for newer PDF spec files,               | Fails on old/encrypted
#            | handles cross-reference tables well          | PDFs, no positional info
# pdfminer   | Alternative LAParams tuning (loose) —        | May merge 2-col text
#  (loose)   | better for single-column dense text docs     |
# ─────────────────────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Normalise whitespace and remove garbage characters common in PDF extraction."""
    # Remove null bytes and control characters except newlines/tabs
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Collapse excessive whitespace
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove lines that are pure noise (lone punctuation, single chars repeated)
    lines = [l for l in text.split("\n") if not re.fullmatch(r"[^\w\u0600-\u06FF]{1,3}", l.strip())]
    return "\n".join(lines).strip()


def _extract_with_pdfplumber(pdf_path: Path, site: str) -> list[dict]:
    """
    Primary extractor. Uses PDF rendering pipeline for accurate character
    positions. Falls back to word-level extraction for sparse/image-heavy pages.
    Good for: clean digital regulatory PDFs, tables.
    """
    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            total = len(pdf.pages)
            for i, page in enumerate(pdf.pages, 1):
                text = (page.extract_text() or "").strip()
                if not text:
                    # Word-level fallback — sorts words by position
                    words = page.extract_words(x_tolerance=3, y_tolerance=3,
                                               keep_blank_chars=False)
                    if words:
                        text = " ".join(w["text"] for w in words)
                text = _clean_text(text)
                if len(text) >= 30:
                    pages.append({"page": i, "text": text, "total_pages": total})
        return pages
    except Exception as e:
        log.debug(f"[{site}] pdfplumber: {e}")
        return []


def _extract_with_pdfminer_strict(pdf_path: Path, site: str) -> list[dict]:
    """
    pdfminer with strict LAParams tuned for multi-column regulatory documents.
    line_margin=0.3 prevents text from adjacent columns bleeding into each other.
    detect_vertical=True handles rotated text in tables and annexes.
    Good for: 2-column layouts (common in SAMA/NCA documents).
    """
    try:
        from pdfminer.high_level import extract_pages as pm_extract
        from pdfminer.layout import LTTextContainer, LAParams

        laparams = LAParams(
            line_margin=0.3,
            char_margin=2.0,
            word_margin=0.1,
            boxes_flow=0.5,
            detect_vertical=True,
        )
        page_list = list(pm_extract(str(pdf_path), laparams=laparams))
        total = len(page_list)
        pages = []
        for i, layout in enumerate(page_list, 1):
            parts = [e.get_text().strip()
                     for e in layout if isinstance(e, LTTextContainer)]
            text = _clean_text("\n".join(p for p in parts if p))
            if len(text) >= 30:
                pages.append({"page": i, "text": text, "total_pages": total})
        return pages
    except Exception as e:
        log.debug(f"[{site}] pdfminer-strict: {e}")
        return []


def _extract_with_pdfminer_loose(pdf_path: Path, site: str) -> list[dict]:
    """
    pdfminer with loose LAParams — higher line_margin and char_margin.
    Better for single-column dense text documents where strict params
    fragment paragraphs unnecessarily.
    Good for: SAMA circulars, guidelines, single-column policy documents.
    """
    try:
        from pdfminer.high_level import extract_pages as pm_extract
        from pdfminer.layout import LTTextContainer, LAParams

        laparams = LAParams(
            line_margin=0.7,
            char_margin=3.5,
            word_margin=0.2,
            boxes_flow=0.3,
            detect_vertical=False,
        )
        page_list = list(pm_extract(str(pdf_path), laparams=laparams))
        total = len(page_list)
        pages = []
        for i, layout in enumerate(page_list, 1):
            parts = [e.get_text().strip()
                     for e in layout if isinstance(e, LTTextContainer)]
            text = _clean_text("\n".join(p for p in parts if p))
            if len(text) >= 30:
                pages.append({"page": i, "text": text, "total_pages": total})
        return pages
    except Exception as e:
        log.debug(f"[{site}] pdfminer-loose: {e}")
        return []


def _extract_with_pymupdf_blocks(pdf_path: Path, site: str) -> list[dict]:
    """
    PyMuPDF in 'blocks' mode — extracts text as paragraph blocks sorted by
    position. Fastest extractor, handles widest PDF version range including
    older and compressed files. Blocks mode preserves paragraph boundaries.
    Good for: large documents, older PDF formats, compressed files.
    """
    try:
        import fitz
        doc   = fitz.open(str(pdf_path))
        total = len(doc)
        pages = []
        for i, page in enumerate(doc, 1):
            blocks = page.get_text("blocks", sort=True)
            parts  = [b[4].strip() for b in blocks if b[6] == 0 and b[4].strip()]
            text   = _clean_text("\n".join(parts))
            if len(text) >= 30:
                pages.append({"page": i, "text": text, "total_pages": total})
        doc.close()
        return pages
    except Exception as e:
        log.debug(f"[{site}] pymupdf-blocks: {e}")
        return []


def _extract_with_pymupdf_dict(pdf_path: Path, site: str) -> list[dict]:
    """
    PyMuPDF in 'dict' mode — extracts full span information including font
    names, sizes, and flags (bold/italic). Better for flowing text documents
    and enables font-size-based section title detection. Slower than blocks mode
    but produces richer text with proper line grouping.
    Good for: text-heavy regulatory documents, Arabic+English mixed content.
    """
    try:
        import fitz
        doc   = fitz.open(str(pdf_path))
        total = len(doc)
        pages = []
        for i, page in enumerate(doc, 1):
            data  = page.get_text("dict", sort=True)
            lines_out: list[str] = []
            for block in data.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    line_text = " ".join(s["text"] for s in spans if s["text"].strip())
                    if line_text.strip():
                        lines_out.append(line_text.strip())
            text = _clean_text("\n".join(lines_out))
            if len(text) >= 30:
                pages.append({"page": i, "text": text, "total_pages": total})
        doc.close()
        return pages
    except Exception as e:
        log.debug(f"[{site}] pymupdf-dict: {e}")
        return []


def _extract_with_pypdf(pdf_path: Path, site: str) -> list[dict]:
    """
    pypdf (formerly PyPDF2) — good for newer PDF spec files. Uses a different
    parser than pdfplumber/pdfminer that handles cross-reference tables and
    object streams differently. Often succeeds on files that cause the others
    to produce garbled output.
    Good for: newer PDFs, encrypted-then-decrypted files, forms.
    Install: pip install pypdf
    """
    try:
        import pypdf
        reader = pypdf.PdfReader(str(pdf_path))
        total  = len(reader.pages)
        pages  = []
        for i, page in enumerate(reader.pages, 1):
            text = (page.extract_text() or "").strip()
            text = _clean_text(text)
            if len(text) >= 30:
                pages.append({"page": i + 1, "text": text, "total_pages": total})
        return pages
    except Exception as e:
        log.debug(f"[{site}] pypdf: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Quality scorer — used to pick the BEST result across all extractors
#
# Runs all extractors that produce output, scores each on 4 dimensions, and
# returns the highest-scoring result. Never falls back to a worse result just
# because it ran later.
#
# Scoring dimensions (all normalised 0-1, weighted sum → final score 0-100):
#
#   1. Coverage (weight 40)
#      Total characters extracted relative to the best-performing extractor.
#      Penalises extractors that miss large portions of the document.
#
#   2. Lexical richness (weight 25)
#      Average unique-word ratio per page: unique_words / total_words.
#      Low ratio = repetitive/garbled output (common when two columns merge).
#      Target: ≥ 0.4 for regulatory prose.
#
#   3. Paragraph structure (weight 20)
#      Ratio of pages that contain at least one paragraph break (\n\n).
#      Wall-of-text output (no breaks) usually means columns were merged.
#
#   4. Garbage penalty (weight 15)
#      Fraction of characters that are printable (not control chars, not
#      excessive punctuation runs). High garbage = extraction artefacts.
# ─────────────────────────────────────────────────────────────────────────────

def _score_extraction(pages: list[dict]) -> float:
    """
    Score an extraction result on 4 quality dimensions.
    Returns a float 0–100. Higher is better.
    """
    if not pages:
        return 0.0

    all_text = " ".join(p["text"] for p in pages)
    total_chars = len(all_text)
    if total_chars == 0:
        return 0.0

    # 1. Coverage — raw character count (normalised later against peers)
    coverage_raw = total_chars

    # 2. Lexical richness — average unique-word ratio across pages
    richness_scores = []
    for p in pages:
        words = re.findall(r"[\w\u0600-\u06FF]+", p["text"].lower())
        if len(words) >= 10:
            richness_scores.append(len(set(words)) / len(words))
    lexical_richness = sum(richness_scores) / len(richness_scores) if richness_scores else 0.3

    # 3. Paragraph structure — fraction of pages with at least one \n\n break
    pages_with_breaks = sum(1 for p in pages if "\n\n" in p["text"])
    structure_score   = pages_with_breaks / len(pages)

    # 4. Garbage penalty — fraction of printable chars
    printable = sum(1 for c in all_text if c.isprintable() or c in "\n\t")
    garbage_score = printable / total_chars

    # Weighted sum (coverage_raw is normalised to 1.0 at call site)
    # We return a partial score here; coverage weight applied in caller
    partial = (
        lexical_richness  * 25 +
        structure_score   * 20 +
        garbage_score     * 15
    )
    return partial, coverage_raw   # type: ignore[return-value]


def extract_pages(pdf_path: Path, site: str) -> list[dict]:
    """
    Run ALL 6 extractors, score each result on 4 quality dimensions, and
    return the highest-scoring output.

    Extractors tried:
      1. pdfplumber         (accurate char positions, word fallback)
      2. pdfminer-strict    (2-column layout, tight LAParams)
      3. pdfminer-loose     (single-column, relaxed LAParams)
      4. pymupdf-blocks     (fast, paragraph blocks)
      5. pymupdf-dict       (span-level, richer structure)
      6. pypdf              (newer PDF spec, cross-ref parser)

    Scoring (0-100, higher = better):
      Coverage        40%  — total characters extracted
      Lexical richness 25% — unique-word ratio (catches garbled/merged text)
      Paragraph breaks 20% — pages with proper paragraph structure
      Printable ratio  15% — fraction of non-garbage characters

    The winner is logged with its score so you can audit decisions.
    """
    EXTRACTORS = [
        ("pdfplumber",      _extract_with_pdfplumber),
        ("pdfminer-strict", _extract_with_pdfminer_strict),
        ("pdfminer-loose",  _extract_with_pdfminer_loose),
        ("pymupdf-blocks",  _extract_with_pymupdf_blocks),
        ("pymupdf-dict",    _extract_with_pymupdf_dict),
        ("pypdf",           _extract_with_pypdf),
    ]

    candidates: list[tuple[str, list[dict], float, int]] = []
    # (name, pages, partial_score, coverage_raw)

    for name, fn in EXTRACTORS:
        try:
            pages = fn(pdf_path, site)
            if not pages:
                log.debug(f"[{site}] │         {name}: empty")
                continue
            partial, coverage_raw = _score_extraction(pages)  # type: ignore[misc]
            candidates.append((name, pages, partial, coverage_raw))
            total_chars = sum(len(p["text"]) for p in pages)
            log.debug(f"[{site}] │         {name}: {len(pages)} pages, {total_chars:,} chars, partial={partial:.1f}")
        except Exception as e:
            log.debug(f"[{site}] │         {name} crashed: {e}")

    if not candidates:
        log.warning(f"[{site}] │ EXTRACT  all 6 extractors failed — likely scanned/image PDF")
        return []

    # Normalise coverage against the best-performing extractor
    max_coverage = max(c[3] for c in candidates) or 1
    scored: list[tuple[float, str, list[dict]]] = []
    for name, pages, partial, coverage_raw in candidates:
        coverage_score = (coverage_raw / max_coverage) * 40
        final_score    = partial + coverage_score   # 0-100
        scored.append((final_score, name, pages))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_name, best_pages = scored[0]

    total_chars = sum(len(p["text"]) for p in best_pages)
    log.info(
        f"[{site}] │ EXTRACT  winner={best_name} "
        f"score={best_score:.1f}/100 "
        f"pages={len(best_pages)} "
        f"chars={total_chars:,}"
    )

    # Log runner-up if close (within 5 points) so operator can audit
    if len(scored) > 1:
        runner_score, runner_name, _ = scored[1]
        if best_score - runner_score <= 5.0:
            log.info(
                f"[{site}] │         runner-up={runner_name} "
                f"score={runner_score:.1f} (close call)"
            )

    return best_pages

# ─────────────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────────────

def chunk_page(text: str, page_num: int, document_name: str) -> list[dict]:
    lines    = text.split("\n")
    section  = _detect_section_title(lines)
    raw      = re.split(r"(?<=[.!?؟\n])\s+|\n{2,}", text)
    sentences = [s.strip() for s in raw if len(s.strip()) > 10]

    chunks:         list[dict] = []
    current_sents:  list[str]  = []
    current_tokens: int        = 0

    def flush():
        chunk_text = " ".join(current_sents).strip()
        if len(chunk_text) >= MIN_CHUNK_CHARS:
            chunks.append({
                "document_name": document_name,
                "page_start":    page_num,
                "page_end":      page_num,
                "section_title": section,
                "content":       chunk_text,
                "token_count":   current_tokens,
                "language":      _detect_language(chunk_text),
            })

    for sent in sentences:
        t = _approx_tokens(sent)
        if current_tokens + t > CHUNK_SIZE and current_sents:
            flush()
            overlap        = current_sents[-CHUNK_OVERLAP:] if CHUNK_OVERLAP > 0 else []
            current_sents  = overlap[:]
            current_tokens = sum(_approx_tokens(s) for s in current_sents)
        current_sents.append(sent)
        current_tokens += t
    if current_sents:
        flush()
    return chunks

# ─────────────────────────────────────────────────────────────────────────────
# Core ingest pipeline
# ─────────────────────────────────────────────────────────────────────────────

def ingest_pdf_immediately(
    url:         str,
    label:       str,
    source_type: str,
    site:        str,
) -> dict:
    doc_name = label
    dest     = url_to_local_path(url, site.split("-")[0])   # SAMA-1 → SAMA folder

    log.info(f"[{site}] ┌ START    {doc_name[:65]}")

    existing_id = document_exists(doc_name)
    if existing_id and chunks_exist(existing_id):
        log.info(f"[{site}] │ SKIP     already in DB")
        return {"document_name": doc_name, "status": "skipped", "url": url}

    if dest.exists() and dest.stat().st_size > 2048:
        log.info(f"[{site}] │ CACHED   {dest.name} ({dest.stat().st_size // 1024} KB)")
    else:
        log.info(f"[{site}] │ DOWNLOAD → {dest.parent.name}/{dest.name}")
        data = download_pdf_bytes(url, site)
        if not data:
            log.warning(f"[{site}] └ FAIL     download failed after 3 attempts")
            return {"document_name": doc_name, "status": "download_failed", "url": url}
        dest.write_bytes(data)
        size_kb = dest.stat().st_size // 1024
        if size_kb < 2:
            log.warning(f"[{site}] └ FAIL     file too small ({size_kb} KB)")
            dest.unlink(missing_ok=True)
            return {"document_name": doc_name, "status": "download_failed", "url": url}
        log.info(f"[{site}] │ SAVED    {size_kb} KB")

    pages = extract_pages(dest, site)
    if not pages:
        return {"document_name": doc_name, "status": "no_text", "url": url}
    total_pages = pages[0]["total_pages"]
    log.info(f"[{site}] │         {total_pages} total pages, {len(pages)} non-blank")

    all_chunks: list[dict] = []
    for p in pages:
        all_chunks.extend(chunk_page(p["text"], p["page"], doc_name))
    log.info(f"[{site}] │ CHUNK    {len(all_chunks)} chunks produced")

    doc_id = upsert_document(doc_name, source_type, total_pages)
    log.info(f"[{site}] │ DB DOC   {doc_id[:12]}...")
    log.info(f"[{site}] │ EMBED    {len(all_chunks)} chunks ...")
    inserted = insert_chunks(doc_id, all_chunks, site)
    log.info(f"[{site}] └ DONE     {inserted}/{len(all_chunks)} chunks → DB")

    return {
        "document_name":   doc_name,
        "status":          "ok",
        "url":             url,
        "source_type":     source_type,
        "total_pages":     total_pages,
        "chunks_total":    len(all_chunks),
        "chunks_inserted": inserted,
        "document_id":     doc_id,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Skip patterns
# ─────────────────────────────────────────────────────────────────────────────
SKIP_PATTERNS = [
    r"/news/", r"/media/", r"/events/", r"/careers/",
    r"/contact", r"\.jpg$", r"\.png$", r"\.gif$",
    r"\.mp4$", r"\.zip$", r"\.xlsx$", r"\.docx$",
    # NCA non-document pages
    r"nca\.gov\.sa/ar/",
    r"nca\.gov\.sa/en/search",
    r"nca\.gov\.sa/en/board",
    r"nca\.gov\.sa/en/vision",
    r"nca\.gov\.sa/en/nca-achiev",
    # SAMA non-document pages
    r"sama\.gov\.sa/en-US/About",
    r"sama\.gov\.sa/en-US/MediaCenter",
    r"sama\.gov\.sa/en-US/Careers",
    r"sama\.gov\.sa/en-US/EconomicReports",
    r"sama\.gov\.sa/en-US/Statistics",
]

# ─────────────────────────────────────────────────────────────────────────────
# BFS crawler
# ─────────────────────────────────────────────────────────────────────────────

def crawl_and_ingest(source: dict, validate_only: bool = False) -> list[dict]:
    root_url     = source["root_url"]
    must_contain = source.get("url_must_contain", [])
    source_label = source["label"]
    source_type  = source["source_type"]
    site         = source.get("worker_id", source_label)

    visited_pages: set[str] = set()
    results:       list[dict] = []
    queue: deque[tuple[str, int]] = deque([(root_url, 0)])

    log.info(f"[{site}] {'='*48}")
    log.info(f"[{site}] root: {root_url}")
    log.info(f"[{site}] max depth={MAX_DEPTH}  max pages={MAX_PAGES_PER_SITE}")

    while queue and len(visited_pages) < MAX_PAGES_PER_SITE:
        current_url, depth = queue.popleft()
        current_url = current_url.replace(" ", "%20")

        if current_url in visited_pages:
            continue

        if is_pdf_url(current_url):
            if not _site_seen(source_label, current_url):
                label = clean_name("", current_url, source_label)
                if validate_only:
                    results.append({"url": current_url, "label": label, "status": "found"})
                else:
                    results.append(ingest_pdf_immediately(current_url, label, source_type, site))
            continue

        visited_pages.add(current_url)
        log.info(f"[{site}] [depth={depth}] {current_url}")

        html = fetch_page_html(current_url, site)
        if not html:
            log.warning(f"[{site}] No HTML: {current_url}")
            time.sleep(REQUEST_DELAY)
            continue

        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
        except Exception as e:
            log.warning(f"[{site}] BS4 error: {e}")
            time.sleep(REQUEST_DELAY)
            continue

        new_links: list[str] = []

        for tag in soup.find_all("a", href=True):
            href = tag["href"].strip()
            if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
                continue
            abs_url = urljoin(current_url, href).replace(" ", "%20").split("#")[0].rstrip("/")
            if not abs_url.startswith("http"):
                continue

            if is_pdf_url(abs_url):
                if not _site_seen(source_label, abs_url):
                    link_text = tag.get_text(strip=True) or ""
                    label     = clean_name(link_text, abs_url, source_label)
                    log.info(f"[{site}] [PDF] {label}")
                    if validate_only:
                        results.append({"url": abs_url, "label": label, "status": "found"})
                    else:
                        results.append(ingest_pdf_immediately(abs_url, label, source_type, site))
                continue

            if depth >= MAX_DEPTH:
                continue
            if source.get("stay_on_domain", True) and not same_domain(abs_url, root_url):
                continue
            if abs_url in visited_pages:
                continue
            if must_contain and not any(kw.lower() in abs_url.lower() for kw in must_contain):
                continue
            if any(re.search(p, abs_url, re.I) for p in SKIP_PATTERNS):
                continue
            new_links.append(abs_url)

        for tag in soup.find_all(["iframe", "embed"], src=True):
            src     = tag["src"].strip()
            abs_url = urljoin(current_url, src).replace(" ", "%20")
            if is_pdf_url(abs_url) and not _site_seen(source_label, abs_url):
                label = clean_name("", abs_url, source_label)
                if validate_only:
                    results.append({"url": abs_url, "label": label, "status": "found"})
                else:
                    results.append(ingest_pdf_immediately(abs_url, label, source_type, site))

        for link in new_links:
            if link not in visited_pages:
                queue.append((link, depth + 1))

        ok  = sum(1 for r in results if r.get("status") == "ok")
        skp = sum(1 for r in results if r.get("status") == "skipped")
        log.info(f"[{site}] pages={len(visited_pages)} ok={ok} skip={skp} queue={len(queue)}")
        time.sleep(REQUEST_DELAY)

    ok  = sum(1 for r in results if r.get("status") == "ok")
    skp = sum(1 for r in results if r.get("status") == "skipped")
    fail = sum(1 for r in results if r.get("status") not in ("ok","skipped","found"))
    log.info(f"[{site}] DONE — ok={ok} skip={skp} fail={fail}")
    return results

# ─────────────────────────────────────────────────────────────────────────────
# Build 6 worker source dicts (3 SAMA + 3 NCA)
# ─────────────────────────────────────────────────────────────────────────────

def _build_worker_sources() -> list[dict]:
    workers = []
    base_sama = next(s for s in SOURCES if s["label"] == "SAMA")
    for i, entry in enumerate(SAMA_ENTRY_POINTS, 1):
        workers.append({**base_sama, "root_url": entry, "worker_id": f"SAMA-{i}"})
    base_nca = next(s for s in SOURCES if s["label"] == "NCA")
    for i, entry in enumerate(NCA_ENTRY_POINTS, 1):
        workers.append({**base_nca, "root_url": entry, "worker_id": f"NCA-{i}"})
    return workers

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: list[dict]) -> None:
    ok      = [r for r in results if r.get("status") == "ok"]
    skipped = [r for r in results if r.get("status") == "skipped"]
    failed  = [r for r in results if r.get("status") not in ("ok", "skipped", "found")]
    total_chunks = sum(r.get("chunks_inserted", 0) for r in ok)
    total_pages  = sum(r.get("total_pages",     0) for r in ok)

    print(f"\n{'='*55}")
    print("  INGESTION SUMMARY")
    print(f"{'='*55}")
    print(f"  ✓ Ingested       : {len(ok)}")
    print(f"  - Skipped (dup)  : {len(skipped)}")
    print(f"  ✗ Failed         : {len(failed)}")
    print(f"{'─'*55}")
    print(f"  Pages ingested   : {total_pages}")
    print(f"  Chunks inserted  : {total_chunks}")
    if DRY_RUN:
        print(f"\n  [DRY RUN] — no data written to DB.")
    print(f"{'='*55}\n")
    if failed:
        print("  Failed:")
        for r in failed:
            print(f"    [{r.get('status')}] {r.get('document_name') or r.get('url','')}")

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global DRY_RUN, STEALTH_MODE, MAX_DEPTH, CHUNK_SIZE, MIN_CHUNK_CHARS, CHUNK_OVERLAP

    parser = argparse.ArgumentParser(description="SAMA & NCA PDF Scraper — 6 workers")
    parser.add_argument("--url",           help="Crawl/ingest a single URL or PDF")
    parser.add_argument("--file",          help="Ingest a local PDF file")
    parser.add_argument("--name",          help="Document name for --file / --url")
    parser.add_argument("--source",        help="source_type override", default="Manual")
    parser.add_argument("--depth",         type=int, default=None,
                        help=f"Max crawl depth (default: {MAX_DEPTH})")
    parser.add_argument("--chunk-size",    type=int, default=None,
                        help=f"Chunk size in tokens (default: {CHUNK_SIZE})")
    parser.add_argument("--chunk-overlap", type=int, default=None,
                        help=f"Overlap sentences (default: {CHUNK_OVERLAP})")
    parser.add_argument("--dry-run",       action="store_true")
    parser.add_argument("--no-stealth",    action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if args.dry_run:       DRY_RUN = True
    if args.no_stealth:    STEALTH_MODE = False
    if args.depth:         MAX_DEPTH = args.depth
    if args.chunk_size:    CHUNK_SIZE = args.chunk_size
    if args.chunk_overlap: CHUNK_OVERLAP = args.chunk_overlap

    # ── Local file ────────────────────────────────────────────────────────
    if args.file:
        pdf_path = Path(args.file)
        if not pdf_path.exists():
            log.error(f"File not found: {pdf_path}"); sys.exit(1)
        doc_name = args.name or pdf_path.stem
        pages    = extract_pages(pdf_path, "local")
        if not pages:
            log.error("No text extracted."); sys.exit(1)
        total  = pages[0]["total_pages"]
        chunks = []
        for p in pages:
            chunks.extend(chunk_page(p["text"], p["page"], doc_name))
        log.info(f"Pages: {total} | Chunks: {len(chunks)}")
        doc_id   = upsert_document(doc_name, args.source, total)
        inserted = insert_chunks(doc_id, chunks, "local")
        print_summary([{"status": "ok", "document_name": doc_name,
                        "total_pages": total, "chunks_inserted": inserted}])
        return

    # ── Single URL ────────────────────────────────────────────────────────
    if args.url:
        source = {
            "root_url": args.url, "label": args.name or "Manual",
            "source_type": args.source, "stay_on_domain": True,
            "url_must_contain": [], "worker_id": "Manual",
        }
        if is_pdf_url(args.url):
            label   = args.name or clean_name("", args.url)
            results = [ingest_pdf_immediately(args.url, label, args.source, "Manual")]
        else:
            results = crawl_and_ingest(source, validate_only=args.validate_only)
        if not args.validate_only:
            print_summary(results)
        else:
            for r in results:
                print(f"  {r.get('label','')}\n    {r.get('url','')}")
        return

    # ── 6-worker full crawl ───────────────────────────────────────────────
    worker_sources = _build_worker_sources()

    log.info(f"\n{'='*55}")
    log.info(f"  SAMA NORA Scraper — {len(worker_sources)} workers")
    log.info(f"  SAMA-1/2/3  NCA-1/2/3")
    log.info(f"  Chunk size: {CHUNK_SIZE} tokens | Overlap: {CHUNK_OVERLAP} sentences")
    log.info(f"  Extraction: pdfplumber → pdfminer → pymupdf")
    log.info(f"{'='*55}\n")

    all_results: list[dict] = []

    with ThreadPoolExecutor(max_workers=len(worker_sources), thread_name_prefix="W") as executor:
        future_map = {
            executor.submit(crawl_and_ingest, src, args.validate_only): src
            for src in worker_sources
        }
        for future in as_completed(future_map):
            src = future_map[future]
            wid = src.get("worker_id", src["label"])
            try:
                all_results.extend(future.result())
            except Exception as exc:
                log.error(f"[{wid}] Worker crashed: {exc}", exc_info=True)

    if not args.validate_only:
        print_summary(all_results)
    else:
        by_site: dict[str, list] = {}
        for r in all_results:
            s = r.get("source_type", "?")
            by_site.setdefault(s, []).append(r)
        for site, items in by_site.items():
            print(f"\n  {site}: {len(items)} PDFs")
            for r in items:
                print(f"    - {r.get('label','')}\n      {r.get('url','')}")


if __name__ == "__main__":
    main()