"""
apply_scraper_patch.py
Run: python apply_scraper_patch.py C:\path\to\scraper.py
Produces: scraper_patched.py in the same folder
"""
import sys, ast, os

if len(sys.argv) < 2:
    print("Usage: python apply_scraper_patch.py /path/to/scraper.py")
    sys.exit(1)

path = sys.argv[1]
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

print(f"Read {len(content.splitlines())} lines from {path}")

# ── NEW BLOCK to insert before Chunking section ──────────────────────────────
NEW_BLOCK = '''
CONTEXTUAL_HEADERS = _env_bool("CONTEXTUAL_HEADERS", True)
_openai_client_scraper = None


def _get_scraper_llm():
    """Lazy OpenAI/Azure client for context header generation at ingest time."""
    global _openai_client_scraper
    if _openai_client_scraper is None:
        import openai as _oai
        _backend = os.getenv("LLM_BACKEND", "openai")
        if (_backend == "azure" and os.getenv("AZURE_OPENAI_API_KEY")
                and os.getenv("AZURE_OPENAI_ENDPOINT")):
            _openai_client_scraper = _oai.AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version="2024-02-01",
            )
        else:
            _openai_client_scraper = _oai.OpenAI(
                api_key=os.getenv("OPENAI_API_KEY", "")
            )
    return _openai_client_scraper


def _infer_source_from_name(document_name: str) -> str:
    """Infer regulatory body from document name — same logic as enrich_chunks.py."""
    name = document_name.upper()
    NCA_SIGNALS  = ["ECC", "CCC", "OTCC", "NCA", "CYBERSECURITY CONTROL",
                    "ESSENTIAL CYBERSECURITY", "CLOUD CYBERSECURITY",
                    "OPERATIONAL TECHNOLOGY", "CYBERSECURITY STEERING"]
    PDPL_SIGNALS = ["PDPL", "PERSONAL DATA", "DATA PROTECTION", "SDAIA", "NDMO"]
    ISO_SIGNALS  = ["ISO 27001", "ISO 27701", "ISO 22301", "ISO 20000",
                    "ISO 42001", "ISO 23200", "ISO 27400"]
    for s in NCA_SIGNALS:
        if s in name: return "NCA (National Cybersecurity Authority)"
    for s in PDPL_SIGNALS:
        if s in name: return "PDPL (Personal Data Protection Law / SDAIA)"
    for s in ISO_SIGNALS:
        if s in name: return "ISO International Standard"
    return "SAMA (Saudi Arabian Monetary Authority)"


_HEADER_PROMPT = (
    "You are a Saudi banking and cybersecurity regulation expert.\\n\\n"
    "Regulatory body: {source_type}\\n"
    "Document name  : {document_name}\\n"
    "Section title  : {section_title}\\n"
    "Content preview: {content_preview}\\n\\n"
    "Write ONE sentence (max 70 words) describing:\\n"
    "1. Which specific {source_type} regulation or framework this text belongs to\\n"
    "2. What regulatory topic it covers\\n"
    "3. What entity type or scenario it applies to (SME, individual, bank, etc.)\\n\\n"
    "Important: Use {source_type} as the regulatory body — do not substitute another.\\n"
    "Return ONLY the sentence. No preamble, no labels."
)


def _generate_chunk_context(
    document_name: str,
    section_title: Optional[str],
    chunk_text: str,
) -> Optional[str]:
    """
    [Step 1 — Contextual Headers] Generate a 1-sentence context header for a chunk.
    Called at ingest time for every new chunk — runs once, stored permanently.
    Returns None on any failure so ingestion continues normally without headers.
    Set CONTEXTUAL_HEADERS=false in .env to disable.
    """
    if not CONTEXTUAL_HEADERS or DRY_RUN:
        return None
    try:
        client      = _get_scraper_llm()
        model       = os.getenv("AZURE_DEPLOYMENT", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        source_type = _infer_source_from_name(document_name)
        prompt = _HEADER_PROMPT.format(
            source_type     = source_type,
            document_name   = document_name or "Unknown regulatory document",
            section_title   = section_title or "N/A",
            content_preview = chunk_text[:400].replace("\\n", " "),
        )
        resp = client.chat.completions.create(
            model=model, max_tokens=100, temperature=0.1,
            messages=[{"role": "user", "content": prompt}],
        )
        header = resp.choices[0].message.content.strip().strip('"\\' ')
        return header if header and len(header) > 10 else None
    except Exception as e:
        log.debug(f"[context] Header generation failed (non-fatal): {e}")
        return None

'''

# ── PATCH 1: Insert before Chunking section ───────────────────────────────────
CHUNK_MARKER = "# ─────────────────────────────────────────────────────────────────────────────\n# Chunking"
if CHUNK_MARKER in content:
    content = content.replace(CHUNK_MARKER, NEW_BLOCK + CHUNK_MARKER, 1)
    print("  ✓ Patch 1: New functions inserted before Chunking section")
else:
    print("  ✗ Patch 1: Chunking marker not found — searching for 'def chunk_page'")
    if "def chunk_page" in content:
        content = content.replace("def chunk_page", NEW_BLOCK + "def chunk_page", 1)
        print("  ✓ Patch 1: Inserted before def chunk_page")

# ── PATCH 2: Update flush() inside chunk_page() ───────────────────────────────
OLD_FLUSH = '''    def flush():
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
            })'''

NEW_FLUSH = '''    def flush():
        chunk_text = " ".join(current_sents).strip()
        if len(chunk_text) >= MIN_CHUNK_CHARS:
            # [Step 1 — Contextual Headers] Generate at ingest time
            header = _generate_chunk_context(document_name, section, chunk_text)
            stored_content = (
                f"[Context: {header}]\\n\\n{chunk_text}" if header else chunk_text
            )
            chunks.append({
                "document_name": document_name,
                "page_start":    page_num,
                "page_end":      page_num,
                "section_title": section,
                "content":       stored_content,
                "token_count":   current_tokens,
                "language":      _detect_language(chunk_text),
            })'''

if OLD_FLUSH in content:
    content = content.replace(OLD_FLUSH, NEW_FLUSH, 1)
    print("  ✓ Patch 2: chunk_page() flush() updated")
else:
    print("  ✗ Patch 2: flush() not found — check indentation in your scraper.py")

# ── Syntax check ──────────────────────────────────────────────────────────────
try:
    ast.parse(content)
    print("  ✓ Syntax OK")
except SyntaxError as e:
    print(f"  ✗ Syntax error at line {e.lineno}: {e.msg}")
    sys.exit(1)

# ── Write output ──────────────────────────────────────────────────────────────
out = path.replace('.py', '_patched.py')
with open(out, 'w', encoding='utf-8') as f:
    f.write(content)
print(f"  Written → {out}")
print()
print("  Next steps:")
print("  1. Review the patched file")
print("  2. Rename scraper_patched.py → scraper.py")
print("  3. New documents ingested will automatically get context headers")