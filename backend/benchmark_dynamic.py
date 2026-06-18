"""
SAMA NORA — Dynamic Benchmark with Supabase-Generated Questions
================================================================
Phase 1  Generate questions from live Supabase chunks (GPT-4o-mini).
         Skipped automatically if generated_questions.json already exists.
Phase 2  Run benchmark with auto-resume from last completed row.

Usage
-----
    python benchmark_dynamic.py              # full run (~1500 questions)
    python benchmark_dynamic.py --limit 50   # quick smoke-test
    python benchmark_dynamic.py --regen      # force regenerate questions
    python benchmark_dynamic.py --target 2000 # aim for 2000 questions

Output files (all resumable / live-written)
    generated_questions.json   cached generated questions
    benchmark_results.csv      one row per question, written immediately
    benchmark_stats.json       final + per-category stats
    benchmark.log              full verbose log
"""

from __future__ import annotations
import argparse, csv, json, logging, os, random, sys, time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
SUPABASE_URL  = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY  = (os.environ.get("SUPABASE_KEY")
                 or os.environ.get("SUPABASE_SERVICE_ROLE_KEY", ""))
OPENAI_KEY    = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL  = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
API_URL       = "http://localhost:8000"
API_KEY       = "0d52daf5f34807f9adfb5bca028a770f25a294156ecf22a4247b38d6c0c666cd"
TIMEOUT       = 120

OUT_QUESTIONS = "generated_questions.json"
OUT_CSV       = "benchmark_results.csv"
OUT_STATS     = "benchmark_stats.json"
OUT_LOG       = "benchmark.log"

# Generation tuning
CHUNKS_PER_CALL     = 3    # chunks batched in one OpenAI call
QUESTIONS_PER_CHUNK = 3    # 2 EN + 1 AR per chunk
SAMPLE_PER_DOC      = 18   # chunks sampled per document

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(OUT_LOG, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("bench")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Question Generation
# ═══════════════════════════════════════════════════════════════════════════════

def _doc_to_category(doc_name: str) -> str:
    d = doc_name.lower()
    if any(x in d for x in ["1644", "bank account", "rules for bank", "account opening"]):
        return "bank_account"
    if any(x in d for x in ["1704", "1428", "aml", "fatf", "kyc", "anti-money"]):
        return "kyc_aml"
    if any(x in d for x in ["3487", "basel", "capital", "liquidity", "lcr",
                              "nsfr", "pillar", "leverage", "ifrs"]):
        return "capital"
    if any(x in d for x in ["nca", "ecc", "ccc", "otcc", "sacs", "cybersec",
                              "aramco", "cyber", "ict"]):
        return "cybersec"
    if any(x in d for x in ["pdpl", "personal data", "privacy", "sdaia", "ndmo"]):
        return "pdpl"
    if any(x in d for x in ["iso", "27001", "22301", "42001", "27701",
                              "27400", "23200", "20000", "grc", "cobit"]):
        return "iso_grc"
    return "general"


def _fetch_supabase_chunks(sample_per_doc: int = SAMPLE_PER_DOC) -> list[dict]:
    """
    Fetch a stratified sample of chunks from Supabase.
    Spreads sampling across all documents and page ranges.
    """
    from supabase import create_client
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    # 1. Get all unique document names
    log.info("[gen] Fetching document list from Supabase...")
    res = sb.table("sama_nora_chunks").select("document_name").execute()
    all_docs = list({r["document_name"] for r in (res.data or [])})
    log.info(f"[gen] Found {len(all_docs)} documents")

    all_chunks: list[dict] = []

    for doc in all_docs:
        # Count pages in this document
        count_res = sb.table("sama_nora_chunks")\
            .select("page_start", count="exact")\
            .eq("document_name", doc)\
            .execute()
        total = count_res.count or 0
        if total == 0:
            continue

        # Sample evenly across the document
        step   = max(1, total // sample_per_doc)
        offsets= list(range(0, total, step))[:sample_per_doc]
        random.shuffle(offsets)

        for offset in offsets:
            res2 = sb.table("sama_nora_chunks")\
                .select("id,document_name,content,language,page_start")\
                .eq("document_name", doc)\
                .order("page_start")\
                .range(offset, offset)\
                .execute()
            if res2.data:
                chunk = res2.data[0]
                chunk["_category"] = _doc_to_category(doc)
                all_chunks.append(chunk)

    log.info(f"[gen] Sampled {len(all_chunks)} chunks across {len(all_docs)} documents")
    return all_chunks


def _generate_questions_from_chunks(chunks: list[dict]) -> list[dict]:
    """
    Call OpenAI once per batch of CHUNKS_PER_CALL chunks.
    Returns a list of question dicts: {q, cat, lang, expect, source_doc}
    """
    import openai
    client = openai.OpenAI(api_key=OPENAI_KEY)
    questions: list[dict] = []

    batches = [chunks[i:i+CHUNKS_PER_CALL]
               for i in range(0, len(chunks), CHUNKS_PER_CALL)]

    log.info(f"[gen] Generating questions — {len(batches)} OpenAI calls...")

    for b_idx, batch in enumerate(batches, 1):
        passages = ""
        for j, chunk in enumerate(batch, 1):
            content = (chunk.get("content") or "")[:900]
            passages += f"\n--- Passage {j} ({chunk.get('document_name','?')}) ---\n{content}\n"

        prompt = (
            "You are a Saudi banking and regulatory compliance examiner.\n"
            "Below are regulatory text passages. For EACH passage generate:\n"
            f"  - 2 specific questions IN ENGLISH that can be answered directly from that passage\n"
            f"  - 1 specific question IN ARABIC that can be answered from that passage\n\n"
            "Rules:\n"
            "- Questions must be answerable from the passage text alone\n"
            "- Questions must be about Saudi banking, cybersecurity, or data protection regulations\n"
            "- Use formal regulatory phrasing (avoid 'SME', prefer 'juristic person' etc.)\n"
            "- No generic questions — each must be specific to the passage content\n\n"
            f"{passages}\n"
            "Return a JSON array only, no explanation:\n"
            '[\n'
            '  {"passage": 1, "q": "English question", "lang": "en"},\n'
            '  {"passage": 1, "q": "English question 2", "lang": "en"},\n'
            '  {"passage": 1, "q": "Arabic question", "lang": "ar"},\n'
            '  {"passage": 2, ...}\n'
            ']\n'
            "Return ONLY valid JSON. No markdown, no backticks."
        )

        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=600,
                    temperature=0.4,
                )
                raw = resp.choices[0].message.content.strip()
                raw = raw.lstrip("```json").lstrip("```").rstrip("```").strip()
                items = json.loads(raw)
                for item in items:
                    p_idx = item.get("passage", 1) - 1
                    if 0 <= p_idx < len(batch):
                        chunk = batch[p_idx]
                    else:
                        chunk = batch[0]
                    q_text = item.get("q", "").strip()
                    if len(q_text) < 10:
                        continue
                    questions.append({
                        "q":          q_text,
                        "cat":        chunk.get("_category", "general"),
                        "lang":       item.get("lang", "en"),
                        "expect":     "answered",
                        "source_doc": chunk.get("document_name", ""),
                    })
                break
            except Exception as e:
                if attempt == 2:
                    log.warning(f"[gen] Batch {b_idx} failed after 3 attempts: {e}")
                else:
                    time.sleep(1.5 ** attempt)

        # Progress update every 20 batches
        if b_idx % 20 == 0:
            log.info(f"[gen] {b_idx}/{len(batches)} batches done — {len(questions)} questions so far")

    return questions


def _add_meta_questions() -> list[dict]:
    """Add identity, out-of-scope, and edge-case questions."""
    return [
        {"q":"Who are you?",                               "cat":"identity","lang":"en","expect":"identity"},
        {"q":"What are you?",                              "cat":"identity","lang":"en","expect":"identity"},
        {"q":"What is your name?",                         "cat":"identity","lang":"en","expect":"identity"},
        {"q":"من أنت؟",                                    "cat":"identity","lang":"ar","expect":"identity"},
        {"q":"ما اسمك؟",                                  "cat":"identity","lang":"ar","expect":"identity"},
        {"q":"What is the weather in Riyadh?",             "cat":"out_of_scope","lang":"en","expect":"out_of_scope"},
        {"q":"Who is the president of the United States?", "cat":"out_of_scope","lang":"en","expect":"out_of_scope"},
        {"q":"What is the recipe for kabsa?",              "cat":"out_of_scope","lang":"en","expect":"out_of_scope"},
        {"q":"Hello",                                      "cat":"out_of_scope","lang":"en","expect":"out_of_scope"},
        {"q":"Okay thanks",                                "cat":"out_of_scope","lang":"en","expect":"out_of_scope"},
        # Edge / informal
        {"q":"whats the kyc requirements for companies",   "cat":"kyc_aml","lang":"en","expect":"answered"},
        {"q":"how much capital do saudi banks need",        "cat":"capital","lang":"en","expect":"answered"},
        {"q":"what r the aml rules",                       "cat":"kyc_aml","lang":"en","expect":"answered"},
        {"q":"nca ecc controls",                           "cat":"cybersec","lang":"en","expect":"answered"},
        {"q":"pdpl penalties saudi arabia",                "cat":"pdpl","lang":"en","expect":"answered"},
        {"q":"sme bank account requirements sama",          "cat":"bank_account","lang":"en","expect":"answered"},
        {"q":"lcr ratio requirement",                       "cat":"capital","lang":"en","expect":"answered"},
        {"q":"pep edd requirements",                       "cat":"kyc_aml","lang":"en","expect":"answered"},
        {"q":"who all cannot create a bank account?",      "cat":"bank_account","lang":"en","expect":"answered"},
        {"q":"what are the SAMA regulations for opening bank accounts for SMEs?","cat":"bank_account","lang":"en","expect":"answered"},
        {"q":"What documents does an SME need to open a bank account?","cat":"bank_account","lang":"en","expect":"answered"},
        {"q":"What is SAMA?",                              "cat":"general","lang":"en","expect":"answered"},
        {"q":"What is NORA?",                              "cat":"general","lang":"en","expect":"answered"},
        {"q":"What is KYC?",                               "cat":"general","lang":"en","expect":"answered"},
        {"q":"ما هي ساما؟",                               "cat":"general","lang":"ar","expect":"answered"},
    ]


def generate_questions(target: int, regen: bool) -> list[dict]:
    """
    Phase 1: Generate or load questions.
    Returns combined list ready for benchmarking.
    """
    if not regen and os.path.exists(OUT_QUESTIONS):
        with open(OUT_QUESTIONS, encoding="utf-8") as f:
            existing = json.load(f)
        log.info(f"[gen] Loaded {len(existing)} cached questions from {OUT_QUESTIONS}")
        if len(existing) >= target * 0.8:
            return existing
        log.info(f"[gen] Cached count below target {target}. Regenerating...")

    if not SUPABASE_URL or not SUPABASE_KEY:
        log.error("[gen] SUPABASE_URL / SUPABASE_KEY not set in .env. Cannot generate questions.")
        sys.exit(1)
    if not OPENAI_KEY:
        log.error("[gen] OPENAI_API_KEY not set in .env. Cannot generate questions.")
        sys.exit(1)

    # Calculate how many chunks we need
    # target questions ≈ chunks × QUESTIONS_PER_CHUNK
    # subtract meta questions (25)
    needed_generated = max(target - 25, target)
    sample_per_doc   = max(SAMPLE_PER_DOC, needed_generated // 30)  # assume ~30 docs

    chunks    = _fetch_supabase_chunks(sample_per_doc=sample_per_doc)
    generated = _generate_questions_from_chunks(chunks)
    meta      = _add_meta_questions()
    all_qs    = generated + meta

    # Shuffle so categories are mixed
    random.shuffle(all_qs)

    with open(OUT_QUESTIONS, "w", encoding="utf-8") as f:
        json.dump(all_qs, f, ensure_ascii=False, indent=2)

    log.info(f"[gen] Saved {len(all_qs)} questions to {OUT_QUESTIONS}")
    return all_qs


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Benchmark Runner
# ═══════════════════════════════════════════════════════════════════════════════

def _count_existing_results() -> int:
    """Count completed rows in benchmark_results.csv (excluding header)."""
    if not os.path.exists(OUT_CSV):
        return 0
    try:
        with open(OUT_CSV, encoding="utf-8", newline="") as f:
            return max(0, sum(1 for _ in csv.reader(f)) - 1)
    except Exception:
        return 0


def _classify(result: dict, expect: str) -> str:
    method = result.get("method", "")
    answer = result.get("answer", "").lower()
    sources= result.get("sources", [])

    if expect == "identity":
        return "PASS" if method == "identity" else "FAIL"
    if expect == "out_of_scope":
        return "PASS" if method == "out_of_scope" else "FAIL"
    if expect == "answered":
        if method in ("not_found", "out_of_scope"):
            return "FAIL"
        if "does not contain" in answer or "cannot find" in answer:
            return "NOT_FOUND"
        return "PASS" if sources else "PARTIAL"
    return "UNKNOWN"


MAX_RETRIES  = 3   # attempts per question on error
RETRY_DELAY  = 3   # base seconds between retries (doubles each attempt)
TIMEOUT      = 60  # seconds per attempt — reduced so failures are detected faster

NOT_FOUND_PHRASES = [
    "does not contain", "cannot find", "not found in",
    "لا تتوفر", "لم أجد",
]

def _is_not_found_answer(answer: str) -> bool:
    a = answer.lower()
    return any(p in a for p in NOT_FOUND_PHRASES)


def _is_error_result(result: dict, expect: str) -> bool:
    """
    Return True if this result should be retried.
    Returns False for valid 'not found' answers — those are correct responses,
    not errors, and retrying them wastes time and doesn't change the outcome.
    """
    if "error" in result:                          return True
    if result.get("method") in ("error", ""):      return True
    if result.get("answer", "") == "":             return True
    if expect == "answered" and not result.get("sources"):
        # A proper "not found" answer with no sources is VALID — don't retry
        if _is_not_found_answer(result.get("answer", "")):
            return False
        method = result.get("method", "")
        if method not in ("not_found", "out_of_scope", "identity", "cached"):
            return True
    return False


def _send(question: str) -> tuple[dict, float]:
    import urllib.request as _ur, json as _j
    body = _j.dumps({"query": question}).encode()
    req  = _ur.Request(
        f"{API_URL}/api/query", data=body,
        headers={"Content-Type":"application/json","X-API-Key":API_KEY},
        method="POST",
    )
    t0 = time.time()
    try:
        with _ur.urlopen(req, timeout=TIMEOUT) as r:
            return _j.loads(r.read().decode()), (time.time()-t0)*1000
    except Exception as e:
        return {"error":str(e),"answer":"","sources":[],"method":"error"}, (time.time()-t0)*1000


def _send_with_retry(question: str, expect: str) -> tuple[dict, float, int]:
    """
    Send query with automatic retry on error.
    Returns (result, total_ms, attempts_used).
    Retries when _is_error_result() is True, up to MAX_RETRIES times.
    """
    total_ms = 0.0
    for attempt in range(1, MAX_RETRIES + 1):
        result, ms = _send(question)
        total_ms += ms
        if not _is_error_result(result, expect):
            if attempt > 1:
                log.info(f"  ✔ Retry {attempt} succeeded")
            return result, total_ms, attempt
        if attempt < MAX_RETRIES:
            delay = RETRY_DELAY * (2 ** (attempt - 1))
            log.warning(f"  ⚠ Attempt {attempt}/{MAX_RETRIES} failed "
                        f"(method={result.get('method','?')}). "
                        f"Retrying in {delay}s...")
            time.sleep(delay)
        else:
            log.warning(f"  ✗ All {MAX_RETRIES} attempts failed for: {question[:60]}")
    return result, total_ms, MAX_RETRIES


def _write_header(f):
    w = csv.writer(f)
    # 12 columns only — no source_doc or crag_triggered, keeps format
    # stable across versions so append/resume never causes column shift
    w.writerow(["idx","question","category","lang","expect",
                "status","method","sources","reranker_score",
                "answer_len","time_ms","answer_preview"])
    return w


def run_benchmark(questions: list[dict], start_idx: int, limit: int | None):
    total    = len(questions)
    todo     = questions[start_idx:]
    if limit:
        todo = todo[:limit]

    log.info("=" * 65)
    log.info(f"SAMA NORA Dynamic Benchmark")
    log.info(f"Total questions : {total}")
    log.info(f"Starting from   : #{start_idx + 1}")
    log.info(f"To run          : {len(todo)}")
    log.info(f"API             : {API_URL}")
    log.info("=" * 65)

    counts    = {"PASS":0,"FAIL":0,"NOT_FOUND":0,"PARTIAL":0,"ERROR":0,"UNKNOWN":0}
    times_ms  : list[float] = []
    rscores   : list[float] = []
    by_cat    : dict        = {}
    crag_hits = 0

    # Open CSV — append if resuming, create with header if new
    is_new = not os.path.exists(OUT_CSV) or start_idx == 0
    mode   = "w" if is_new else "a"
    csv_f  = open(OUT_CSV, mode, newline="", encoding="utf-8")
    writer = _write_header(csv_f) if is_new else csv.writer(csv_f)

    t_start = time.time()

    try:
        for i, item in enumerate(todo, 1):
            global_idx = start_idx + i
            q      = item["q"]
            cat    = item.get("cat","general")
            lang   = item.get("lang","en")
            expect = item.get("expect","answered")
            src    = item.get("source_doc","")

            log.info(f"[{global_idx}/{total}] {cat}/{lang} | {q[:72]}")

            result, ms, attempts = _send_with_retry(q, expect)
            times_ms.append(ms)

            status = "ERROR" if "error" in result else _classify(result, expect)
            counts[status] = counts.get(status, 0) + 1

            rs = result.get("reranker_top_score")
            if rs is not None:
                rscores.append(float(rs))

            answer  = result.get("answer", "")
            sources = result.get("sources", [])
            method  = result.get("method", "")

            # Detect CRAG trigger from server log (if method=generative + sources exist after retry)
            crag_log = "[crag] Retry succeeded" in answer or result.get("crag_triggered", False)
            if crag_log:
                crag_hits += 1

            log.info(f"  → {status} | method={method} | src={len(sources)} | rs={rs} | {ms:.0f}ms")
            if status in ("FAIL","NOT_FOUND","ERROR"):
                log.warning(f"  ✗ {answer[:200]}")
            else:
                log.info(f"  ✓ {answer[:140]}")

            writer.writerow([
                global_idx, q, cat, lang, expect,
                status, method, len(sources),
                round(float(rs),4) if rs else "",
                len(answer), round(ms),
                answer[:220].replace("\n"," "),
            ])
            csv_f.flush()

            by_cat.setdefault(cat, {"PASS":0,"FAIL":0,"NOT_FOUND":0,"PARTIAL":0,"ERROR":0})
            by_cat[cat][status] = by_cat[cat].get(status,0) + 1

            # Progress every 50 questions
            if i % 50 == 0:
                elapsed = time.time() - t_start
                eta     = (elapsed / i) * (len(todo) - i)
                done    = counts.get("PASS",0) + counts.get("PARTIAL",0)
                pct     = done / (start_idx + i) * 100
                log.info(f"\n{'─'*60}")
                log.info(f"  Progress : {global_idx}/{total} ({i/len(todo)*100:.0f}%)")
                log.info(f"  Pass rate: {pct:.1f}%  |  ETA: {eta/60:.1f} min")
                log.info(f"  Counts   : {counts}")
                log.info(f"{'─'*60}\n")

    except KeyboardInterrupt:
        log.info("\n[!] Interrupted by user. Saving stats and exiting...")
    finally:
        csv_f.close()

    _save_stats(counts, times_ms, rscores, by_cat, crag_hits, time.time()-t_start, total)


def _save_stats(counts, times_ms, rscores, by_cat, crag_hits, elapsed, total):
    done       = sum(counts.values())
    pass_count = counts.get("PASS",0) + counts.get("PARTIAL",0)

    stats = {
        "run_date":        datetime.now().isoformat(),
        "total_questions": total,
        "completed":       done,
        "elapsed_seconds": round(elapsed),
        "elapsed_min":     round(elapsed/60, 1),
        "counts":          counts,
        "pass_rate_pct":   round(pass_count/done*100, 1) if done else 0,
        "not_found_pct":   round(counts.get("NOT_FOUND",0)/done*100, 1) if done else 0,
        "fail_rate_pct":   round(counts.get("FAIL",0)/done*100, 1) if done else 0,
        "avg_time_ms":     round(sum(times_ms)/len(times_ms)) if times_ms else 0,
        "p50_time_ms":     round(sorted(times_ms)[len(times_ms)//2]) if times_ms else 0,
        "p95_time_ms":     round(sorted(times_ms)[int(len(times_ms)*0.95)]) if times_ms else 0,
        "avg_reranker":    round(sum(rscores)/len(rscores),4) if rscores else None,
        "crag_retries":    crag_hits,
        "by_category":     by_cat,
    }
    with open(OUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    log.info("\n" + "═"*65)
    log.info("  BENCHMARK SUMMARY")
    log.info("═"*65)
    log.info(f"  Completed : {done} / {total}")
    log.info(f"  PASS      : {counts.get('PASS',0)}")
    log.info(f"  PARTIAL   : {counts.get('PARTIAL',0)}")
    log.info(f"  NOT_FOUND : {counts.get('NOT_FOUND',0)}")
    log.info(f"  FAIL      : {counts.get('FAIL',0)}")
    log.info(f"  ERROR     : {counts.get('ERROR',0)}")
    log.info(f"  Pass rate : {stats['pass_rate_pct']}%")
    log.info(f"  CRAG hits : {crag_hits}")
    log.info(f"  Avg time  : {stats['avg_time_ms']} ms")
    log.info(f"  P95 time  : {stats['p95_time_ms']} ms")
    log.info(f"  Avg rerank: {stats['avg_reranker']}")
    log.info(f"  Duration  : {stats['elapsed_min']} min")
    log.info("═"*65)
    log.info(f"  CSV    : {OUT_CSV}")
    log.info(f"  Stats  : {OUT_STATS}")
    log.info(f"  Log    : {OUT_LOG}")
    log.info("═"*65)

    # Per-category summary
    log.info("\n  Per-category pass rates:")
    for cat, c in sorted(by_cat.items()):
        t = sum(c.values())
        p = c.get("PASS",0) + c.get("PARTIAL",0)
        log.info(f"    {cat:<16} {p}/{t}  ({p/t*100:.0f}%)" if t else f"    {cat}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def _repair_csv_if_needed():
    """
    Detect and fix column-shifted rows written by a mismatched script version.

    Root cause: if an old 12-column CSV was resumed by a newer script that
    wrote 14 values per row (with source_doc + crag_triggered), every column
    from position 4 onward shifts by 2 — so 'expect' ends up in the 'status'
    column, 'status' ends up in 'method', etc.

    Detection: any row where the 'status' column holds an expect value
    ('answered', 'identity', 'out_of_scope') instead of a real status.

    Fix: remove the extra source_doc value at position 4, trim to 12 columns.
    """
    if not os.path.exists(OUT_CSV):
        return 0

    EXPECT_VALS  = {"answered", "identity", "out_of_scope"}
    VALID_STATUS = {"PASS", "FAIL", "ERROR", "NOT_FOUND", "PARTIAL", "UNKNOWN"}
    CANONICAL_HEADER = ["idx","question","category","lang","expect",
                        "status","method","sources","reranker_score",
                        "answer_len","time_ms","answer_preview"]

    with open(OUT_CSV, encoding="utf-8", newline="") as f:
        reader  = csv.reader(f)
        header  = next(reader)
        all_rows = list(reader)

    ci = {h: i for i, h in enumerate(header)}
    if "status" not in ci:
        return 0

    fixed = 0
    repaired = []
    for row in all_rows:
        status_val = row[ci["status"]] if len(row) > ci["status"] else ""
        if status_val in EXPECT_VALS:
            # Shifted row: remove the extra source_doc at position 4
            # Row layout (14 values): idx q cat lang [src] expect status method
            #                         sources rs ans_len ms [crag] preview
            fixed_row = row[:4] + row[5:]    # drop position 4 (source_doc)
            fixed_row = fixed_row[:12]       # trim crag_triggered + extras
            repaired.append(fixed_row)
            fixed += 1
        else:
            repaired.append(row[:12])        # trim any extra cols defensively

    if fixed > 0:
        log.info(f"[repair] ⚠ Found {fixed} column-shifted rows — repairing CSV...")
        with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(CANONICAL_HEADER)
            w.writerows(repaired)
        log.info(f"[repair] ✔ CSV repaired. {fixed} rows corrected.")
    else:
        log.info("[repair] CSV format OK — no column shifts detected.")

    return fixed


def retry_error_rows():
    """
    Post-run cleanup pass.
    Reads benchmark_results.csv, finds every row where:
      - status  == "ERROR"
      - method  == "error" or method is empty
      - sources == "0"  (no sources for an answered question)
    Retries each one (up to MAX_RETRIES), updates the row in-place,
    then rewrites the CSV.
    """
    if not os.path.exists(OUT_CSV):
        return

    log.info(f"\n{'═'*65}")
    log.info("  POST-RUN ERROR RETRY PASS")
    log.info(f"{'═'*65}")

    # ── Quick server health check before attempting 1000+ retries ────────────
    try:
        import urllib.request as _ur, json as _j
        req = _ur.Request(f"{API_URL}/health", method="GET")
        with _ur.urlopen(req, timeout=10) as r:
            log.info(f"  Server health: OK ({r.status})")
    except Exception as e:
        log.error(f"  Server not reachable: {e}")
        log.error("  Start the backend first, then re-run to trigger the retry pass.")
        return

    # ── Load all rows ─────────────────────────────────────────────────────────
    with open(OUT_CSV, encoding="utf-8", newline="") as f:
        reader  = csv.reader(f)
        header  = next(reader)
        all_rows = list(reader)

    # Column indices
    ci = {h: i for i, h in enumerate(header)}

    def _needs_retry(row: list) -> bool:
        if len(row) <= max(ci.get("status",5), ci.get("method",6)):
            return True   # malformed row — retry
        status  = row[ci["status"]]
        method  = row[ci["method"]]
        sources = row[ci.get("sources", 7)] if len(row) > ci.get("sources",7) else "0"
        expect  = row[ci.get("expect", 4)]  if len(row) > ci.get("expect",4)  else ""
        preview = row[ci.get("answer_preview", 11)] if len(row) > ci.get("answer_preview",11) else ""
        # Never retry a valid "not found" response — it's a correct answer, not an error
        if _is_not_found_answer(preview):
            return False
        if status  == "ERROR":                          return True
        if method  in ("error", ""):                   return True
        if sources == "0" and expect == "answered":    return True
        return False

    error_rows = [(i, r) for i, r in enumerate(all_rows) if _needs_retry(r)]
    if not error_rows:
        log.info("  No error rows found — nothing to retry.")
        return

    log.info(f"  Found {len(error_rows)} error rows to retry...")
    fixed = 0

    for row_i, row in error_rows:
        q      = row[ci["question"]]
        expect = row[ci.get("expect", 4)] if len(row) > ci.get("expect",4) else "answered"
        cat    = row[ci["category"]]
        log.info(f"  Retrying row #{row[ci['idx']]} [{cat}]: {q[:65]}")

        result, ms, attempts = _send_with_retry(q, expect)
        status = "ERROR" if "error" in result else _classify(result, expect)
        answer  = result.get("answer", "")
        sources = result.get("sources", [])
        method  = result.get("method", "")
        rs      = result.get("reranker_top_score")

        log.info(f"    → {status} | method={method} | src={len(sources)} | attempts={attempts}")

        # Update the row in-place — pad if needed, then write 12 columns
        while len(row) < len(header):
            row.append("")
        row[ci["status"]]                    = status
        row[ci["method"]]                    = method
        row[ci.get("sources", 7)]            = str(len(sources))
        row[ci.get("reranker_score", 8)]     = str(round(float(rs),4)) if rs else ""
        row[ci.get("answer_len", 9)]         = str(len(answer))
        row[ci.get("time_ms", 10)]           = str(round(ms))
        row[ci.get("answer_preview", 11)]    = answer[:220].replace("\n"," ")
        all_rows[row_i] = row[:12]

        if status != "ERROR" and method != "error":
            fixed += 1

    # ── Rewrite CSV with updated rows ─────────────────────────────────────────
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header[:12])
        w.writerows([r[:12] for r in all_rows])

    log.info(f"\n  Retry pass complete: {fixed}/{len(error_rows)} rows fixed")
    log.info(f"  CSV updated: {OUT_CSV}")



def main():
    global API_URL
    parser = argparse.ArgumentParser(description="SAMA NORA Dynamic Benchmark")
    parser.add_argument("--target",  type=int, default=1500,
                        help="Target number of questions to generate (default: 1500)")
    parser.add_argument("--limit",   type=int, default=None,
                        help="Cap how many questions to run this session")
    parser.add_argument("--regen",   action="store_true",
                        help="Force regenerate questions even if cache exists")
    parser.add_argument("--no-gen",  action="store_true",
                        help="Skip generation entirely, use cached questions only")
    parser.add_argument("--url",     default=API_URL,
                        help="API base URL (default: http://localhost:8000)")
    args = parser.parse_args()
    API_URL = args.url

    # ── Phase 1: Generate / Load Questions ───────────────────────────────────
    if args.no_gen:
        if not os.path.exists(OUT_QUESTIONS):
            log.error(f"--no-gen specified but {OUT_QUESTIONS} not found. Run without --no-gen first.")
            sys.exit(1)
        with open(OUT_QUESTIONS, encoding="utf-8") as f:
            questions = json.load(f)
        log.info(f"[gen] Loaded {len(questions)} cached questions (--no-gen mode)")
    else:
        questions = generate_questions(target=args.target, regen=args.regen)

    if not questions:
        log.error("No questions available. Exiting.")
        sys.exit(1)

    log.info(f"[bench] Total question pool: {len(questions)}")

    # ── Phase 1.5: Repair any column-shifted CSV rows from old script versions ─
    _repair_csv_if_needed()

    # ── Phase 2: Auto-resume ─────────────────────────────────────────────────
    completed = _count_existing_results()
    if completed > 0:
        log.info(f"[bench] Resuming from question #{completed + 1} ({completed} already done)")
    else:
        log.info("[bench] Starting fresh run")

    # ── Phase 2: Run ─────────────────────────────────────────────────────────
    run_benchmark(questions, start_idx=completed, limit=args.limit)

    # ── Phase 3: Retry any remaining error rows ───────────────────────────────
    if not args.limit:   # skip retry pass on --limit runs (smoke tests)
        retry_error_rows()


if __name__ == "__main__":
    main()