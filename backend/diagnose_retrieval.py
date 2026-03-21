"""
diagnose_retrieval.py — Find out exactly why questions are returning not_found

Run from backend/ directory:
    python diagnose_retrieval.py

What it checks:
  1. Raw vector similarity scores for each failing question
  2. Whether query expansion is firing (which keywords are being appended)
  3. Top 10 chunks returned and their similarity scores
  4. Whether the LOW_CONF_THRESHOLD is the culprit
  5. Keyword (BM25) search results separately
  6. What threshold would be needed to pass each question

[FIX] FAILING list updated to reflect real 30 failures from test_run_20260321.
[FIX] Threshold analysis now shows the new 0.72 guard alongside 0.79 for comparison.
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from simple_rag import (
    _expand_query,
    _embed,
    fetch_chunks,
    fetch_chunks_keyword,
    fetch_chunks_hybrid,
    LOW_CONF_THRESHOLD,
    SIMILARITY_THRESHOLD,
    RERANK_FETCH_K,
)

DIVIDER = "=" * 70
SUB     = "-" * 70

# ── Failing questions to diagnose ────────────────────────────────────────────
# Bucket 1 — wrong chunk retrieved (retrieval / expansion problem)
# Bucket 3 — LLM refuses despite good chunks (generation problem)
# AR       — Arabic queries
# BASELINE — should pass; verify they still do after changes
FAILING = [
    # ── Bucket 1: capital adequacy / minimum % ────────────────────────────────
    ("What is the minimum capital adequacy ratio for banks",                "EN_B1"),
    ("What is the capital adequacy minimum percentage requirement",          "EN_B1"),
    ("What are the capital adequacy requirements for Saudi banks",           "EN_B1"),
    ("minimum capital requirement for a new bank license",                  "EN_B1"),

    # ── Bucket 1: savings products ────────────────────────────────────────────
    ("What are the general rules for savings products",                     "EN_B1"),
    ("What are the rules for savings accounts in Saudi banks",              "EN_B1"),

    # ── Bucket 1: admin service charges ──────────────────────────────────────
    ("What is the maximum administrative service charge",                   "EN_B1"),
    ("What is the cap on admin service charges for banking",                "EN_B1"),

    # ── Bucket 1: loan-to-deposit ratio ──────────────────────────────────────
    ("What is the loan to deposit ratio reporting requirement",             "EN_B1"),
    ("loan deposit ratio disclosure requirements banks",                    "EN_B1"),

    # ── Bucket 1: annual disclosure ───────────────────────────────────────────
    ("What are the annual disclosure requirements for banks",               "EN_B1"),
    ("pillar 3 disclosure requirements total assets",                       "EN_B1"),

    # ── Bucket 1: PDPL penalties ──────────────────────────────────────────────
    ("What are the PDPL penalties for violations",                          "EN_B1"),
    ("What are the fines for violating personal data protection law",       "EN_B1"),

    # ── Bucket 1: NCA-SAMA relationship ──────────────────────────────────────
    ("What is the relationship between NCA and SAMA",                       "EN_B1"),
    ("How does NCA cybersecurity framework relate to SAMA",                 "EN_B1"),

    # ── Bucket 3: LLM refuses despite good chunks ─────────────────────────────
    ("What are the IFRS 9 non-performing exposure classifications",         "EN_B3"),
    ("What is the SAMA cybersecurity framework third party requirement",     "EN_B3"),
    ("What are the clawback arrangements for deferred remuneration",        "EN_B3"),
    ("What are the binding common rules for data transfer",                 "EN_B3"),
    ("What is the loss event threshold for operational risk",               "EN_B3"),
    ("What are the leverage ratio requirements for banks",                  "EN_B3"),

    # ── Arabic failures ───────────────────────────────────────────────────────
    ("ما هي عقوبات مخالفة نظام حماية البيانات الشخصية",                  "AR"),
    ("ما هي نسبة القرض إلى الودائع",                                       "AR"),
    ("ما هي متطلبات الإفصاح السنوي للبنوك",                               "AR"),
    ("ما هو الحد الأدنى لنسبة كفاية رأس المال",                           "AR"),
    ("ما هي قواعد منتجات الادخار",                                         "AR"),
    ("ما هي العلاقة بين الهيئة الوطنية للأمن السيبراني وساما",            "AR"),

    # ── Baselines — must still pass ───────────────────────────────────────────
    ("What is SAMA",                                                        "EN_BASELINE"),
    ("What is the liquidity coverage ratio",                                "EN_BASELINE"),
    ("SAMA cybersecurity framework applicable sectors",                     "EN_BASELINE"),
    ("What are the KYC requirements for retail customers",                  "EN_BASELINE"),
    ("What is BYOD policy in cybersecurity",                                "EN_BASELINE"),
]


def diagnose(query: str, lang: str):
    print(f"\n{DIVIDER}")
    print(f"  QUERY [{lang}]: {query}")
    print(DIVIDER)

    # Step 1: Query expansion
    expanded = _expand_query(query)
    if expanded != query:
        appended = expanded[len(query):].strip()
        print(f"\n  ✓ EXPANSION fired — appended:")
        print(f"    {appended[:200]}")
    else:
        print(f"\n  ✗ NO EXPANSION — query sent as-is")
        if lang.startswith("EN_B1"):
            print(f"    ← This is a Bucket 1 failure candidate — consider adding an expansion key")

    # Step 2: Embed
    vec = _embed(expanded)
    print(f"\n  Embedding dim: {len(vec)}")

    # Step 3: Vector search (raw, threshold=0.0 to see all scores)
    print(f"\n  [VECTOR SEARCH] top 10 (threshold=0.0, no confidence guard)")
    try:
        from supabase import create_client
        sb = create_client(os.environ["SUPABASE_URL"],
                           os.environ.get("SUPABASE_KEY") or
                           os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or "")
        resp = sb.rpc("match_chunks", {
            "query_embedding": vec,
            "match_threshold": 0.0,   # bypass threshold to see all scores
            "match_count": 10,
        }).execute()
        chunks = resp.data or []
    except Exception as e:
        print(f"  ERROR: {e}")
        chunks = []

    if not chunks:
        print("  No chunks returned at all — DB issue?")
    else:
        for i, c in enumerate(chunks[:10], 1):
            sim  = float(c.get("similarity", 0))
            doc  = c.get("document_name", "?")[:45]
            p    = c.get("page_start", "?")
            snip = (c.get("content") or "")[:80].replace("\n", " ")
            bar  = "█" * int(sim * 20)
            flag = ""
            if sim >= LOW_CONF_THRESHOLD:
                flag = f" ✓ PASSES (≥{LOW_CONF_THRESHOLD})"
            elif sim >= 0.72:
                flag = f" ✓ PASSES new guard (≥0.72)"
            elif sim >= SIMILARITY_THRESHOLD:
                flag = f" ⚠ above retrieval threshold but BELOW confidence guard ({sim:.4f} < {LOW_CONF_THRESHOLD})"
            else:
                flag = " ✗ below retrieval threshold"
            print(f"  [{i}] {sim:.4f} {bar:<20}{flag}")
            print(f"       {doc} p{p}")
            print(f"       {snip}...")

    top_sim = max((float(c.get("similarity", 0)) for c in chunks), default=0.0)
    print(f"\n  Top similarity score : {top_sim:.4f}")
    print(f"  LOW_CONF_THRESHOLD   : {LOW_CONF_THRESHOLD}  (active .env value)")
    print(f"  SIMILARITY_THRESHOLD : {SIMILARITY_THRESHOLD}")

    if top_sim < LOW_CONF_THRESHOLD:
        gap = LOW_CONF_THRESHOLD - top_sim
        print(f"\n  ✗ VERDICT: Fails confidence guard by {gap:.4f}")
        print(f"    → Needs {gap:.4f} more similarity to pass")
        if top_sim >= 0.70:
            print(f"    → CLOSE MISS — query expansion should close this gap")
        elif top_sim >= 0.50:
            print(f"    → FAR MISS — expansion not bridging to right chunks")
            if lang == "AR":
                print(f"    → Arabic query: check if Arabic bridge key exists in QUERY_EXPANSIONS")
        else:
            print(f"    → VERY FAR — content likely not in DB for this topic")
    else:
        print(f"\n  ✓ VERDICT: Passes confidence guard — generation should work")
        if lang == "EN_B3":
            print(f"    → Bucket 3 case: retrieval is fine, check LLM system prompt or TOP_K")

    # Step 4: Keyword (BM25) search
    print(f"\n  [KEYWORD/BM25 SEARCH] top 5")
    try:
        kw_chunks = fetch_chunks_keyword(expanded, limit=5)
        if not kw_chunks:
            print("  No keyword results (keyword_search_chunks RPC may not exist)")
        else:
            for i, c in enumerate(kw_chunks[:5], 1):
                doc  = c.get("document_name", "?")[:45]
                p    = c.get("page_start", "?")
                sim  = c.get("similarity", c.get("rank", 0))
                snip = (c.get("content") or "")[:80].replace("\n", " ")
                print(f"  [{i}] rank={sim:.4f}  {doc} p{p}")
                print(f"       {snip}...")
    except Exception as e:
        print(f"  ERROR: {e}")

    # Step 5: Threshold analysis — shows both old (0.79) and new (0.72) guards
    if chunks:
        print(f"\n  [THRESHOLD ANALYSIS]")
        print(f"  {'threshold':<12} {'chunks pass':<14} {'status'}")
        print(f"  {'-'*40}")
        for t in [0.79, 0.75, 0.72, 0.70, 0.65, 0.60]:
            passing = [c for c in chunks if float(c.get("similarity", 0)) >= t]
            marker = ""
            if t == 0.79:
                marker = "  ← old guard"
            elif abs(t - LOW_CONF_THRESHOLD) < 0.001:
                marker = "  ← active guard (.env)"
            print(f"  {t:<12.2f} {len(passing):<14}{marker}")


def _print_summary(results: list[tuple[str, str, bool]]) -> None:
    """Print a pass/fail summary table at the end."""
    print(f"\n{DIVIDER}")
    print("  SUMMARY")
    print(DIVIDER)
    buckets: dict[str, list] = {}
    for query, lang, passed in results:
        buckets.setdefault(lang, []).append((query, passed))

    for lang, items in sorted(buckets.items()):
        passed_count = sum(1 for _, p in items if p)
        print(f"\n  [{lang}]  {passed_count}/{len(items)} pass")
        for q, p in items:
            icon = "✓" if p else "✗"
            print(f"    {icon}  {q[:65]}")


if __name__ == "__main__":
    print(f"\nRetrieval Diagnostic  —  SAMA NORA")
    print(f"LOW_CONF_THRESHOLD  = {LOW_CONF_THRESHOLD}")
    print(f"SIMILARITY_THRESHOLD = {SIMILARITY_THRESHOLD}")
    print(f"RERANK_FETCH_K      = {RERANK_FETCH_K}")
    print(f"Total questions     = {len(FAILING)}")

    summary_results: list[tuple[str, str, bool]] = []

    for query, lang in FAILING:
        # Run full diagnostic
        diagnose(query, lang)

        # Quick pass/fail for summary table (re-embed silently)
        try:
            exp = _expand_query(query)
            vec = _embed(exp)
            from supabase import create_client
            sb = create_client(os.environ["SUPABASE_URL"],
                               os.environ.get("SUPABASE_KEY") or
                               os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or "")
            resp = sb.rpc("match_chunks", {
                "query_embedding": vec,
                "match_threshold": 0.0,
                "match_count": 5,
            }).execute()
            chunks = resp.data or []
            top_sim = max((float(c.get("similarity", 0)) for c in chunks), default=0.0)
            passed = top_sim >= LOW_CONF_THRESHOLD
        except Exception:
            passed = False
        summary_results.append((query, lang, passed))

    _print_summary(summary_results)

    print(f"\n{DIVIDER}")
    print("DIAGNOSTIC COMPLETE")
    print(f"{DIVIDER}")
    print("""
WHAT TO DO WITH RESULTS:
─────────────────────────────────────────────
EN_B1 ✗ + NO EXPANSION fires  → Add expansion key to QUERY_EXPANSIONS in simple_rag.py
EN_B1 ✗ + EXPANSION fires     → Wrong chunk returned; check top chunk content vs expected answer
EN_B1 ✗ + top_sim < 0.50      → Content not in DB; ingest the source document
EN_B3 ✓ but answer still wrong → Retrieval fine; tune SYSTEM_PROMPT or raise TOP_K
AR    ✗                        → Check Arabic bridge key exists in QUERY_EXPANSIONS
BASELINE ✗                     → Regression; something broke — investigate before deploying
─────────────────────────────────────────────
After any changes: clear Redis cache → restart backend → re-run this script.
""")