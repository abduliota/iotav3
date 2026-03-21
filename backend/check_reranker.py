"""
check_reranker.py — Verify reranker + hybrid search work before running full test suite
Run:  python check_reranker.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 60)
print("SAMA Chatbot — Reranker + Hybrid Search Diagnostic")
print("=" * 60)

# ── 1. Test CrossEncoder import ───────────────────────────────
print("\n[1] Testing CrossEncoder import...")
try:
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    test_pairs = [
        ("What is the leverage ratio?", "The leverage ratio framework applies to all domestic banks."),
        ("What is the leverage ratio?", "The weather in Riyadh is hot."),
    ]
    scores = reranker.predict(test_pairs)
    print(f"   CrossEncoder OK. Scores: {[round(float(s),3) for s in scores]}")
    print(f"   (regulatory chunk score should be higher than irrelevant one)")
except Exception as e:
    print(f"   FAIL: {e}")
    print("   Fix: pip install sentence-transformers --break-system-packages")

# ── 2. Test Supabase keyword_search_chunks RPC ────────────────
print("\n[2] Testing keyword_search_chunks Supabase RPC...")
try:
    from simple_rag import fetch_chunks_keyword
    results = fetch_chunks_keyword("leverage ratio tier 1 capital", limit=3)
    if results:
        print(f"   Keyword search OK. Got {len(results)} results:")
        for r in results:
            print(f"     - {r.get('document_name','?')} p{r.get('page_start','?')} | rank={r.get('rank', r.get('similarity','?'))}")
    else:
        print("   WARNING: No results — check if supabase_hybrid_search.sql was run")
except Exception as e:
    print(f"   FAIL: {e}")
    print("   Fix: Run supabase_hybrid_search.sql in Supabase SQL Editor first")

# ── 3. Test hybrid fetch ──────────────────────────────────────
print("\n[3] Testing hybrid fetch (vector + keyword merged)...")
try:
    from simple_rag import _embed, _expand_query, fetch_chunks_hybrid
    query = "How should banks calculate the leverage ratio?"
    expanded = _expand_query(query)
    vec = _embed(expanded)
    candidates = fetch_chunks_hybrid(expanded, vec, limit=15)
    print(f"   Hybrid OK. Got {len(candidates)} candidates.")
    print(f"   Top 3:")
    for c in candidates[:3]:
        print(f"     - {c.get('document_name','?')} p{c.get('page_start','?')} sim={c.get('similarity',0):.3f}")
except Exception as e:
    print(f"   FAIL: {e}")

# ── 4. Test full pipeline with reranker ───────────────────────
print("\n[4] Testing full answer pipeline with reranker...")
try:
    from simple_rag import answer_query
    result = answer_query("What is the cap on cash inflows as a percentage of total cash outflows?", debug=True)
    method = result.get("method", "?")
    answer = result.get("answer", "")[:120]
    print(f"   Method: {method}")
    print(f"   Answer: {answer}...")
    if "75" in result.get("answer", "") or "cash inflows" in result.get("answer", "").lower():
        print("   CORRECT: Answer contains expected content")
    else:
        print("   WARNING: Answer may not be correct — check manually")
except Exception as e:
    print(f"   FAIL: {e}")

print("\n" + "=" * 60)
print("Diagnostic complete.")
print("If all checks pass, run: python api.py")
print("Then in another terminal: python test_questions.py")
print("=" * 60)