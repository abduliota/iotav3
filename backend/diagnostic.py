"""
Run this first to verify the full stack before touching the LLM:
    python diagnostic.py
"""

import os, sys, json
from dotenv import load_dotenv
load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = (
    os.environ.get("SUPABASE_KEY") or
    os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or ""
)
TABLE     = os.environ.get("SUPABASE_TABLE", "sama_nora_chunks")
EMB_MODEL = os.environ.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
TEST_QUERY = "what is the capital adequacy requirement"

SEP = "─" * 60
def header(t): print(f"\n{SEP}\n  {t}\n{SEP}")
def ok(m):   print(f"  ✓  {m}")
def fail(m): print(f"  ✗  {m}")
def info(m): print(f"  →  {m}")

header("STEP 1: Environment variables")
missing = []
if SUPABASE_URL:    ok("SUPABASE_URL is set")
else:               fail("SUPABASE_URL MISSING"); missing.append("SUPABASE_URL")
if SUPABASE_KEY:    ok("Supabase key is set")
else:               fail("SUPABASE_KEY or SUPABASE_SERVICE_ROLE_KEY MISSING"); missing.append("key")
if missing:
    print("\n  Fix .env first."); sys.exit(1)

header("STEP 2: Supabase connection")
try:
    from supabase import create_client
    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    ok("Supabase client created")
except Exception as e:
    fail(f"Cannot connect: {e}"); sys.exit(1)

header(f"STEP 3: Table '{TABLE}' — row count")
try:
    resp = client.table(TABLE).select("id", count="exact").limit(1).execute()
    count = resp.count
    ok(f"Table exists. Row count: {count}")
    if count == 0:
        fail("Table is EMPTY"); sys.exit(1)
except Exception as e:
    fail(f"Cannot query table '{TABLE}': {e}")
    info("Check SUPABASE_TABLE in .env — should be 'sama_nora_chunks'"); sys.exit(1)

header("STEP 4: Embedding column dimensions")
try:
    row = client.table(TABLE).select("id, embedding").limit(1).execute().data[0]
    emb = row.get("embedding")
    if emb is None:
        fail("embedding column is NULL — run reembed.py"); sys.exit(1)
    if isinstance(emb, str):
        emb = json.loads(emb)
    dim = len(emb)
    ok(f"Embeddings exist. Stored dimension: {dim}")
    info("  e5-small  → 384 dim")
    info("  bge-m3    → 1024 dim")
    info("  OpenAI    → 1536 dim")
    if dim == 384:
        info("You have e5-small embeddings. EMBEDDING_MODEL must be intfloat/multilingual-e5-small")
except Exception as e:
    fail(f"Could not read embedding: {e}"); sys.exit(1)

stored_dim = dim

header(f"STEP 5: Embedding model — {EMB_MODEL}")
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMB_MODEL)
    test_vec = model.encode("test", normalize_embeddings=True).tolist()
    query_dim = len(test_vec)
    ok(f"Model loaded. Query embedding dimension: {query_dim}")
    if query_dim != stored_dim:
        fail(f"DIMENSION MISMATCH — stored={stored_dim}, query={query_dim}")
        info("Change EMBEDDING_MODEL in .env to match stored dim, or run reembed.py")
        sys.exit(1)
    ok(f"Dimensions MATCH ({query_dim}) — embedding layer is healthy")
except Exception as e:
    fail(f"Cannot load model: {e}")
    info("pip install sentence-transformers"); sys.exit(1)

header("STEP 6: Embed test query")
info(f"Query: '{TEST_QUERY}'")
prefixed = f"query: {TEST_QUERY}" if "e5" in EMB_MODEL.lower() else TEST_QUERY
query_vec = model.encode(prefixed, normalize_embeddings=True).tolist()
ok(f"Query embedded. Dim={len(query_vec)}")

header("STEP 7: Raw Supabase similarity — NO threshold filter")
info("Testing if ANY similarity scores come back at all...")
try:
    for rpc_name, params in [
        ("match_chunks", {"query_embedding": query_vec, "match_threshold": 0.0, "match_count": 5}),
        ("match_chunks", {"query_embedding": query_vec, "match_count": 5, "snippet_char_limit": 200}),
    ]:
        try:
            raw = client.rpc(rpc_name, params).execute()
            if raw.data:
                ok(f"RPC '{rpc_name}' works. Top 5 similarity scores:")
                for i, r in enumerate(raw.data[:5]):
                    sim = round(r.get("similarity", 0), 4)
                    preview = str(r.get("content", ""))[:100].replace("\n", " ")
                    print(f"     [{i+1}] sim={sim}  →  {preview}...")
                top_sim = raw.data[0].get("similarity", 0)
                if top_sim < 0.3:
                    fail("Highest similarity < 0.3 — likely embedding model mismatch")
                elif top_sim < 0.6:
                    info("Similarity is low (0.3–0.6). Try a query that matches your doc text.")
                else:
                    ok("Similarity looks healthy (>0.6)")
                    info("If retrieval fails at runtime, lower SIMILARITY_THRESHOLD in .env to 0.5")
                break
        except Exception:
            continue
    else:
        fail("match_chunks RPC not working — check Supabase SQL function")
except Exception as e:
    fail(f"RPC call failed: {e}")

threshold = float(os.environ.get("SIMILARITY_THRESHOLD", "0.5"))
header(f"STEP 8: Retrieval with current threshold ({threshold})")
try:
    resp = client.rpc("match_chunks", {
        "query_embedding": query_vec,
        "match_threshold": threshold,
        "match_count": 5,
    }).execute()
    if resp.data:
        ok(f"{len(resp.data)} chunks returned above threshold {threshold}")
        for i, r in enumerate(resp.data):
            sim = round(r.get("similarity", 0), 4)
            doc = r.get("document_name", "?")
            print(f"     [{i+1}] sim={sim} | {doc}")
    else:
        fail(f"0 chunks above threshold {threshold}")
        info("Lower SIMILARITY_THRESHOLD in .env to 0.5")
except Exception as e:
    fail(f"Threshold retrieval failed: {e}")

header("DIAGNOSTIC COMPLETE")
print("  All green = ready to run: python simple_rag.py\n")
