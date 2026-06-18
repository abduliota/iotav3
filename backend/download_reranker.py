"""
download_reranker.py
Downloads BAAI/bge-reranker-v2-m3 to the correct HF cache.
Run: python download_reranker.py
"""
import sys
import os
os.environ["HF_HUB_OFFLINE"] = "0"  # Force online mode

# Flush all output immediately
def p(msg):
    print(msg, flush=True)
    sys.stdout.flush()

p("Step 1: Setting cache path...")
CACHE = r"C:\Users\Abdul Salam M\.cache\huggingface\hub"
os.environ["HF_HUB_CACHE"] = CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = CACHE
p(f"Cache: {CACHE}")

p("Step 2: Importing huggingface_hub...")
try:
    from huggingface_hub import snapshot_download
    p("Import OK")
except Exception as e:
    p(f"Import FAILED: {e}")
    sys.exit(1)

p("Step 3: Downloading BAAI/bge-reranker-v2-m3 (~1GB)...")
p("This will take 5-15 minutes. Progress bars will appear below.")
try:
    path = snapshot_download(
        repo_id="BAAI/bge-reranker-v2-m3",
        cache_dir=CACHE,
    )
    p(f"\nDownloaded to: {path}")
except Exception as e:
    p(f"\nDownload FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

p("\nStep 4: Verifying with CrossEncoder load...")
try:
    from sentence_transformers import CrossEncoder
    m = CrossEncoder("BAAI/bge-reranker-v2-m3")
    p("CrossEncoder loaded OK!")
except Exception as e:
    p(f"CrossEncoder load failed: {e}")

p("\nDONE. Restart your server now.")