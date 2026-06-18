import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

print("1: torch import")
import torch
print(f"   torch {torch.__version__}, cuda={torch.cuda.is_available()}")

print("2: sentence_transformers import")
from sentence_transformers import SentenceTransformer
print("   imported OK")

print("3: finding cached model file")
import pathlib
cache = pathlib.Path.home() / ".cache" / "huggingface" / "hub"
bins = list(cache.glob("**/bge-m3/**/pytorch_model.bin"))
safes = list(cache.glob("**/bge-m3/**/*.safetensors"))
print(f"   .bin files: {len(bins)}")
print(f"   .safetensors files: {len(safes)}")
for f in bins + safes:
    print(f"   {f} ({f.stat().st_size / 1e9:.2f}GB)")

print("4: loading model directly with torch.load")
if bins:
    try:
        import torch
        state = torch.load(str(bins[0]), map_location="cpu", weights_only=False)
        print(f"   loaded OK, {len(state)} keys")
    except Exception as e:
        import traceback
        traceback.print_exc()

print("5: loading with SentenceTransformer")
try:
    m = SentenceTransformer("BAAI/bge-m3", device="cpu")
    print("   loaded OK")
except Exception as e:
    import traceback
    traceback.print_exc()