"""Check CIC-IDS-2017 column names."""
import os, sys
os.environ["HF_HOME"] = "/home/anima/.hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/home/anima/.hf_cache/hub"

from datasets.splits import SplitInfo
orig_init = SplitInfo.__init__
def patched_init(self, **kwargs):
    kwargs.pop("description", None)
    orig_init(self, **kwargs)
SplitInfo.__init__ = patched_init

from datasets import load_dataset
print("=== bvsam/cic-ids-2017 (machine_learning) ===")
try:
    ds = load_dataset("bvsam/cic-ids-2017", "machine_learning", split="train", streaming=True)
    c = 0
    for ex in ds:
        if c == 0:
            keys = list(ex.keys())
            print(f"Columns ({len(keys)}): {keys}")
            print(f"First values: {[str(v)[:30] for v in list(ex.values())[:10]]}")
        c += 1
        if c >= 3:
            break
    print(f"OK ({c} samples)")
except Exception as e:
    print(f"FAIL: {str(e)[:200]}")

print("\n=== Mireu-Lab/UNSW-NB15 ===")
try:
    ds = load_dataset("Mireu-Lab/UNSW-NB15", split="train", streaming=True)
    c = 0
    for ex in ds:
        if c == 0:
            keys = list(ex.keys())
            print(f"Columns ({len(keys)}): {keys[:20]}...")
            print(f"First values: {[str(v)[:30] for v in list(ex.values())[:10]]}")
        c += 1
        if c >= 3:
            break
    print(f"OK ({c} samples)")
except Exception as e:
    print(f"FAIL: {str(e)[:200]}")
