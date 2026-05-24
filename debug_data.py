"""Train a BPE tokenizer on our synthetic domain data."""
import sys, os, torch
sys.path.insert(0, "/home/anima/src/")
from data_loader import MultiDomainDataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

# Load all domain data
loader = MultiDomainDataset(max_chars_per_domain=500000)
loader.load_all()

# Get the raw combined data - it should be token IDs
print(f"Type of loader.data: {type(loader.data)}")
if isinstance(loader.data, torch.Tensor):
    print(f"Shape: {loader.data.shape}, dtype: {loader.data.dtype}")
    print(f"First 10 values: {loader.data[:10].tolist()}")
elif isinstance(loader.data, list):
    print(f"Length: {len(loader.data)}")
    print(f"Type of elements: {type(loader.data[0])}")
    print(f"First 10: {loader.data[:10]}")

# Get text from each domain
all_text = []
for domain_name in ['code', 'security', 'network', 'text']:
    d = loader.domains[domain_name]
    if d['data'] is not None:
        print(f"\n{domain_name} data type: {type(d['data'])}")
        if isinstance(d['data'], torch.Tensor):
            ids = d['data'][:50000].tolist()
            chars = [loader.itos.get(i, '?') for i in ids]
        elif isinstance(d['data'], list) and len(d['data']) > 0:
            # Check if it's chars or IDs
            sample = d['data'][:10]
            print(f"  Sample: {sample}")
            if all(isinstance(x, str) and len(x) == 1 for x in sample):
                chars = d['data'][:50000]
            else:
                ids = [int(x) for x in d['data'][:50000]]
                chars = [loader.itos.get(i, '?') for i in ids]
        else:
            print(f"  Unhandled type: {type(d['data'])}")
            continue
        text = ''.join(chars)
        all_text.append(text)
        print(f"{domain_name}: {len(text)} chars")
