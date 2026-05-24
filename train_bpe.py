"""Train a BPE tokenizer on our synthetic domain data (v3 - fixed data types)."""
import sys, os
sys.path.insert(0, "/home/anima/src/")
from data_loader import MultiDomainDataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

# Load all domain data (don't need augmentation for tokenizer training)
loader = MultiDomainDataset(max_chars_per_domain=50000)
loader.load_all()

# Collect raw text from each domain (data is stored as raw strings)
all_text = []
for domain_name in ['code', 'security', 'network', 'text']:
    d = loader.domains[domain_name]
    if d['data'] is not None and isinstance(d['data'], str) and len(d['data']) > 100:
        all_text.append(d['data'])
        print(f"{domain_name}: {len(d['data'])} chars")

# Also add the full combined data decoded via itos
if loader.data is not None:
    ids = loader.data.tolist()
    chars = [loader.itos.get(i, '?') for i in ids[:200000]]
    all_text.append(''.join(chars))
    print(f"combined: {len(chars)} chars")

print(f"\nTotal training text: {sum(len(t) for t in all_text)} chars")

# Train BPE tokenizer
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

trainer = trainers.BpeTrainer(
    vocab_size=4096,
    special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"],
    min_frequency=2,
    show_progress=True,
)

print("\nTraining BPE tokenizer...")
tokenizer.train_from_iterator(all_text, trainer)
print(f"Vocab size: {tokenizer.get_vocab_size()}")

# Save
os.makedirs("/home/anima/tokenizer", exist_ok=True)
tokenizer.save("/home/anima/tokenizer/bpe_4k.json")
print("Tokenizer saved!")

# Test
test_texts = [
    "def fibonacci(n):",
    "The future of AI is",
    "CVE-2024-1234 vulnerability",
    "[FLOW] src=192.168.1.1 dst=10.0.0.1",
]
print("\n=== Tokenizer Test ===")
for t in test_texts:
    encoded = tokenizer.encode(t)
    decoded = tokenizer.decode(encoded.ids)
    print(f"  Input:    {t}")
    print(f"  Tokens:   {encoded.tokens}")
    print(f"  IDs:      {encoded.ids}")
    print(f"  Decoded:  {decoded}")
    print(f"  Match:    {t == decoded}")
    print()

# Compression test
long_text = "def check_buffer_overflow(data):\n    if len(data) > 1024:\n        return True\n    return False"
encoded = tokenizer.encode(long_text)
decoded = tokenizer.decode(encoded.ids)
print(f"Compression: {len(long_text)} chars -> {len(encoded.ids)} tokens ({len(long_text)/max(1,len(encoded.ids)):.1f}x)")
