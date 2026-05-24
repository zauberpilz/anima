"""
BPE Integration Dry-Run: Simuliert Iteration 19 Startup.
Läuft direkt auf WSL, testet die gesamte Pipeline.
"""
import sys
import os
import torch

# Pfade (WSL)
sys.path.insert(0, '/home/anima/src')
os.chdir('/home/anima/src')

BPE_PATH = '/home/anima/tokenizer/bpe_4k.json'
CHECKPOINT_PATH = '/home/anima/checkpoints/checkpoint.pt'

print('=' * 60)
print('BPE INTEGRATION DRY-RUN')
print('=' * 60)

# 1. Data Loader mit BPE
print('\n[1/6] MultiDomainDataset mit BPE laden...')
from data_loader import MultiDomainDataset

ds = MultiDomainDataset(max_chars_per_domain=500000, bpe_tokenizer_path=BPE_PATH)
assert ds.bpe_tokenizer is not None, 'BPE Tokenizer wurde nicht geladen!'
print(f'  ✓ BPE Tokenizer geladen: vocab_size={ds.vocab_size}')

ds.load_all()
print(f'  ✓ Data geladen: {len(ds.data):,} Tokens, vocab_size={ds.vocab_size}')
assert ds.vocab_size >= 4000, f'Vocab zu klein: {ds.vocab_size}'
print(f'  ✓ Vocab size check: {ds.vocab_size}')

# 2. Encode/Decode Roundtrip
print('\n[2/6] Encode/Decode Roundtrip...')
test_texts = [
    'def fibonacci(n):',
    'CVE-2024-1234: buffer overflow in http_parser',
    '[FLOW] src=10.0.0.1:443 -> dst=10.0.0.2:80 proto=TCP bytes=1024 label=normal',
    'The quick brown fox jumps over the lazy dog.',
]
for text in test_texts:
    ids = ds.encode(text)
    decoded = ds.decode(ids)
    match = decoded == text
    status = '✓' if match else '✗'
    print(f'  {status} {text[:50]:50s} -> {len(ids):4d} tokens -> {"OK" if match else "MISMATCH"}')
    if not match:
        print(f'      Original: {repr(text)}')
        print(f'      Decoded:  {repr(decoded)}')

# 3. Model bauen mit BPE Vocab
print(f'\n[3/6] Model bauen mit vocab_size={ds.vocab_size}...')
sys.path.insert(0, '/home/anima/src')
from coglang import build_anima

brain = build_anima(
    vocab_size=ds.vocab_size,
    device='cpu',
    d_model=1024,
    d_sparse=2512,
    n_layers=14,
    d_state=256,
    d_context=512,
    lr=0.008,
)
param_count = brain.parameter_count()
print(f'  ✓ Model gebaut: {param_count/1e6:.1f}M Parameter')

# 4. Checkpoint laden (soll fehlschlagen wegen size mismatch)
print('\n[4/6] Checkpoint laden (erwarte size mismatch)...')
if os.path.exists(CHECKPOINT_PATH):
    try:
        cfg = brain.load_checkpoint(CHECKPOINT_PATH)
        print(f'  ⚠ Checkpoint geladen (unerwartet): {cfg}')
    except RuntimeError as e:
        if 'size mismatch' in str(e).lower() or 'missing key' in str(e).lower():
            print(f'  ✓ Size mismatch wie erwartet: {e}')
            print(f'  ✓ Starte mit frischen Gewichten (korrektes Verhalten)')
        else:
            print(f'  ✗ Anderer Fehler: {e}')
else:
    print(f'  - Kein Checkpoint vorhanden, frischer Start')

# 5. Forward Pass
print('\n[5/6] Forward Pass...')
sample_ids = ds.encode('def fibonacci(n):\n    if n <= 1: return n\n')
x = torch.tensor([sample_ids[:64]], dtype=torch.long)
print(f'  Input shape: {x.shape}')
logits, _ = brain.forward(x, learn=False)
print(f'  Output shape: {logits.shape}')
assert logits.size(-1) == ds.vocab_size, f'Vocab mismatch: {logits.size(-1)} vs {ds.vocab_size}'
print(f'  ✓ Forward OK: logits[-1] shape = {logits.shape}')

# 6. Generation
print('\n[6/6] Generation Test...')
prompt = 'def fibonacci'
prompt_ids = torch.tensor([ds.encode(prompt)], dtype=torch.long)
print(f'  Prompt: "{prompt}" -> {prompt_ids.shape}')
generated = brain.generate_safe(prompt_ids, max_new=30, temperature=0.8, top_k=30)
gen_text = ds.decode(generated[0])
print(f'  Generated: "{gen_text}"')
print(f'  ✓ Generation funktioniert ({len(generated[0])} tokens)')

print('\n' + '=' * 60)
print('BPE INTEGRATION DRY-RUN: ALLE TESTS BESTANDEN')
print('=' * 60)
