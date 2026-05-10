"""Anima 10K-Step Training (skaliert ~3M Parameter)."""
import torch, sys, time
sys.path.insert(0, '/home/anima')
from anima.network import Anima
from anima.data import get_shakespeare_data

device = 'cuda'
torch.cuda.reset_peak_memory_stats()
torch.manual_seed(42)

print('='*60)
print('ANIMA 10K STEP — SKALIERT (3M Parameter)')
print('='*60)

data, stoi, itos, vocab_size = get_shakespeare_data(max_chars=200000)
data = torch.tensor(data, dtype=torch.long)
print(f'Vocab: {vocab_size}, Data: {len(data)}')

model = Anima(
    vocab_size=vocab_size, d_model=256, d_sparse=1024,
    d_state=64, d_context=128, n_layers=4, sparsity=0.02,
    mem_capacity=8192, max_seq_len=8192,
).to(device)

params = sum(p.numel() for p in model.parameters())
print(f'Parameter: {params/1e6:.2f}M')
print()

history = []
t_start = time.time()
B, S = 16, 128

for step in range(10000):
    idx = torch.randint(0, len(data) - B * S, (1,)).item()
    batch = data[idx:idx + B * S].view(B, S).to(device)
    loss = model.learn_from_interaction(batch)
    history.append(loss)

    if step % 1000 == 0:
        mem = torch.cuda.max_memory_allocated() / 1024 / 1024
        avg = sum(history[-200:])/len(history[-200:]) if len(history)>=200 else loss
        speed = (step+1)/(time.time()-t_start)
        print(f'Step {step:5d} | loss={avg:.4f} | mem={mem:.0f}MB | {speed:.1f}step/s')
        if step > 0:
            ref = model.self_reflect(loss=loss)
            changes = [c for c in ref['changes'] if c]
            if changes:
                s = model.get_status()
                print(f'  [Meta] lr={s["online_lr"]:.4f}', end='')
                for c in changes[:2]:
                    print(f' | {c[:50]}', end='')
                print()

print('\n' + '='*60)
print('ERGEBNISSE')
print('='*60)
first = sum(history[:500])/500
last = sum(history[-500:])/500
print(f'Loss: {history[0]:.4f} -> {history[-1]:.4f}')
print(f'First 500 avg: {first:.4f} -> Last 500 avg: {last:.4f}')
print(f'Verbesserung: {(last-first)/first*100:.1f}%')
print(f'Max VRAM: {torch.cuda.max_memory_allocated()/1024/1024:.0f}MB')
print(f'Zeit: {time.time()-t_start:.0f}s')

model.eval()
print(f'\nGeneriere 3 Samples...')
for prompt in ['ROMEO:', 'KING', 'The ']:
    ctx = torch.tensor([[stoi.get(c, 0) for c in prompt]], device=device)
    out = model.generate(ctx, max_new=150, temp=0.8)
    gen = ''.join(itos.get(int(i), '?') for i in out[0])
    print(f'\n[{prompt}]')
    print(f'  {gen[:150]}')

print('\nFERTIG!')
