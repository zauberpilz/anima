#!/usr/bin/env python3
"""Anima 50K-Step Training — 50M Parameter Modell, 100GB RAM nutzbar."""
import torch, sys, time, json, os
sys.path.insert(0, '/home/anima')
from anima.network import Anima
from anima.data import get_shakespeare_data

device = 'cuda'
torch.cuda.reset_peak_memory_stats()
torch.manual_seed(42)

print('='*60)
print('ANIMA 50K STEP — MEGA-SKALIERT (50M Parameter)')
print('='*60)

data, stoi, itos, vocab_size = get_shakespeare_data(max_chars=200000)
data = torch.tensor(data, dtype=torch.long)
print(f'Vocab: {vocab_size}, Daten: {len(data):,} Tokens')
print(f'RAM verfügbar: >100GB DDR4 + 8GB VRAM')

model = Anima(
    vocab_size=vocab_size,
    d_model=512,
    d_sparse=4096,
    d_state=256,
    d_context=512,
    n_layers=8,
    sparsity=0.02,
    mem_capacity=20000,
).to(device)

params = sum(p.numel() for p in model.parameters())
print(f'Parameter: {params/1e6:.1f}M')
print(f'VRAM Budget: ~{params*4/1024/1024:.0f}MB + ~300MB Aktivierungen')
print()

history = []
t_start = time.time()
B, S = 16, 128
log_interval = 2000

for step in range(50000):
    idx = torch.randint(0, len(data) - B * S, (1,)).item()
    batch = data[idx:idx + B * S].view(B, S).to(device)
    loss = model.learn_from_interaction(batch)
    history.append(loss)

    if step % log_interval == 0:
        mem = torch.cuda.max_memory_allocated() / 1024 / 1024
        avg = sum(history[-500:])/len(history[-500:]) if len(history)>=500 else loss
        speed = (step+1)/(time.time()-t_start)
        pct = step / 50000 * 100
        print(f'[{pct:5.1f}%] Step {step:5d} | loss={avg:.4f} | VRAM={mem:.0f}MB | {speed:.1f}step/s')
        if step > 0:
            ref = model.self_reflect(loss=loss)
            changes = [c for c in ref['changes'] if c]
            if changes:
                s = model.get_status()
                print(f'        lr={s["online_lr"]:.4f} sparsity={s["sparsity"]:.3f}', end='')
                for c in changes[:1]:
                    print(f' | {c[:60]}', end='')
                print()

# Ergebnisse
print('\n' + '='*60)
print('ERGEBNISSE')
print('='*60)
first = sum(history[:1000])/1000
last = sum(history[-1000:])/1000
print(f'Loss-Start: {history[0]:.4f}')
print(f'Loss-Ende:  {history[-1]:.4f}')
print(f'Ø Step 0-1000:    {first:.4f}')
print(f'Ø Step 49000-5K:  {last:.4f}')
print(f'Verbesserung:     {(last-first)/first*100:.1f}%')
print(f'Max VRAM:         {torch.cuda.max_memory_allocated()/1024/1024:.0f}MB')
print(f'Gesamtzeit:       {time.time()-t_start:.0f}s')
print(f'Steps total:      50000')

# Generation
model.eval()
print(f'\n{"="*60}')
print(f'GENERATION (3 Samples)')
print(f'{"="*60}')
for prompt in ['ROMEO:', 'KING ', 'The ']:
    ctx = torch.tensor([[stoi.get(c, 0) for c in prompt]], device=device)
    out = model.generate(ctx, max_new=200, temp=0.8)
    gen = ''.join(itos.get(int(i), '?') for i in out[0])
    print(f'\n[{prompt}]')
    print(f'  {gen[:200]}')
    print()

# Log speichern
log = {
    'steps': 50000,
    'params': params,
    'config': {'d_model': 512, 'd_sparse': 4096, 'd_state': 256,
               'd_context': 512, 'n_layers': 8, 'vocab': vocab_size},
    'loss_first': history[0],
    'loss_last': history[-1],
    'loss_first1k': first,
    'loss_last1k': last,
    'vram_mb': torch.cuda.max_memory_allocated()/1024/1024,
    'time_s': time.time()-t_start,
}
with open('/home/anima/training_50k_log.json', 'w') as f:
    json.dump(log, f, indent=2)

print(f'\nFERTIG! Log: /home/anima/training_50k_log.json')
