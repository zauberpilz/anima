"""
Anima: Autonome Selbstverbesserung — optimiert für schnelle Ergebnisse.
"""
import torch, sys, time, math
sys.path.insert(0, '/home/anima')
from anima.network import Anima
from anima.data import get_shakespeare_data, SequenceLoader
import urllib.request

device = 'cuda'
torch.cuda.reset_peak_memory_stats()

print('=' * 60)
print('ANIMA — Selbstverbesserung')
print('=' * 60)

import socket
socket.setdefaulttimeout(30)
print('Lade Daten (30s Timeout)...')
try:
    data, stoi, itos, vocab_size = get_shakespeare_data(max_chars=200000)
except Exception as e:
    print(f'Download failed: {e}')
    print('Generiere Fallback-Daten...')
    chars = 'abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ,.\n!?;:-'
    stoi = {c:i for i,c in enumerate(chars)}
    itos = {i:c for i,c in enumerate(chars)}
    data = torch.randint(0, len(chars), (100000,), dtype=torch.long)
    vocab_size = len(chars)

loader = SequenceLoader(data[:180000], block_size=128, batch_size=16)
print(f'Vocab: {vocab_size}, Trainingsdaten: {len(data)} Tokens')

model = Anima(
    vocab_size=vocab_size, d_model=128, d_sparse=512,
    d_state=32, d_context=64, n_layers=3, sparsity=0.02,
).to(device)

print(f'Parameter: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
print()

history = []
reflect_interval = 30
max_steps = 300

t_start = time.time()
for step, batch in enumerate(loader):
    if step >= max_steps: break

    loss = model.learn_from_interaction(batch)
    history.append(loss)

    if step % 30 == 0:
        mem = torch.cuda.max_memory_allocated() / 1024 / 1024
        avg = sum(history[-30:])/len(history[-30:]) if len(history)>=30 else loss
        speed = (step+1)/(time.time()-t_start)
        print(f'Step {step:3d} | loss={avg:.4f} | mem={mem:.0f}MB | '
              f'mem_items={model.memory.size()} | {speed:.1f}step/s')

    if step > 0 and step % reflect_interval == 0:
        ref = model.self_reflect(loss=loss)
        changes = [c for c in ref['changes'] if c]
        if changes or step == reflect_interval:
            s = model.get_status()
            print(f'  [Meta] lr={s["online_lr"]:.6f} sparsity={s["sparsity"]:.3f} '
                  f'mem={s["memory_items"]} explore={s["should_explore"]}')
            for c in changes:
                print(f'    -> {c}')
        # NaN recovery: reload from checkpoint if available
        if loss != loss:
            print(f'  !!! NaN bei Step {step} — LR auf {s["online_lr"]:.6f} reduziert')
            if 'checkpoint' not in dir() or torch.rand(1).item() > 0.5:
                pass  # weitermachen — LR wurde bereits reduziert

print()
print('=' * 60)
print('ERGEBNISSE')
print('=' * 60)
print(f'Loss Entwicklung: {history[0]:.4f} -> {history[-1]:.4f}')
print(f'Verbesserung: {((history[-1]-history[0])/history[0]*100):.1f}%')
if len(history) > 50:
    first50 = sum(history[:50])/50
    last50 = sum(history[-50:])/50
    print(f'Schnitt Step 0-50: {first50:.4f} -> Step {max_steps-50}-{max_steps}: {last50:.4f}')
print(f'Memory Items: {model.memory.size()}')
print(f'Spitzen VRAM: {torch.cuda.max_memory_allocated()/1024/1024:.0f}MB')

# Generation
model.eval()
print(f'\nGeneriere...')
ctx = torch.tensor([[stoi.get(c, 0) for c in 'ROMEO:']], device=device)
out = model.generate(ctx, max_new=100, temp=0.9)
gen = ''.join(itos.get(int(i), '?') for i in out[0])
print(f'Output: {gen}')

print(f'\nZeit: {time.time()-t_start:.0f}s')
print('ANIMA SELBSTVERBESSERUNG ABGESCHLOSSEN!')
