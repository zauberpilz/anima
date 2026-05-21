"""Cobra Training — 50K Steps mit ETA und Live-Output."""
import torch, torch.nn.functional as F, sys, time
sys.path.insert(0, '/home/anima')
from coglang import build_anima
from anima.data import get_shakespeare_data

device = 'cuda'
torch.cuda.reset_peak_memory_stats()
torch.manual_seed(42)

print('='*60)
print('COBRA — 50K STEP TRAINING')
print('='*60)
print('W_pred: NLMS Hebbian | W_error: LMS Hebbian (0.2x) | Gate: frozen')
print()

data, stoi, itos, vocab_size = get_shakespeare_data(max_chars=200000)
if isinstance(data, torch.Tensor): data = data.long()
else: data = torch.tensor(data, dtype=torch.long)
print(f'Vocab: {vocab_size}, Daten: {len(data):,} Tokens')

brain = build_anima(vocab_size=vocab_size, device=device)
print(f'VRAM nach Init: {torch.cuda.memory_allocated()/1024/1024:.0f}MB\n')

history = []
t_start = time.time()
B, S = 16, 128
last_log = 0
log_interval = 100

for step in range(50000):
    idx = torch.randint(0, len(data) - B * S, (1,)).item()
    batch = data[idx:idx + B * S].view(B, S).to(device)
    loss, _ = brain.learn(batch)
    history.append(loss)

    if step % 2000 == 0:
        mem = torch.cuda.max_memory_allocated() / 1024 / 1024
        avg = sum(history[-500:]) / 500 if len(history) >= 500 else loss
        elapsed = time.time() - t_start
        speed = (step + 1) / elapsed if elapsed > 0 else 0
        pct = step / 50000 * 100
        remaining_steps = 50000 - step
        eta_secs = remaining_steps / speed if speed > 0 else 0
        eta_m, eta_s = divmod(int(eta_secs), 60)
        elapsed_m, elapsed_s = divmod(int(elapsed), 60)
        status = 'OK' if loss == loss else 'NAN'
        print(f'[{pct:5.1f}%] Step {step:5d} | loss={avg:.4f} ({status}) | '
              f'VRAM={mem:.0f}MB | {speed:.1f}step/s | '
              f'+{elapsed_m:02d}:{elapsed_s:02d} | ETA {eta_m:02d}:{eta_s:02d}')
        last_log = step
        if loss != loss:
            print('NaN — Abbruch')
            break

print('\n' + '='*60)
print('ERGEBNISSE')
print('='*60)
print(f'Loss Start: {history[0]:.4f}')
vals = [v for v in history if v == v]
if vals:
    print(f'Loss Ende:  {vals[-1]:.4f}')
    print(f'Bester:     {min(vals):.4f}')
elapsed = time.time() - t_start
elapsed_m, elapsed_s = divmod(int(elapsed), 60)
print(f'Max VRAM:   {torch.cuda.max_memory_allocated()/1024/1024:.0f}MB')
print(f'Zeit:       {elapsed_m:02d}:{elapsed_s:02d}')

if vals and vals[-1] == vals[-1]:
    print('\nGeneration:')
    for prompt in ['ROMEO:', 'KING ']:
        ctx = torch.tensor([[stoi.get(c, 0) for c in prompt]], device=device)
        for _ in range(200):
            out, info = brain.forward(ctx[:, -128:], learn=False)
            probs = F.softmax(out[:, -1, :] / 0.8, dim=-1)
            ctx = torch.cat([ctx, torch.multinomial(probs, 1)], dim=-1)
        gen = ''.join(itos.get(int(i), '?') for i in ctx[0])
        print(f'[{prompt}] {gen[:150]}')
        gen_path = f'/home/anima/gen_step_{step}.txt'
        with open(gen_path, 'w') as f:
            f.write(f'Prompt: {prompt}\n{gen}')
        print(f'  (gespeichert: {gen_path})')

print('\nFERTIG!')
