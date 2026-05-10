"""Anima 5000-Step Convergence Test."""
import torch, sys, time
sys.path.insert(0, '/home/anima')
from anima.network import Anima
from anima.data import get_shakespeare_data

device = 'cuda'
torch.cuda.reset_peak_memory_stats()
print('='*60)
print('ANIMA 5000-STEP TRAINING')
print('='*60)

data, stoi, itos, vocab_size = get_shakespeare_data(max_chars=200000)
data = torch.tensor(data, dtype=torch.long)
print(f'Vocab: {vocab_size}, Data: {len(data)}')

model = Anima(vocab_size=vocab_size, d_model=128, d_sparse=512,
              d_state=32, d_context=64, n_layers=3, sparsity=0.02).to(device)
print(f'Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
print()

history = []
t_start = time.time()
B, S = 16, 128

for step in range(5000):
    idx = torch.randint(0, len(data) - B * S, (1,)).item()
    batch = data[idx:idx + B * S].view(B, S).to(device)
    loss = model.learn_from_interaction(batch)
    history.append(loss)

    if step % 500 == 0:
        mem = torch.cuda.max_memory_allocated() / 1024 / 1024
        avg = sum(history[-100:])/len(history[-100:]) if len(history)>=100 else loss
        speed = (step+1)/(time.time()-t_start)
        print(f'Step {step:4d} | loss={avg:.4f} | mem={mem:.0f}MB | {speed:.1f}step/s')
        if step > 0:
            ref = model.self_reflect(loss=loss)
            changes = [c for c in ref['changes'] if c]
            if changes:
                s = model.get_status()
                print(f'  [Meta] lr={s["online_lr"]:.6f}', end='')
                for c in changes[:2]:
                    print(f' | {c[:50]}', end='')
                print()

# Results
print('\n' + '='*60)
print('RESULTS')
print('='*60)
print(f'Loss: {history[0]:.4f} -> {history[-1]:.4f}')
first = sum(history[:200])/200
last = sum(history[-200:])/200
print(f'First 200 avg: {first:.4f} -> Last 200 avg: {last:.4f}')
print(f'Improvement: {(last-first)/first*100:.1f}%')
print(f'Peak VRAM: {torch.cuda.max_memory_allocated()/1024/1024:.0f}MB')
print(f'Time: {time.time()-t_start:.0f}s')

# Generation
model.eval()
print(f'\nGenerating (3 samples)...')
for prompt in ['ROMEO:', 'KING', 'The ']:
    ctx = torch.tensor([[stoi.get(c, 0) for c in prompt]], device=device)
    out = model.generate(ctx, max_new=200, temp=0.8)
    gen = ''.join(itos.get(int(i), '?') for i in out[0])
    print(f'\n[{prompt}]')
    print(f'  {gen[:200]}')

print('\nDONE!')
