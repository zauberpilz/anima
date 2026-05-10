#!/bin/bash
source ~/venv/bin/activate
cd /home/anima
python << 'PYTHON_SCRIPT'
import torch, sys, time, math
sys.path.insert(0, '/home/anima')
from anima.network import Anima
from anima.trainer import AnimaTrainer
from anima.data import get_shakespeare_data, SequenceLoader

device = 'cuda'
torch.cuda.reset_peak_memory_stats()

print('Lade Daten...')
data, stoi, itos, vocab_size = get_shakespeare_data()

model = Anima(
    vocab_size=vocab_size, d_model=256, d_sparse=1024,
    d_state=64, d_context=128, n_layers=4, sparsity=0.02,
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f'Parameter: {n_params/1e6:.2f}M')

trainer = AnimaTrainer(model, lr=0.005, device=device)

loader = SequenceLoader(data[:300000], block_size=128, batch_size=32)

print('\nTraining...')
t0 = time.time()
for step, batch in enumerate(loader):
    if step >= 500: break
    metrics = trainer.train_step(batch)
    mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    if step % 50 == 0:
        dt = time.time() - t0
        ppl = math.exp(metrics['recon_loss'])
        print(f'Step {step:4d} | recon={metrics["recon_loss"]:.4f} | ppl={ppl:.2f} | mem={mem:.0f}MB | {dt:.1f}s')
        t0 = time.time()

print(f'\nDone. Peak VRAM: {torch.cuda.max_memory_allocated()/1024/1024:.0f}MB')

model.eval()
ctx = torch.tensor([[stoi[c] for c in 'ROMEO:']], device=device)
out = model.generate(ctx, max_new_tokens=500, temperature=0.8)
gen = ''.join(itos[int(i)] for i in out[0])
print(f'\nGenerated:\n{gen}')

torch.save(model.state_dict(), '/home/anima/anima_model.pt')
print('Model saved!')
PYTHON_SCRIPT
