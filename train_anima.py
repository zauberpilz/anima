import torch, sys, time
sys.path.insert(0, '/home/anima')
from anima.network import Anima
from anima.trainer import AnimaTrainer
from anima.data import get_shakespeare_data, SequenceLoader

device = 'cuda'
torch.cuda.reset_peak_memory_stats()

print('Lade Daten...')
data, stoi, itos, vocab_size = get_shakespeare_data(max_chars=100000)
print(f'Vocab: {vocab_size}, Daten: {len(data)} Tokens')

model = Anima(
    vocab_size=vocab_size,
    d_model=256,
    d_sparse=1024,
    d_state=64,
    d_context=128,
    n_layers=4,
    sparsity=0.02,
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f'Parameter: {n_params/1e6:.2f}M')

trainer = AnimaTrainer(model, lr=0.005, device=device)

loader = SequenceLoader(data[:90000], block_size=128, batch_size=32)

print('\nStarte Training (Predictive Coding + lokales Lernen)...')
print('=' * 60)

t0 = time.time()
for step, batch in enumerate(loader):
    if step >= 100:
        break

    metrics = trainer.train_step(batch)
    mem = torch.cuda.max_memory_allocated() / 1024 / 1024

    if step % 10 == 0:
        dt = time.time() - t0
        print(f'Step {step:3d} | pred_loss={metrics["predictive_loss"]:.4f} | '
              f'recon_loss={metrics["recon_loss"]:.4f} | '
              f'density={metrics["density"]:.3f} | '
              f'mem={mem:.0f}MB | time={dt:.1f}s')
        t0 = time.time()

    if step > 0 and step % 50 == 0:
        suggestions = model.get_optimization_suggestions()
        for s in suggestions:
            if 'stabil' not in s:
                print(f'  [Meta] {s}')

print('\n=== Training abgeschlossen ===')
print(f'Spitzen VRAM: {torch.cuda.max_memory_allocated()/1024/1024:.0f} MB')

# Generierung
print('\nGeneriere Text...')
model.eval()
prompt = torch.tensor([[stoi[c] for c in 'ROMEO:']], device=device)
out = model.generate(prompt, max_new_tokens=200, temperature=0.8)
gen = ''.join(itos[int(i)] for i in out[0])
print(f'\n{gen}')
