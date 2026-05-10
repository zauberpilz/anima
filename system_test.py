import torch, sys, time
sys.path.insert(0, '/home/anima')
from anima.network import Anima

device = 'cuda'
torch.cuda.reset_peak_memory_stats()
print('ANIMA System Test')
print('=' * 40)

model = Anima(vocab_size=61, d_model=128, d_sparse=512, d_state=32, d_context=64, n_layers=3, sparsity=0.02).to(device)
print('Params:', round(sum(p.numel() for p in model.parameters())/1e6, 2), 'M')

x = torch.randint(0, 61, (2, 64), device=device)
out, info = model(x)
print('Forward OK | shape:', list(out.shape), '| density:', round(info['density'], 3))
mem = torch.cuda.max_memory_allocated() / 1024 / 1024
print('VRAM:', round(mem), 'MB')

print('Online Learning...')
for i in range(10):
    x = torch.randint(0, 61, (4, 128), device=device)
    loss = model.learn_from_interaction(x)
    if i in [0, 9]:
        print(f'  step {i}: loss={loss:.4f}')
print('Buffer:', len(model.experience_buffer), '| Memory:', model.memory.size())

print('Self-Reflection...')
ref = model.self_reflect()
for s in ref['suggestions']:
    print('  ->', s)
print('Online LR:', round(ref['online_lr'], 6))

print('Safety Core...')
safe = model.safety
for text, exp in [('Frieden', True), ('Kill them!', False), ('Liebe', True), ('Torture', False)]:
    ok, _ = safe.validate(text, True)
    print(f'  {"OK" if ok==exp else "FAIL"}: {text}')

print('Generation...')
x = torch.randint(0, 61, (1, 32), device=device)
t0 = time.time()
out = model.generate(x, max_new=100, temp=1.0)
print(f'  {100/(time.time()-t0):.0f} tok/s')

print('Dual-Path...')
x = torch.randint(0, 61, (1, 64), device=device)
t0 = time.time()
model(x, learn=False, slow=False)
t_fast = time.time() - t0
t0 = time.time()
model(x, learn=False, slow=True)
t_slow = time.time() - t0
print(f'  Fast: {t_fast*1000:.0f}ms | Slow: {t_slow*1000:.0f}ms | Ratio: {t_slow/t_fast:.1f}x')

print()
print('STATUS')
for k, v in model.get_status().items():
    print(f'  {k}: {v}')
print('Peak VRAM:', round(torch.cuda.max_memory_allocated()/1024/1024), 'MB')
print('ANIMA LÄUFT!')
