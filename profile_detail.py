"""Detailliertes Profiling von learn_from_interaction."""
import torch, sys, time
sys.path.insert(0, '/home/anima')
from anima.network import Anima

m = Anima(vocab_size=62, d_model=128, d_sparse=512, d_state=32, n_layers=3).cuda()
x = torch.randint(0, 62, (16, 128))

print('Step 1: learn_from_interaction warmup...')
loss = m.learn_from_interaction(x)
print(f'  loss={loss:.4f}')

# Single step, timed in parts
x = torch.randint(0, 62, (16, 128))
x = x.cuda()

# Part 1: device transfer + forward
torch.cuda.synchronize()
t1 = time.time()
for _ in range(5):
    m.train()
    inp = torch.randint(0, 62, (16, 128)).cuda()
    out, info = m(inp, learn=True)
torch.cuda.synchronize()
t_forward = (time.time() - t1) / 5

# Part 2: forward + loss + memory write
torch.cuda.synchronize()
t1 = time.time()
for _ in range(5):
    inp = torch.randint(0, 62, (16, 128)).cuda()
    loss = m.learn_from_interaction(inp)
torch.cuda.synchronize()
t_full = (time.time() - t1) / 5

print(f'\nForward (learn=True): {t_forward*1000:.0f}ms')
print(f'Full learn_from_interaction: {t_full*1000:.0f}ms')
print(f'Overhead (loss+memory+etc): {(t_full-t_forward)*1000:.0f}ms')

# Test with learn=False
torch.cuda.synchronize()
t1 = time.time()
for _ in range(20):
    inp = torch.randint(0, 62, (16, 128)).cuda()
    m.eval()
    out, _ = m(inp, learn=False)
torch.cuda.synchronize()
t_eval = (time.time() - t1) / 20
print(f'\nForward eval (learn=False): {t_eval*1000:.0f}ms')
print(f'Eval speed: {1/t_eval:.0f} step/s')
