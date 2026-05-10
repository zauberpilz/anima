import torch, sys, time
sys.path.insert(0, '/home/anima')
from anima.network import Anima

model = Anima(vocab_size=62, d_model=128, d_sparse=512, d_state=32, n_layers=3).cuda()
x = torch.randint(0, 62, (16, 128)).cuda()

# Forward only (no learn, no backward)
torch.cuda.synchronize()
t0 = time.time()
for i in range(20):
    model(x, learn=False)
torch.cuda.synchronize()
dt_fwd = (time.time() - t0) / 20
print(f'Forward only: {dt_fwd*1000:.1f}ms ({1/dt_fwd:.1f} step/s)')

# Forward with Hebbian but no backward
torch.cuda.synchronize()
t0 = time.time()
for i in range(20):
    model(x, learn=True)
torch.cuda.synchronize()
dt_hebb = (time.time() - t0) / 20
print(f'Forward+Hebbian: {dt_hebb*1000:.1f}ms ({1/dt_hebb:.1f} step/s)')

# Full learn_from_interaction (forward + hebbian + backward)
torch.cuda.synchronize()
t0 = time.time()
for i in range(10):
    model.learn_from_interaction(x)
torch.cuda.synchronize()
dt_full = (time.time() - t0) / 10
print(f'Full learn: {dt_full*1000:.1f}ms ({1/dt_full:.1f} step/s)')

print(f'\nBreakdown:')
print(f'  Forward:          {dt_fwd*1000:.1f}ms')
print(f'  + Hebbian:        {dt_hebb*1000:.1f}ms ({(dt_hebb-dt_fwd)*1000:.1f}ms Hebbian)')
print(f'  + Backward:       {dt_full*1000:.1f}ms ({(dt_full-dt_hebb)*1000:.1f}ms Backward)')
