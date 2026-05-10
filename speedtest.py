import torch, sys, time
sys.path.insert(0, '/home/anima')
from anima.network import Anima

m = Anima(vocab_size=62, d_model=128, d_sparse=512, d_state=32, n_layers=3).cuda()

# CPU data → learn_from_interaction moves to device
x = torch.randint(0, 62, (16, 128))

torch.cuda.synchronize()
t0 = time.time()
losses = []
for i in range(10):
    x = torch.randint(0, 62, (16, 128))
    loss = m.learn_from_interaction(x)
    losses.append(loss)
torch.cuda.synchronize()
dt = time.time() - t0
print(f'10 steps: {dt:.1f}s = {10/dt:.1f} step/s')
print(f'Losses: min={min(losses):.4f} max={max(losses):.4f} last={losses[-1]:.4f}')
