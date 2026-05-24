"""Schnelltest: 50M Parameter Modell initialisieren + 1 Step."""
import torch, sys
sys.path.insert(0, '/home/anima')
from anima.network import Anima

torch.cuda.reset_peak_memory_stats()
print('Initialisiere 50M Modell...')
m = Anima(vocab_size=62, d_model=512, d_sparse=4096,
          d_state=256, d_context=512, n_layers=8).cuda()
p = sum(p.numel() for p in m.parameters())
print(f'Parameter: {p/1e6:.1f}M')
mem = torch.cuda.memory_allocated()/1024/1024
print(f'VRAM nach Init: {mem:.0f}MB')

x = torch.randint(0, 62, (16, 128)).cuda()
loss = m.learn_from_interaction(x)
mem2 = torch.cuda.max_memory_allocated()/1024/1024
print(f'VRAM max: {mem2:.0f}MB')
print(f'Loss: {loss:.4f}')
print('OK!')
