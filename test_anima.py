import torch, sys
sys.path.insert(0, '/home/anima')
from anima.network import Anima

device = 'cuda'
model = Anima(vocab_size=256, d_model=128, d_sparse=512, d_state=32, n_layers=3).to(device)

x = torch.randint(0, 256, (2, 64), device=device)
out = model(x)
print('Output shape:', out.shape)

status = model.get_status()
print('Status:', status)

suggestions = model.get_optimization_suggestions()
print('Suggestions:', suggestions[:2])

mem = torch.cuda.max_memory_allocated() / 1024 / 1024
print('GPU Memory:', round(mem, 1), 'MB')
print('ANIMA OK!')
