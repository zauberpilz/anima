import torch, sys
sys.path.insert(0, '/home/anima')
from anima.network import Anima

model = Anima(vocab_size=61, d_model=128, d_sparse=512, d_state=32, n_layers=2).cuda()
x = torch.randint(0, 61, (2, 64))
loss = model.learn_from_interaction(x)
print(f'Forward OK, loss={loss:.4f}')
print('Device test passed!')
