"""CogLang Quicktest."""
import torch, sys
sys.path.insert(0, '/home/anima')
from coglang import build_anima

torch.cuda.reset_peak_memory_stats()
print('Building Anima in CogLang...')
brain = build_anima(62, 'cuda')
print(f'VRAM: {torch.cuda.memory_allocated()/1024/1024:.0f}MB')

x = torch.randint(0, 62, (16, 128)).cuda()
loss, info = brain.learn(x)
print(f'Loss: {loss:.4f}')
print(f'VRAM max: {torch.cuda.max_memory_allocated()/1024/1024:.0f}MB')
print('OK!')
