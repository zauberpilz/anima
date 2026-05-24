"""CogLang memory stability test."""
import torch, sys
sys.path.insert(0, '/home/anima')
from coglang import build_anima

torch.cuda.reset_peak_memory_stats()
brain = build_anima(62, 'cuda')
for p in brain.modules.parameters():
    p.requires_grad_(False)
print(f'VRAM Init: {torch.cuda.memory_allocated()/1024/1024:.0f}MB')

x = torch.randint(0, 62, (16, 128)).cuda()
for i in range(20):
    loss, info = brain.learn(x)
    if i == 0:
        print(f'VRAM max nach Step 0: {torch.cuda.max_memory_allocated()/1024/1024:.0f}MB')
    if i == 0:
        print(f'VRAM aktuell Step 0: {torch.cuda.memory_allocated()/1024/1024:.0f}MB')
    print(f'Step {i}: loss={loss:.4f}')
print(f'VRAM max gesamt: {torch.cuda.max_memory_allocated()/1024/1024:.0f}MB')
print(f'VRAM aktuell Ende: {torch.cuda.memory_allocated()/1024/1024:.0f}MB')
print('OK!')
