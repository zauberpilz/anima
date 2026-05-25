"""Diagnose NaN origin in forward pass."""
import os, sys, torch
sys.path.insert(0, "/home/anima/src")
os.chdir("/home/anima/src")

from coglang import build_anima, CogModule
from data_loader import MultiDomainDataset

device = torch.device("cuda")
ds = MultiDomainDataset(max_chars_per_domain=500000, bpe_tokenizer_path="/home/anima/tokenizer/bpe_4k.json")
ds.load_all()

brain = build_anima(vocab_size=ds.vocab_size, device=device,
    d_model=1024, d_sparse=1812, n_layers=15, d_state=256, d_context=512, lr=0.05)

ckpt = torch.load("/home/anima/checkpoints/checkpoint.pt", map_location="cpu")
state = brain.modules.state_dict()
for k, v in ckpt["model_state"].items():
    if k in state and state[k].shape == v.shape:
        state[k].copy_(v)

B, S = 1, 16
x, y = ds.get_batch("text", B, S, device)
print(f"x: {x.shape}")

with torch.no_grad():
    emb = brain._sensory.embed(x)
    print(f"emb: [{emb.min():.4f}, {emb.max():.4f}]")
    
    sparse_x = brain._encoder(emb)
    print(f"sparse_x: [{sparse_x.min():.4f}, {sparse_x.max():.4f}] nan={torch.isnan(sparse_x).any()}")
    
    # Check all sparse encoder weights
    print(f"encoder proj: [{brain._encoder.proj.weight.min():.4f}, {brain._encoder.proj.weight.max():.4f}]")
    print(f"encoder norm weight: [{brain._encoder.norm.weight.min():.4f}, {brain._encoder.norm.weight.max():.4f}]")
    
    current = sparse_x
    
    # Forward through layers using actual CogLang.forward
    output, info = brain.forward(x, learn=False)
    print(f"forward output: [{output.min():.4f}, {output.max():.4f}] nan={torch.isnan(output).any()}")
    
    if "errors" in info:
        for i, e in enumerate(info["errors"]):
            nan = torch.isnan(e).any().item()
            inf = torch.isinf(e).any().item()
            print(f"  L{i} error: [{e.min():.4f}, {e.max():.4f}] nan={nan} inf={inf}")
            
    if "states" in info:
        for i, s in enumerate(info["states"]):
            nan = torch.isnan(s).any().item()
            print(f"  L{i} state: [{s.min():.4f}, {s.max():.4f}] nan={nan}")
    
    # Now check HebbianAttn weights
    ha = brain._stack.hebbian_attn
    for name in ['W_q', 'W_k', 'W_v', 'W_out']:
        w = getattr(ha, name).weight
        print(f"hebbian_attn.{name}: [{w.min():.4f}, {w.max():.4f}] nan={torch.isnan(w).any()}")
    
    # Check layer weights
    for i in range(3):
        l = brain._stack.layers[i]
        for name in ['W_pred', 'W_error', 'W_gate']:
            w = getattr(l, name).weight
            print(f"L{i}.{name}: [{w.min():.4f}, {w.max():.4f}] nan={torch.isnan(w).any()}")
    
    # Check PredictiveAttention weights
    pa = brain._stack.attention
    for name in ['W_q', 'W_k', 'W_v', 'W_out']:
        w = getattr(pa, name).weight
        print(f"pred_attn.{name}: [{w.min():.4f}, {w.max():.4f}] nan={torch.isnan(w).any()}")

print("DONE")
