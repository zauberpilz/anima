"""Debug-Anima: Finde den Flaschenhals."""
import torch, sys, time
sys.path.insert(0, '/home/anima')
from anima.network import Anima
from anima.data import get_shakespeare_data

device = 'cuda'
data, stoi, itos, vocab_size = get_shakespeare_data(max_chars=200000)
data = torch.tensor(data, dtype=torch.long)

B, S = 16, 128

model = Anima(vocab_size=vocab_size, d_model=128, d_sparse=512,
              d_state=32, d_context=64, n_layers=3, sparsity=0.02).to(device)

# Train 200 steps first
print('Training 200 steps...')
for step in range(200):
    idx = torch.randint(0, len(data) - B * S, (1,)).item()
    batch = data[idx:idx + B * S].view(B, S).to(device)
    model.learn_from_interaction(batch)

print('\n=== BOTTLENECK ANALYSIS ===')
idx = torch.randint(0, len(data) - B * S, (1,)).item()
x = data[idx:idx + B * S].view(B, S).to(device)

# 1. Sparse encoder analysis
model.eval()
with torch.no_grad():
    emb = model.embed(x)
    print(f'\n1. EMBEDDING:')
    print(f'   shape={emb.shape}, mean={emb.mean():.3f}, std={emb.std():.3f}')
    
    sparse_x = model.sparse_enc(emb)
    print(f'\n2. SPARSE ENCODER:')
    print(f'   shape={sparse_x.shape}')
    print(f'   sparsity={(sparse_x!=0).float().mean():.4f} ({model.sparsity:.2%} target)')
    print(f'   active_mean={sparse_x[sparse_x!=0].mean():.3f}')
    print(f'   active_std={sparse_x[sparse_x!=0].std():.3f}')
    print(f'   min={sparse_x.min():.3f}, max={sparse_x.max():.3f}')

    pos = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
    ctx = model.context_embed(pos)
    print(f'\n3. CONTEXT EMBED:')
    print(f'   shape={ctx.shape}, mean={ctx.mean():.3f}, std={ctx.std():.3f}')

    # Predictive stack
    errors, states, predictions = model.predictive_stack(sparse_x, ctx, learn=False)
    for i, (e, s, p) in enumerate(zip(errors, states, predictions)):
        print(f'\n4. PREDICTIVE LAYER {i}:')
        print(f'   error shape={e.shape}, norm={e.norm():.1f}, mean={e.mean():.3f}, std={e.std():.3f}')
        print(f'   pred shape={p.shape}, mean={p.mean():.3f}, std={p.std():.3f}')
        print(f'   state shape={s.shape}, mean={s.mean():.3f}, std={s.std():.3f}')
        # Correlation between error and input
        if i > 0:
            prev_e = errors[i-1]
            corr = (e * prev_e).mean()
            print(f'   cross-correlation with prev error: {corr:.4f}')
    
    # Output decoder
    pred = predictions[0]
    hidden = model.out_proj(pred)
    output = model.out_head(hidden)
    print(f'\n5. OUTPUT DECODER:')
    print(f'   pred -> hidden: {pred.shape} -> {hidden.shape}, mean={hidden.mean():.3f}')
    print(f'   hidden -> output: {hidden.shape} -> {output.shape}')
    print(f'   out_head.weight range: [{model.out_head.weight.min():.3f}, {model.out_head.weight.max():.3f}]')
    print(f'   out_proj.weight range: [{model.out_proj.weight.min():.3f}, {model.out_proj.weight.max():.3f}]')
    
    # Loss breakdown
    logits = output
    probs = torch.softmax(logits, dim=-1)
    target_probs = torch.softmax(torch.zeros_like(logits), dim=-1)  # uniform
    uniform_ce = -torch.log(probs).mean()
    print(f'\n6. LOSS ANALYSIS:')
    print(f'   Actual loss: {torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), x.view(-1)):.4f}')
    print(f'   Uniform prediction loss: {uniform_ce:.4f}')
    
    # Check if output is learning ANYTHING
    top_probs, top_idx = probs.topk(5, dim=-1)
    most_common = top_idx[0, 0, 0].item()
    print(f'\n7. OUTPUT BIAS CHECK:')
    print(f'   Most common prediction for first token: {itos[most_common]} (idx={most_common})')
    # Count most frequent char in data
    char_counts = torch.bincount(data, minlength=vocab_size)
    top_char = char_counts.argmax().item()
    print(f'   Most common char in data: {itos[top_char]} (idx={top_char})')
    print(f'   Top-5 chars: {[itos[i.item()] for i in char_counts.topk(5).indices]}')

# Check if predictive weights are changing
print(f'\n8. WEIGHT EVOLUTION:')
for i, layer in enumerate(model.predictive_stack.layers):
    w = layer.W_pred.weight
    print(f'   Layer {i} W_pred range: [{w.min():.3f}, {w.max():.3f}], mean={w.mean():.3f}, std={w.std():.3f}')
    w_e = layer.W_error.weight
    print(f'   Layer {i} W_error range: [{w_e.min():.3f}, {w_e.max():.3f}], mean={w_e.mean():.3f}')

# Compare with a completely untrained model
print(f'\n9. COMPARISON WITH UNTRAINED:')
model0 = Anima(vocab_size=vocab_size, d_model=128, d_sparse=512,
               d_state=32, d_context=64, n_layers=3, sparsity=0.02).to(device)
model0.eval()
with torch.no_grad():
    out0, _ = model0(x, learn=False)
    loss0 = torch.nn.functional.cross_entropy(out0.view(-1, out0.size(-1)), x.view(-1))
    print(f'   Untrained loss: {loss0:.4f}')
    out_t, _ = model(x, learn=False)
    loss_t = torch.nn.functional.cross_entropy(out_t.view(-1, out_t.size(-1)), x.view(-1))
    print(f'   Trained (200 steps) loss: {loss_t:.4f}')
    print(f'   Improvement: {(loss0-loss_t):.4f}')

# Time breakdown
print(f'\n10. TIME BREAKDOWN (per step, eval):')
model.eval()
torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    b = torch.randint(0, len(data) - B * S, (1,)).item()
    inp = data[b:b + B * S].view(B, S).to(device)
    out, info = model(inp, learn=False)
torch.cuda.synchronize()
print(f'   Forward eval: {(time.time()-t0)/100*1000:.1f}ms')

model.train()
torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    b = torch.randint(0, len(data) - B * S, (1,)).item()
    inp = data[b:b + B * S].view(B, S).to(device)
    loss = model.learn_from_interaction(inp)
torch.cuda.synchronize()
print(f'   learn_from_interaction: {(time.time()-t0)/100*1000:.1f}ms')

print('\nDone!')
