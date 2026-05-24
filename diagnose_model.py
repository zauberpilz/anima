import torch
import sys
sys.path.insert(0, "/home/anima/src/")
from coglang import CogLang

# Build model with current config
brain = CogLang(use_mixed_precision=True)
brain.build(
    vocab_size=103,
    d_model=1015,
    d_sparse=4096,
    n_layers=12,
    n_heads=4,
    d_state=256,
    d_context=512,
    d_memory=64,
    n_rules=16,
    d_es=8,
    n_skills=8,
    n_motives=4,
    use_mixed_precision=True
)

# Load checkpoint
ckpt = torch.load("/home/anima/checkpoints/checkpoint.pt", map_location="cpu", weights_only=True)
brain.modules.load_state_dict(ckpt["model_state"], strict=True)
brain = brain.cuda()
print("Checkpoint loaded!")

# Generate test input
test_input = torch.randint(0, 103, (2, 64)).cuda()

# Run forward without learn
with torch.no_grad():
    output, info = brain.forward(test_input, learn=False)
    probs = torch.softmax(output, dim=-1)
    mean_prob = probs.mean().item()
    max_prob = probs.max().item()
    min_prob = probs.min().item()
    uniform_prob = 1.0 / 103
    print(f"Output probs: mean={mean_prob:.6f}, max={max_prob:.6f}, min={min_prob:.6f}")
    print(f"Uniform baseline: {uniform_prob:.6f}")
    print(f"Is uniform? {abs(mean_prob - uniform_prob) < 0.0001}")
    
    # Check hidden
    hidden = info.get('hidden', None)
    if hidden is not None:
        print(f"Hidden shape: {hidden.shape}")
        print(f"Hidden values: mean={hidden.mean().item():.6f}, std={hidden.std().item():.6f}, max={hidden.max().item():.6f}")
    
    # Check decoder weights
    print(f"out_head.weight: mean={brain._decoder.out_head.weight.mean().item():.6f}, std={brain._decoder.out_head.weight.std().item():.6f}")
    print(f"out_proj.weight: mean={brain._decoder.out_proj.weight.mean().item():.6f}, std={brain._decoder.out_proj.weight.std().item():.6f}")
    
    # Check stack output
    pred = info.get('pred', None)
    if pred:
        print(f"Prediction[-1] shape: {pred[-1].shape}, mean={pred[-1].mean().item():.6f}, std={pred[-1].std().item():.6f}")

# Run learn
print("\n--- Running learn ---")
loss, info = brain.learn(test_input)
print(f"Loss: {loss:.6f}")
