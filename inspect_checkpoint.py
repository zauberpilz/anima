import torch

ckpt = torch.load("/home/anima/checkpoints/checkpoint.pt", map_location="cpu", weights_only=True)
print("Keys in checkpoint:", list(ckpt.keys()))
ms = ckpt["model_state"]
print("Model state keys (first 15):", list(ms.keys())[:15])
print("Total model state keys:", len(ms))

if "iteration" in ckpt:
    print("Iteration:", ckpt["iteration"])
if "loss" in ckpt:
    print("Loss:", ckpt["loss"])
if "step" in ckpt:
    print("Step:", ckpt["step"])

for k in list(ms.keys())[:5]:
    print(f"  {k}: {ms[k].shape}")

stack_keys = [k for k in ms if k.startswith("2.")]
layer_indices = set()
for k in stack_keys:
    parts = k.split(".")
    if len(parts) >= 3:
        layer_indices.add(int(parts[1]))
print(f"Stack keys: {len(stack_keys)}, Layer indices: {sorted(layer_indices)}")
for k in sorted(stack_keys)[:6]:
    print(f"  {k}: {ms[k].shape}")

# Check decoder (module 3) keys
dec_keys = [k for k in ms if k.startswith("3.")]
print(f"Decoder keys ({len(dec_keys)}):")
for k in dec_keys:
    print(f"  {k}: {ms[k].shape}")
