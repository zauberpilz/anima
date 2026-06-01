#!/usr/bin/env python3
"""Check training progress from evolve.log."""
import re

with open("/home/anima/evolve.log", "r", errors="replace") as f:
    text = f.read()

# Find all progress lines  
lines = re.findall(r"\[\s*\d+\.\d%\] Step\s+\d+.*?Phase=\w+", text)
for line in lines[-5:]:
    print(line)

print()
if lines:
    last = lines[-1]
    step_m = re.search(r"Step\s+(\d+)", last)
    loss_m = re.search(r"loss=([\d.]+)", last)
    pct_m = re.search(r"\[(\s*\d+\.\d%)", last)
    lr_m = re.search(r"LR=([\d.]+)", last)
    if step_m and loss_m:
        pct = pct_m.group(1) if pct_m else "?"
        print(f"Step: {step_m.group(1)} / 50000 ({pct})  Loss: {loss_m.group(1)}  LR: {lr_m.group(1) if lr_m else '?'}")

    # Compute average of last 20 losses
    losses = [float(re.search(r"loss=([\d.]+)", l).group(1)) for l in lines[-20:] if re.search(r"loss=([\d.]+)", l)]
    if losses:
        print(f"Avg loss (last 20): {sum(losses)/len(losses):.4f}  Min: {min(losses):.4f}  Max: {max(losses):.4f}")
