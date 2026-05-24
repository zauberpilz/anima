"""Evaluator Daemon — generiert Samples während Training läuft (CPU/GPU hybrid)."""
import subprocess, time, json, torch, torch.nn.functional as F, sys, os
sys.path.insert(0, "/home/anima")
from coglang import build_anima
from anima.data import get_shakespeare_data

stoi, itos = get_shakespeare_data(max_chars=1000)[1], get_shakespeare_data(max_chars=1000)[2]
device = "cuda"
LAST_STEP_EVAL = 0

def get_current_step():
    try:
        out = subprocess.run(
            ["tmux", "-S", "/tmp/tmux-anima.sock", "capture-pane", "-p", "-t", "anima", "-S", "-3"],
            capture_output=True, text=True, timeout=5
        )
        for line in out.stdout.split("\n"):
            if "| loss=" not in line or "Step" not in line: continue
            try: return int(line.split("Step")[1].split()[0])
            except: pass
    except: pass
    return 0

def load_brain():
    brain = build_anima(62, device)
    state_path = "/home/anima/brain_state.pt"
    if os.path.exists(state_path):
        brain.modules.load_state_dict(torch.load(state_path))
    return brain

def generate(brain, step):
    torch.manual_seed(step)
    for prompt in ["ROMEO:", "KING "]:
        ctx = torch.tensor([[stoi.get(c, 0) for c in prompt]], device=device)
        for _ in range(150):
            out, _ = brain.forward(ctx[:, -128:], learn=False)
            probs = F.softmax(out[:, -1, :] / 0.8, dim=-1)
            ctx = torch.cat([ctx, torch.multinomial(probs, 1)], dim=-1)
        gen = "".join(itos.get(int(i), "?") for i in ctx[0])
        path = f"/home/anima/gen_step_{step}.txt"
        with open(path, "w") as f:
            f.write(f"Step {step} | Prompt: {prompt}\n{gen}")
        print(f"[EVAL] Step {step}: {prompt} -> {gen[:50]}...")

print("[EVALUATOR] Gestartet — evaluiert alle 10000 Steps")
while True:
    step = get_current_step()
    if step > LAST_STEP_EVAL + 10000 and step >= 5000:
        LAST_STEP_EVAL = step
        try:
            brain = load_brain()
            generate(brain, step)
        except Exception as e:
            print(f"[EVAL] Fehler: {e}")
    time.sleep(60)
