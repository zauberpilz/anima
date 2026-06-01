import json
d = json.load(open('/home/anima/train_state.json'))
print(f"Step: {d['step']}, Loss: {d['multi_task_ce']:.2f}, Phase: {d['current_phase']}, VRAM: {d.get('vram_mb','?')}MB")
