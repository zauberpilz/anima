"""Quick training status check via dashboard API."""
import json, subprocess
r = subprocess.run(['curl', '-s', 'http://localhost:8080/api/status'], capture_output=True, text=True)
d = json.loads(r.stdout)
t = d['training']
c = d['config']
g = d['gpu']
print("=" * 50)
print(f"  Step:    {t['step']} / {t['total_steps']} ({t['progress_pct']:.1f}%)")
print(f"  Loss:    {t['loss']}")
print(f"  LR:      {t['lr']}")
print(f"  Speed:   {t['speed']} step/s")
print(f"  ETA:     {t['eta']}")
print(f"  VRAM:    {g['vram_used_mb']}MB / {g['vram_total_mb']}MB")
print(f"  GPU:     {g['gpu_util_pct']}% @ {g['gpu_temp_c']}C")
print(f"  Iter:    {t['iteration']}")
print(f"  Params:  {t['params_m']}M")
print(f"  Running: {t['running']}")
print("=" * 50)
