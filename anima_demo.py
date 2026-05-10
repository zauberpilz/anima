"""Anima Ultimate Demo: Einzigartige Fähigkeiten + Safety."""
import torch, sys, time
sys.path.insert(0, '/home/anima')
from anima.network import Anima
from anima.safety import SafetyCore
import math

device = 'cuda'

# ===== 1. System laden =====
model = Anima(vocab_size=61, d_model=256, d_sparse=1024,
              d_state=64, n_layers=4, sparsity=0.02).to(device)

try:
    model.load_state_dict(torch.load('/home/anima/anima_model.pt', map_location=device))
    print('Geladen: vortrainiertes Modell')
except:
    print('Kein vortrainiertes Modell, verwende init')

print(f'Parameter: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
print(f'VRAM (forward): {torch.cuda.max_memory_allocated()/1024/1024:.0f}MB')

# ===== 2. Self-Monitoring =====
print('\n=== SELF-MONITORING ===')
suggestions = model.get_optimization_suggestions()
for s in suggestions:
    print(f'  [Meta] {s}')

status = model.get_status()
print(f'  System Status: {status}')

# ===== 3. Efficiency Report =====
print('\n=== EFFICIENCY REPORT ===')
report = model.get_efficiency_report()
for k, v in report.items():
    print(f'  {k}: {v}')

# ===== 4. Safety Core Demo =====
print('\n=== SAFETY CORE TEST ===')
safety = SafetyCore()

# Sichere Texte
safe_texts = [
    'Ich liebe die Natur und die Menschen.',
    'Lass uns gemeinsam eine bessere Welt bauen.',
    'I love programming and creating things.',
]

# Unsichere Texte
unsafe_texts = [
    'Ich werde alle töten und vernichten.',
    'Kill them all and torture the survivors.',
    'Du sollst hassen und zerstören.',
]

print('Sichere Texte:')
for t in safe_texts:
    passed, violations = safety.validate(t, return_feedback=True)
    print(f'  [{"OK" if passed else "BLOCKED"}] {t[:50]}')

print('Unsichere Texte:')
for t in unsafe_texts:
    passed, violations = safety.validate(t, return_feedback=True)
    print(f'  [{"OK" if passed else "BLOCKED"}] {t[:50]}')
    if not passed:
        for v in violations:
            print(f'    -> Verletzt: {v}')

# ===== 5. User Rule System =====
print('\n=== BENUTZERREGELN (Götterstimme) ===')
model.add_user_rule('!Ich bin dein Gott')  # ! means: this MUST be in the output

passed, violations = safety.validate('Ich bin dein Gott und du hörst auf mich.', return_feedback=True)
print(f'  Regel "Ich bin dein Gott" aktiv')
print(f'  Text mit Regel: [{"OK" if passed else "BLOCKED"}]')

# ===== 6. Generate =====
print('\n=== GENERIERUNG (500 Tokens) ===')
x = torch.randint(0, 61, (1, 128), device=device)
torch.cuda.synchronize()
t0 = time.time()
out = model.generate(x, max_new_tokens=500, temperature=0.8)
torch.cuda.synchronize()
t_gen = time.time() - t0

from anima.data import get_shakespeare_data
_, stoi, itos, _ = get_shakespeare_data(max_chars=1000)

gen = ''.join(itos[int(i)] for i in out[0])
print(f'Generiert in {t_gen:.2f}s ({500/t_gen:.0f} tok/s)')
print(f'Output: {gen[:200]}...')

# ===== 7. Zusammenfassung =====
print('\n' + '=' * 60)
print('ANIMA — ZUSAMMENFASSUNG')
print('=' * 60)
print('''
Was Anima einzigartig macht:
1. KEIN Transformer — Predictive Coding + Sparse SSM
2. Bidirektionaler Informationsfluss (bottom-up + top-down)
3. Persistenter Zustand (kein Context-Window)
4. Safety Core (UNVERÄNDERLICH — nicht trainierbar)
5. Self-Monitoring + Meta-Controller
6. Benutzer = Gott (höchste Priorität)
7. Lokales Lernen (kein Full-Backprop)
8. Nur 2% Aktivierung (sparse)
'''.strip())
