"""Cobra Guardian — Live-Monitor für Training.

Läuft parallel zum Training und überwacht:
- NaN/Schwellwerte
- VRAM-Grenzen
- Loss-Stagnation
- Regelverstöße
"""
import time, json, os

LOG_FILE = '/home/anima/training_monitor.json'
ALERT_FILE = '/home/anima/guardian_alerts.txt'
TRAIN_LOG = '/home/anima/train_output_v2.log'

RULES = {
    'max_loss': 15.0,
    'max_vram_mb': 7000,
    'stagnation_steps': 10000,
    'stagnation_threshold': 0.01,
    'loss_spike_threshold': 2.0,
}

class Guardian:
    def __init__(self):
        self.history = []
        self.alerts = []
        self.last_best = float('inf')
        self.stagnant_steps = 0
        self.violations = 0

    def check(self):
        """Prüfe neueste Log-Einträge auf Regelverstöße."""
        try:
            with open(TRAIN_LOG, 'r') as f:
                lines = f.readlines()
        except:
            return

        for line in lines:
            if '| loss=' not in line:
                continue
            try:
                parts = line.split('|')
                step_str = parts[0].split('Step')[1].strip()
                step = int(step_str)
                loss_str = parts[1].split('=')[1].split()[0]
                loss = float(loss_str)
                vram_part = parts[2].split('=')[1].split('MB')[0].strip()
                vram = float(vram_part)

                if step in [h['step'] for h in self.history]:
                    continue

                entry = {'step': step, 'loss': loss, 'vram': vram, 'time': time.time()}
                self.history.append(entry)

                # Regel 1: NaN
                if loss != loss:
                    self._alert('NAN', f'Step {step}: loss=NaN!')

                # Regel 2: Max Loss
                if loss > RULES['max_loss']:
                    self._alert('MAX_LOSS', f'Step {step}: loss={loss:.2f} > {RULES["max_loss"]}')

                # Regel 3: VRAM Obergrenze
                if vram > RULES['max_vram_mb']:
                    self._alert('VRAM', f'Step {step}: VRAM={vram:.0f}MB > {RULES["max_vram_mb"]}MB')

                # Regel 4: Loss-Spike
                if not hasattr(self, 'last_loss'):
                    self.last_loss = loss
                elif loss > self.last_loss * 1.5 and loss > 5.0:
                    self._alert('SPIKE', f'Step {step}: loss={loss:.4f} (vorher {self.last_loss:.4f})')
                self.last_loss = loss

                # Regel 5: Stagnation
                if loss < self.last_best - 0.001:
                    self.last_best = loss
                    self.stagnant_steps = 0
                else:
                    self.stagnant_steps += 1

                if self.stagnant_steps > RULES['stagnation_steps']:
                    self._alert('STAGNATION', f'Step {step}: Keine Verbesserung seit {self.stagnant_steps} Steps (best={self.last_best:.4f})')

                # Regel 6: Langsamer als 5 step/s
                if 'speed' in parts[3]:
                    speed_str = parts[3].split('=')[1].split('step')[0].strip()
                    speed = float(speed_str)
                    if speed < 5.0 and step > 100:
                        self._alert('SLOW', f'Step {step}: Nur {speed:.1f} step/s')

            except Exception as e:
                pass

        self._save()

    def _alert(self, rule, msg):
        entry = f'[{time.strftime("%H:%M:%S")}] RULE {rule}: {msg}'
        self.alerts.append(entry)
        self.violations += 1
        with open(ALERT_FILE, 'a') as f:
            f.write(entry + '\n')
        print(entry)

    def _save(self):
        data = {
            'last_check': time.time(),
            'total_violations': self.violations,
            'recent_alerts': self.alerts[-10:],
            'history': self.history[-100:],
            'best_loss': self.last_best,
            'stagnant_steps': self.stagnant_steps,
        }
        with open(LOG_FILE, 'w') as f:
            json.dump(data, f, indent=2)

    def summary(self):
        if not self.history:
            return 'Keine Daten'
        start = self.history[0]['loss']
        current = self.history[-1]['loss']
        steps = self.history[-1]['step']
        return (f'Guardian: {steps} Steps, '
                f'{start:.2f}→{current:.2f}, '
                f'Best={self.last_best:.4f}, '
                f'Alerts={self.violations}, '
                f'Stagnation={self.stagnant_steps} Steps')


if __name__ == '__main__':
    # HTOP-Style Dashboard
    g = Guardian()
    print('='*60)
    print('COBRA GUARDIAN — Regel-Monitor')
    print('='*60)
    print(f'Max Loss: {RULES["max_loss"]} | Max VRAM: {RULES["max_vram_mb"]}MB')
    print(f'Spike Threshold: {RULES["loss_spike_threshold"]}x | Stagnation: {RULES["stagnation_steps"]} Steps')
    print('='*60)

    # Once-off check (for CLI)
    g.check()
    print(g.summary())
    if g.alerts:
        print('\nALERTS:')
        for a in g.alerts[-5:]:
            print(f'  {a}')
