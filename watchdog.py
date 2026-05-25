#!/usr/bin/env python3
"""
Watchdog für CogLang v3 ANIMA Training.
Überwacht evolve.py, startet bei Crash automatisch neu.
Läuft auf Windows-Seite und prüft WSL-Prozess periodisch.
"""
import subprocess
import time
import os
import json
import sys
from datetime import datetime, timedelta

WSL_DISTRO = "Ubuntu-24.04"
EVOLVE_CMD = [
    "wsl", "-d", WSL_DISTRO, "-e", "bash", "-l", "-c",
    "cd /home/anima/src && nice -19 /home/anima/venv/bin/python3 -u coglang_evolve.py"
]
CHECKPOINT_DIR = "/home/anima/checkpoints"
EVOLVE_LOG = "/home/anima/evolve.log"
CONTROL_DIR = "/home/anima/control"
CONFIG_PATH = "/home/anima/evolution_config.json"
RECOVERY_DIR = "/home/anima/recovery"

POLL_INTERVAL = 60  # Sekunden zwischen Checks
MAX_RESTARTS = 10   # Max Neustarts bevor wir aufgeben
RESTART_WINDOW = 3600  # 1h Fenster für max_restarts

class WSLWatchdog:
    def __init__(self):
        self.restart_count = 0
        self.restart_times = []
        self.last_loss = None
        self.start_time = datetime.now()
    
    def wsl_run(self, cmd):
        """Run command in WSL and return output."""
        try:
            full_cmd = ["wsl", "-d", WSL_DISTRO, "-e", "bash", "-l", "-c", cmd]
            result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=30)
            return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            return "", "TIMEOUT", -1
        except Exception as e:
            return "", str(e), -1
    
    def is_evolve_running(self):
        """Check if evolve.py process exists in WSL."""
        out, _, _ = self.wsl_run("ps aux | grep 'coglang_evolve' | grep -v grep | wc -l")
        try:
            count = int(out.strip())
            return count > 0
        except:
            return False
    
    def get_latest_loss(self):
        """Extract latest loss from evolve log."""
        out, _, _ = self.wsl_run(f"tail -3 {EVOLVE_LOG} 2>/dev/null | grep -oP 'loss=\\K[\\d.]+' | tail -1")
        try:
            loss = float(out.strip())
            return loss
        except:
            return None
    
    def get_training_step(self):
        """Extract latest step from evolve log."""
        out, _, _ = self.wsl_run(f"tail -3 {EVOLVE_LOG} 2>/dev/null | grep -oP 'Step\\s+\\K\\d+' | tail -1")
        try:
            return int(out.strip())
        except:
            return 0
    
    def get_config(self):
        """Read evolution config."""
        out, _, _ = self.wsl_run(f"cat {CONFIG_PATH} 2>/dev/null")
        try:
            return json.loads(out)
        except:
            return None
    
    def verify_checkpoint(self):
        """Check if main checkpoint is valid (no NaN)."""
        out, _, _ = self.wsl_run("""python3 -c "
import torch, sys
try:
    ckpt = torch.load('/home/anima/checkpoints/checkpoint.pt', map_location='cpu')
    total = sum(v.numel() for v in ckpt['model_state'].values())
    nan = sum(torch.isnan(v).sum().item() for v in ckpt['model_state'].values())
    print(f'{nan}/{total}')
except: print('ERROR')
" 2>/dev/null""")
        return out.strip()
    
    def fix_checkpoint_nan(self):
        """Fix NaN weights in checkpoint."""
        out, _, _ = self.wsl_run("""python3 -c "
import torch, os
ckpt = torch.load('/home/anima/checkpoints/checkpoint.pt', map_location='cpu')
ms = ckpt['model_state']
fixed = 0
for k, v in ms.items():
    nan_mask = torch.isnan(v)
    if nan_mask.any():
        rand_vals = torch.randn(v.shape, dtype=v.dtype) * 0.01
        v.data = torch.where(nan_mask, rand_vals, v)
        v.data = v.data.clamp_(-1.0, 1.0)
        fixed += 1
if fixed > 0:
    torch.save(ckpt, '/home/anima/checkpoints/checkpoint.pt')
    print(f'Fixed {fixed} NaN params')
else:
    print('No NaN found')
" 2>/dev/null""")
        return out.strip()
    
    def start_evolve(self):
        """Start evolve.py in background."""
        # Create control dir if needed
        self.wsl_run(f"mkdir -p {CONTROL_DIR} && echo 'resume' > {CONTROL_DIR}/signal.txt")
        
        # Start evolve
        cmd = (
            f"cd /home/anima/src && nohup nice -19 /home/anima/venv/bin/python3 -u "
            f"coglang_evolve.py > {EVOLVE_LOG} 2>&1 &"
        )
        out, err, rc = self.wsl_run(cmd)
        time.sleep(3)
        
        if self.is_evolve_running():
            return True
        else:
            # Try differently - direct bash
            full_cmd = [
                "wsl", "-d", WSL_DISTRO,
                "bash", "-l", "-c",
                f"cd /home/anima/src && nohup nice -19 /home/anima/venv/bin/python3 -u "
                f"coglang_evolve.py > {EVolve_LOG} 2>&1 &"
            ]
            subprocess.Popen(full_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(5)
            return self.is_evolve_running()
    
    def ensure_dashboard_running(self):
        """Start dashboard if not running."""
        out, _, _ = self.wsl_run("ps aux | grep dashboard | grep -v grep | wc -l")
        try:
            if int(out.strip()) == 0:
                self.wsl_run(
                    "cd /home/anima/src && nohup /home/anima/venv/bin/python3 -u "
                    "dashboard.py > /dev/null 2>&1 &"
                )
                return "Dashboard gestartet"
        except:
            pass
        return "Dashboard läuft"
    
    def print_status(self):
        """Print current status."""
        now = datetime.now()
        uptime = now - self.start_time
        
        running = self.is_evolve_running()
        loss = self.get_latest_loss()
        step = self.get_training_step()
        config = self.get_config()
        
        iter_num = config.get('iteration', '?') if config else '?'
        
        status = (
            f"\n{'='*50}\n"
            f"WATCHDOG STATUS ({now.strftime('%H:%M:%S')})\n"
            f"{'='*50}\n"
            f"Uptime: {uptime}\n"
            f"Evolve läuft: {'JA' if running else 'NEIN'}\n"
            f"Iteration: {iter_num}\n"
            f"Step: {step}/50000\n"
            f"Loss: {loss if loss else 'N/A'}\n"
            f"Neustarts: {self.restart_count}\n"
            f"{'='*50}"
        )
        print(status)
        return status
    
    def run_forever(self):
        """Main watchdog loop."""
        print(f"{'='*50}")
        print(f"WATCHDOG GESTARTET um {self.start_time.strftime('%H:%M:%S')}")
        print(f"{'='*50}")
        print(f"Prüfe alle {POLL_INTERVAL}s auf evolve.py...")
        print(f"Max {MAX_RESTARTS} Neustarts in {RESTART_WINDOW}s Fenster")
        print(f"{'='*50}\n")
        
        while True:
            try:
                running = self.is_evolve_running()
                
                if not running:
                    now = time.time()
                    # Clean old restarts from window
                    self.restart_times = [t for t in self.restart_times if now - t < RESTART_WINDOW]
                    
                    if len(self.restart_times) >= MAX_RESTARTS:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                              f"KRITISCH: {MAX_RESTARTS} Neustarts in {RESTART_WINDOW}s! "
                              f"Warte auf manuellen Eingriff.")
                        time.sleep(300)  # Wait 5min before checking again
                        continue
                    
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f" evolve.py NICHT GEFUNDEN! Starte neu...")
                    
                    # Verify checkpoint health first
                    ckpt_status = self.verify_checkpoint()
                    if 'ERROR' in ckpt_status:
                        print(f"  Checkpoint-Fehler! Überspringe...")
                    elif ckpt_status != 'No NaN found':
                        nan_part = ckpt_status.split('/')[0]
                        print(f"  Checkpoint hat {nan_part} NaN! Repariere...")
                        self.fix_checkpoint_nan()
                    
                    # Start evolve
                    self.ensure_dashboard_running()
                    success = self.start_evolve()
                    
                    if success:
                        self.restart_count += 1
                        self.restart_times.append(now)
                        print(f"  ✅ evolve.py gestartet (Neustart #{self.restart_count})")
                    else:
                        print(f"  ❌ evolve.py START FEHLGESCHLAGEN!")
                        # Try direct subprocess approach
                        try:
                            subprocess.Popen(
                                ["wsl", "-d", WSL_DISTRO, "bash", "-l", "-c",
                                 f"cd /home/anima/src && nohup nice -19 /home/anima/venv/bin/python3 "
                                 f"-u coglang_evolve.py > /home/anima/evolve.log 2>&1 &"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                            )
                            time.sleep(5)
                            if self.is_evolve_running():
                                print(f"  ✅ evolve.py via Popen gestartet")
                                self.restart_count += 1
                                self.restart_times.append(now)
                        except Exception as e:
                            print(f"  ❌ Auch Popen fehlgeschlagen: {e}")
                else:
                    # Running - just check progress
                    loss = self.get_latest_loss()
                    step = self.get_training_step()
                    if loss and (self.last_loss is None or loss != self.last_loss):
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                              f"Step {step:>5d} | loss={loss:.4f}" + 
                              (f" (Δ={self.last_loss-loss:+.4f})" if self.last_loss else ""))
                        self.last_loss = loss
                    
                    # Check for loss stagnation (no improvement in 1000 steps)
                    # (would need to track loss history - skip for now)
                
                time.sleep(POLL_INTERVAL)
                
            except KeyboardInterrupt:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Watchdog beendet (Ctrl+C)")
                break
            except Exception as e:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Fehler: {e}")
                time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    wd = WSLWatchdog()
    wd.run_forever()
