"""Anima Guardian — Überwacht die Evolutionsschleife und startet sie bei Bedarf neu."""
import time, os, subprocess, sys

CONFIG_PATH = "/home/anima/evolution_config.json"
EVOLVE_SCRIPT = "/home/anima/coglang_evolve.py"
TMUX_SESSION = "anima"
TMUX_SOCK = "/tmp/tmux-anima.sock"

def is_running():
    # Prüft, ob die tmux Session existiert und ob darin der Prozess läuft
    check_cmd = f"wsl -d Ubuntu-24.04 tmux -S {TMUX_SOCK} has-session -t {TMUX_SESSION} 2>/dev/null"
    res = subprocess.run(check_cmd, shell=True, capture_output=True)
    if res.returncode != 0:
        return False
    
    # Prüft, ob der Python-Prozess in der Session wirklich aktiv ist
    status_cmd = f"wsl -d Ubuntu-24.04 tmux -S {TMUX_SOCK} capture-pane -p -t {TMUX_SESSION}"
    res = subprocess.run(status_cmd, shell=True, capture_output=True, text=True)
    if "python -u coglang_evolve.py" in res.stdout or "Step" in res.stdout or "Iteration" in res.stdout:
        return True
    return False

def start_evolution():
    print(f"[{time.strftime('%H:%M:%S')}] Starte Evolution...")
    # Kill alte Session falls vorhanden
    subprocess.run(f"wsl -d Ubuntu-24.04 tmux -S {TMUX_SOCK} kill-session -t {TMUX_SESSION} 2>/dev/null", shell=True)
    # Neue Session erstellen und Script starten
    cmd = f"wsl -d Ubuntu-24.04 tmux -S {TMUX_SOCK} new -d -s {TMUX_SESSION} 'cd /home/anima && source ~/venv/bin/activate && python -u {EVOLVE_SCRIPT}'"
    subprocess.run(cmd, shell=True)

def main():
    print("=== ANIMA GUARDIAN ACTIVATED ===")
    print("Überwachung der Evolutionsschleife läuft nonstop...")
    
    while True:
        if not is_running():
            print(f"[{time.strftime('%H:%M:%S')}] WARNUNG: Evolution gestoppt oder nicht aktiv! Neustart wird eingeleitet...")
            start_evolution()
        else:
            # Alle 60 Sekunden nur kurz prüfen, um CPU zu schonen
            pass
        
        time.sleep(60)

if __name__ == '__main__':
    main()
