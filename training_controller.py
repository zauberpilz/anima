"""
Training Controller — Pause, Resume, Stop für das Training.
Kommuniziert über Signal-Dateien mit dem Evolution Loop.
"""
import os
import signal
import sys
import time

CONTROL_DIR = '/home/anima/control'
STATE_FILE = os.path.join(CONTROL_DIR, 'state.txt')
PAUSE_FILE = os.path.join(CONTROL_DIR, 'pause')
STOP_FILE = os.path.join(CONTROL_DIR, 'stop')

class TrainingController:
    """
    Ermöglicht Pause/Resume/Stop während des Trainings.
    Der Evolution Loop prüft regelmäßig den Status.
    """
    def __init__(self):
        os.makedirs(CONTROL_DIR, exist_ok=True)
        self.state = 'running'  # running, paused, stopping
        self._write_state('running')
        
    def _write_state(self, state):
        with open(STATE_FILE, 'w') as f:
            f.write(state)
        self.state = state
        
    def check_pause(self):
        """Prüft ob pausiert werden soll. Blockiert bis Resume."""
        if os.path.exists(PAUSE_FILE):
            print("\n[CONTROLLER] ⏸️ TRAINING PAUSIERT")
            print("  Resume: Lösche /home/anima/control/pause")
            print("  Stop:   Lösche /home/anima/control/pause und erstelle /home/anima/control/stop")
            self._write_state('paused')
            while os.path.exists(PAUSE_FILE):
                time.sleep(1)
            self._write_state('running')
            print("[CONTROLLER] ▶️ TRAINING FORTGESETZT")
            return True
        return False
        
    def check_stop(self):
        """Prüft ob gestoppt werden soll."""
        if os.path.exists(STOP_FILE):
            print("\n[CONTROLLER] 🛑 TRAINING STOPP")
            self._write_state('stopped')
            return True
        return False
        
    def pause(self):
        """Pause das Training."""
        open(PAUSE_FILE, 'w').close()
        print("[CONTROLLER] Pause-Signal gesendet")
        
    def resume(self):
        """Resume das Training."""
        if os.path.exists(PAUSE_FILE):
            os.remove(PAUSE_FILE)
        if os.path.exists(STOP_FILE):
            os.remove(STOP_FILE)
        print("[CONTROLLER] Resume-Signal gesendet")
        
    def stop(self):
        """Stoppe das Training."""
        open(STOP_FILE, 'w').close()
        if os.path.exists(PAUSE_FILE):
            os.remove(PAUSE_FILE)
        print("[CONTROLLER] Stop-Signal gesendet")
        
    def get_status(self):
        """Gibt aktuellen Status zurück."""
        if os.path.exists(STOP_FILE):
            return 'stopped'
        if os.path.exists(PAUSE_FILE):
            return 'paused'
        return 'running'


# CLI Interface
if __name__ == '__main__':
    controller = TrainingController()
    
    if len(sys.argv) < 2:
        print("Usage: python3 training_controller.py [pause|resume|stop|status]")
        sys.exit(1)
        
    command = sys.argv[1].lower()
    
    if command == 'pause':
        controller.pause()
    elif command == 'resume':
        controller.resume()
    elif command == 'stop':
        controller.stop()
    elif command == 'status':
        print(f"Status: {controller.get_status()}")
    else:
        print(f"Unbekannter Befehl: {command}")
