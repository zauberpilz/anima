"""
Anima Dashboard — Live-Monitoring für das CogLang v3 Training.
Start: python3 dashboard.py
Dann: http://localhost:8080 im Browser
"""
import os
import json
import time
import threading
from pathlib import Path
from flask import Flask, jsonify, send_file, request

# === Konfiguration ===
WSL_HOME = '/home/anima'
STATE_FILE = os.path.join(WSL_HOME, 'train_state.json')
CONFIG_FILE = os.path.join(WSL_HOME, 'evolution_config.json')
GENERATIONS_DIR = os.path.join(WSL_HOME, 'generations')
CONTROL_DIR = os.path.join(WSL_HOME, 'control')
CHECKPOINT_DIR = os.path.join(WSL_HOME, 'checkpoints')
LOG_DIR = os.path.join(WSL_HOME, 'logs')

app = Flask(__name__)

# Cache für letzte Messwerte
_status_cache = {}
_cache_time = 0

def read_json(path, default=None):
    """Sicheres Lesen einer JSON-Datei."""
    try:
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
    except Exception as e:
        pass
    return default or {}

def get_gpu_info():
    """Liest NVIDIA-SMI Daten aus (fallback-safe)."""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            return {
                'vram_used_mb': int(parts[0]),
                'vram_total_mb': int(parts[1]),
                'gpu_util_pct': int(parts[2]),
                'gpu_temp_c': int(parts[3]),
            }
    except:
        pass
    return {'vram_used_mb': 0, 'vram_total_mb': 8192, 'gpu_util_pct': 0, 'gpu_temp_c': 0}

def get_latest_generations():
    """Liest die letzten 2 Generations-Dateien."""
    try:
        files = sorted(Path(GENERATIONS_DIR).glob('evolve_gen_*.txt'), reverse=True)
        gens = []
        for f in files[:6]:  # Letzte 6 Samples
            try:
                text = f.read_text(encoding='utf-8')[:500]
                gens.append({'name': f.name, 'text': text, 'time': f.stat().st_mtime})
            except:
                pass
        return gens
    except:
        return []

def format_time(seconds):
    """Formatiert Sekunden in hh:mm:ss."""
    if seconds is None or seconds <= 0:
        return '--:--:--'
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'

# ===== API ENDPOINTS =====

@app.route('/')
def index():
    """Serve the dashboard HTML."""
    return send_file(os.path.join(os.path.dirname(__file__), 'dashboard.html'))

@app.route('/api/status')
def api_status():
    """JSON-Status für das Dashboard."""
    state = read_json(STATE_FILE)
    config = read_json(CONFIG_FILE)
    gpu = get_gpu_info()
    
    # Training aktiv?
    train_running = state.get('step', 0) > 0
    
    # Checkpoint Info
    ckpt_path = os.path.join(CHECKPOINT_DIR, 'checkpoint.pt')
    ckpt_age = None
    ckpt_size_mb = None
    if os.path.exists(ckpt_path):
        ckpt_age = time.time() - os.path.getmtime(ckpt_path)
        ckpt_size_mb = os.path.getsize(ckpt_path) / 1e6
    
    # Control Status
    paused = os.path.exists(os.path.join(CONTROL_DIR, 'pause'))
    stopped = os.path.exists(os.path.join(CONTROL_DIR, 'stop'))
    
    response = {
        'training': {
            'running': train_running,
            'paused': paused,
            'stopped': stopped,
            'step': state.get('step', 0),
            'total_steps': state.get('total_steps', 50000),
            'progress_pct': round(state.get('step', 0) / max(1, state.get('total_steps', 1)) * 100, 1),
            'loss': round(state.get('loss', 0), 4) if state.get('loss') else None,
            'best_loss': round(state.get('best_loss', 0), 4) if state.get('best_loss') else None,
            'lr': state.get('lr', 0),
            'speed': round(state.get('speed', 0), 1),
            'elapsed': format_time(state.get('elapsed_s')),
            'eta': format_time(state.get('eta_s')),
            'iteration': state.get('iteration', 0),
            'd_model': state.get('d_model', 0),
            'n_layers': state.get('n_layers', 0),
            'params_m': round(state.get('params_m', 0), 1),
            'batch_size': state.get('batch_size', 0),
            'seq_len': state.get('seq_len', 0),
            'vram_mb': state.get('vram_mb', 0),
            'timestamp': state.get('timestamp', 0),
            'loss_history': state.get('loss_history', []),
        },
        'gpu': gpu,
        'config': {
            'd_model': config.get('d_model', '?'),
            'd_sparse': config.get('d_sparse', '?'),
            'n_layers': config.get('n_layers', '?'),
            'lr': config.get('lr', '?'),
            'max_vram_mb': config.get('max_vram_mb', '?'),
            'iteration': config.get('iteration', '?'),
            'best_loss': round(config.get('best_loss', 0), 4) if config.get('best_loss') else '?',
        },
        'checkpoint': {
            'age_min': round(ckpt_age / 60, 1) if ckpt_age else None,
            'size_mb': round(ckpt_size_mb, 1) if ckpt_size_mb else None,
            'path': str(ckpt_path),
        },
        'generations': get_latest_generations(),
        'server_time': time.time(),
    }
    return jsonify(response)

@app.route('/api/control', methods=['POST'])
def api_control():
    """Steuerung: pause / resume / stop."""
    data = request.get_json()
    action = data.get('action', '')
    
    if action == 'pause':
        open(os.path.join(CONTROL_DIR, 'pause'), 'w').close()
        return jsonify({'status': 'paused', 'message': 'Training pausiert'})
    elif action == 'resume':
        for f in ['pause', 'stop']:
            p = os.path.join(CONTROL_DIR, f)
            if os.path.exists(p):
                os.remove(p)
        return jsonify({'status': 'running', 'message': 'Training fortgesetzt'})
    elif action == 'stop':
        open(os.path.join(CONTROL_DIR, 'stop'), 'w').close()
        return jsonify({'status': 'stopped', 'message': 'Training gestoppt'})
    else:
        return jsonify({'error': f'Unknown action: {action}'}), 400

@app.route('/api/checkpoint-info')
def api_checkpoint_info():
    """Informationen über den aktuellen Checkpoint."""
    try:
        import torch
        ckpt = torch.load(os.path.join(CHECKPOINT_DIR, 'checkpoint.pt'), map_location='cpu')
        ckpt_config = ckpt.get('config', {})
        has_nan = any(torch.isnan(p).any().item() for p in ckpt['model_state'].values())
        total_params = sum(p.numel() for p in ckpt['model_state'].values())
        return jsonify({
            'params_m': round(total_params / 1e6, 1),
            'has_nan': has_nan,
            'config': ckpt_config,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print('=' * 60)
    print('  ANIMA DASHBOARD — Live Training Monitor')
    print('=' * 60)
    print(f'  State:  {STATE_FILE}')
    print(f'  Config: {CONFIG_FILE}')
    print()
    print('  Öffne http://localhost:8080 im Browser')
    print('  Drücke STRG+C zum Beenden')
    print('=' * 60)
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)
