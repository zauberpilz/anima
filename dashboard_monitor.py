"""
CogLang Dashboard Monitor — Live Training Metrics.
Liest Trainingsstatus aus tmux und Config, served via Flask.
"""
import subprocess
import json
import re
import os
from flask import Flask, jsonify, render_template_string

app = Flask(__name__)

TMUX_SOCK = '/tmp/tmux-anima.sock'
CONFIG_PATH = '/home/anima/evolution_config.json'
CHECKPOINT_DIR = '/home/anima/checkpoints'
HOME = '/home/anima'

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="5">
    <title>CogLang AGI Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Consolas', 'Courier New', monospace; background: #0a0a0f; color: #c0c0c0; padding: 20px; }
        h1 { color: #00ff88; font-size: 1.3em; margin-bottom: 15px; }
        h2 { color: #00ccff; font-size: 1.1em; margin: 15px 0 8px; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 8px; margin-bottom: 15px; }
        .metric { background: #12121a; border: 1px solid #2a2a3a; border-radius: 6px; padding: 10px 14px; }
        .metric .label { font-size: 0.7em; color: #888; text-transform: uppercase; letter-spacing: 1px; }
        .metric .value { font-size: 1.2em; font-weight: bold; margin-top: 2px; }
        .green { color: #00ff88; }
        .yellow { color: #ffcc00; }
        .red { color: #ff4444; }
        .blue { color: #00ccff; }
        .dim  { color: #666; }
        .log { background: #0d0d14; border: 1px solid #1a1a2a; border-radius: 6px; padding: 12px; font-size: 0.75em; line-height: 1.5; max-height: 400px; overflow-y: auto; white-space: pre-wrap; word-break: break-all; }
        .bar-bg { background: #1a1a2a; border-radius: 4px; height: 18px; margin-top: 4px; overflow: hidden; }
        .bar-fill { height: 100%; border-radius: 4px; transition: width 1s; }
        .bar-fill.green { background: linear-gradient(90deg, #00cc66, #00ff88); }
        .bar-fill.yellow { background: linear-gradient(90deg, #cc8800, #ffcc00); }
        .footer { margin-top: 20px; color: #444; font-size: 0.7em; text-align: center; }
        .domain-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-bottom: 15px; }
        .domain-card { background: #12121a; border: 1px solid #2a2a3a; border-radius: 6px; padding: 10px; text-align: center; }
        .domain-card .name { font-size: 0.75em; color: #888; }
        .domain-card .weight { font-size: 0.9em; margin-top: 2px; }
        .generation { background: #0d0d14; border: 1px solid #2a2a3a; border-radius: 6px; padding: 10px; margin-bottom: 6px; font-size: 0.78em; white-space: pre-wrap; word-break: break-all; }
    </style>
</head>
<body>
    <h1>⚡ CogLang v3 AGI — Live Monitor</h1>
    <div class="metrics">
        <div class="metric"><div class="label">Iteration</div><div class="value green">{{ m.iteration }}</div></div>
        <div class="metric"><div class="label">Step</div><div class="value">{{ m.step }} / {{ m.total_steps }}</div></div>
        <div class="metric"><div class="label">Loss</div><div class="value {% if m.loss_ok %}green{% else %}yellow{% endif %}">{{ m.loss }}</div></div>
        <div class="metric"><div class="label">Learning Rate</div><div class="value blue">{{ m.lr }}</div></div>
        <div class="metric"><div class="label">VRAM</div><div class="value">{{ m.vram }} MB</div></div>
        <div class="metric"><div class="label">Speed</div><div class="value">{{ m.speed }}</div></div>
        <div class="metric"><div class="label">ETA</div><div class="value {% if m.eta_ok %}green{% else %}yellow{% endif %}">{{ m.eta }}</div></div>
        <div class="metric"><div class="label">Phase</div><div class="value blue">{{ m.phase }}</div></div>
        <div class="metric"><div class="label">Best Loss</div><div class="value green">{{ m.best_loss }}</div></div>
        <div class="metric"><div class="label">Parameters</div><div class="value">{{ m.params }}</div></div>
    </div>
    <div style="margin-bottom:15px">
        <div class="label" style="font-size:0.7em;color:#888;text-transform:uppercase;letter-spacing:1px">Progress</div>
        <div class="bar-bg"><div class="bar-fill green" style="width:{{ m.progress_pct }}%"></div></div>
    </div>
    <h2>Domain Weights (current phase: {{ m.phase }})</h2>
    <div class="domain-grid">
        {% for d, w in m.domain_weights.items() %}
        <div class="domain-card"><div class="name">{{ d }}</div><div class="weight">{{ "%.0f"|format(w*100) }}%</div></div>
        {% endfor %}
    </div>
    <h2>Model Architecture</h2>
    <div class="metrics">
        <div class="metric"><div class="label">d_model</div><div class="value blue">{{ m.d_model }}</div></div>
        <div class="metric"><div class="label">d_sparse</div><div class="value blue">{{ m.d_sparse }}</div></div>
        <div class="metric"><div class="label">Layers</div><div class="value blue">{{ m.n_layers }}</div></div>
        <div class="metric"><div class="label">d_state</div><div class="value blue">{{ m.d_state }}</div></div>
        <div class="metric"><div class="label">Vocab</div><div class="value blue">{{ m.vocab_size }}</div></div>
    </div>
    <h2>Latest Generation Samples</h2>
    {% if m.generations %}
        {% for gen in m.generations %}
        <div class="generation">{{ gen }}</div>
        {% endfor %}
    {% else %}
        <div class="dim">No samples yet</div>
    {% endif %}
    <h2>Training Log (recent)</h2>
    <div class="log">{{ m.log }}</div>
    <div class="footer">Updated: {{ m.timestamp }} | Auto-refresh every 5s | WSL/anima</div>
</body>
</html>
"""


def run_cmd(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return r.stdout
    except:
        return ''


def get_tmux_log():
    """Letzte ~50 Zeilen aus tmux holen."""
    out = run_cmd(['tmux', '-S', TMUX_SOCK, 'capture-pane', '-t', 'anima', '-p', '-S', '-60'])
    return out


def parse_metrics(log_text):
    """Extrahiert Metriken aus tmux-Log."""
    m = {
        'iteration': '?', 'step': '?', 'total_steps': '?',
        'loss': '?', 'loss_ok': True, 'lr': '?', 'vram': '?',
        'speed': '?', 'eta': '?', 'eta_ok': True, 'phase': '?',
        'best_loss': '?', 'params': '?', 'progress_pct': 0,
        'd_model': '?', 'd_sparse': '?', 'n_layers': '?',
        'd_state': '?', 'vocab_size': '?', 'domain_weights': {},
        'generations': [], 'log': log_text[-2000:],
        'timestamp': '', 'domain_weights_str': ''
    }

    # Step/Loss/VRAM/Speed/ETA/Phase
    step_pat = re.compile(r'\[(\d+\.?\d*)%\]\s*Step\s+(\d+)\s*\|\s*loss=([\d.]+)')
    step_m = step_pat.search(log_text)
    if step_m:
        m['progress_pct'] = float(step_m.group(1))
        m['step'] = step_m.group(2)
        m['loss'] = step_m.group(3)

    lr_pat = re.compile(r'LR=([\d.e-]+)')
    lr_m = lr_pat.search(log_text)
    if lr_m:
        m['lr'] = lr_m.group(1)

    vram_pat = re.compile(r'VRAM=(\d+)MB')
    vram_m = vram_pat.search(log_text)
    if vram_m:
        m['vram'] = vram_m.group(1)

    speed_pat = re.compile(r'([\d.]+)step/s')
    speed_m = speed_pat.search(log_text)
    if speed_m:
        m['speed'] = speed_m.group(1) + ' step/s'

    eta_pat = re.compile(r'ETA\s+([\d:]+)')
    eta_m = eta_pat.search(log_text)
    if eta_m:
        m['eta'] = eta_m.group(1)

    phase_pat = re.compile(r'Phase=(\w+)')
    phase_m = phase_pat.search(log_text)
    if phase_m:
        m['phase'] = phase_m.group(1)

    # Iteration
    iter_pat = re.compile(r'ITERATION\s+(\d+)')
    iter_m = iter_pat.search(log_text)
    if iter_m:
        m['iteration'] = iter_m.group(1)

    # Total steps
    total_pat = re.compile(r'Step\s+\d+\s*/\s*(\d+)')
    total_m = total_pat.search(log_text)
    if total_m:
        m['total_steps'] = total_m.group(1)

    # Generation samples
    gen_lines = [l for l in log_text.split('\n') if '[network]' in l.lower() or '[code]' in l.lower() or '[text]' in l.lower() or '[security]' in l.lower()]
    for g in gen_lines[-8:]:
        m['generations'].append(g.strip())

    # Config
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                cfg = json.load(f)
            m['d_model'] = cfg.get('d_model', '?')
            m['d_sparse'] = cfg.get('d_sparse', '?')
            m['n_layers'] = cfg.get('n_layers', '?')
            m['d_state'] = cfg.get('d_state', '?')
            m['best_loss'] = cfg.get('best_loss', '?')
            m['phase'] = cfg.get('current_phase', m['phase'])
            m['domain_weights'] = cfg.get('domain_weights', {})
            if isinstance(cfg.get('generation_step'), int):
                m['total_steps'] = cfg['generation_step']
        except:
            pass

    # Params from model
    params_pat = re.compile(r'([\d.]+)M Parameter')
    params_m = params_pat.search(log_text)
    if params_m:
        m['params'] = params_m.group(1) + 'M'

    # Vocab from data loader
    vocab_pat = re.compile(r'Vocab:\s*(\d+)')
    vocab_m = vocab_pat.search(log_text)
    if vocab_m:
        m['vocab_size'] = vocab_m.group(1)

    from datetime import datetime
    m['timestamp'] = datetime.now().strftime('%H:%M:%S')

    return m


@app.route('/')
def index():
    log = get_tmux_log()
    metrics = parse_metrics(log)
    return render_template_string(HTML_TEMPLATE, m=metrics)


@app.route('/api/metrics')
def api_metrics():
    log = get_tmux_log()
    metrics = parse_metrics(log)
    return jsonify(metrics)


if __name__ == '__main__':
    print('Dashboard starting on http://0.0.0.0:5050')
    app.run(host='0.0.0.0', port=5050, debug=False)
