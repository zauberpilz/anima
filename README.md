# 🧠 Anima — CogLang v3 AGI Architecture

**Eine backprop-freie kognitive Architektur basierend auf Predictive Coding und Hebbian Learning.**

> "Nicht Backprop. Nicht Autograd. Pure Hebbian Intelligence."

---

## 📊 Architektur-Übersicht

CogLang v3 ist eine vollständige AGI-Architektur mit **21 implementierten Phasen**:

| Phase | Feature | Beschreibung |
|-------|---------|--------------|
| 1 | **Working Memory** | Episodischer Speicher für Kontext über Sequenzen hinweg |
| 2 | **Meta-Plastizität** | Modell steuert eigene Lernrate basierend auf Error |
| 3 | **Predictive Attention** | Hebbian-basierter Aufmerksamkeitsmechanismus |
| 4 | **Continual Learning** | EWC verhindert katastrophales Vergessen |
| 5 | **Multi-Scale Hierarchie** | Verschiedene Zeitskalen pro Layer |
| 6 | **Self-Model** | Meta-kognitive Unsicherheit und Self-Confidence |
| 7 | **Intrinsische Motivation** | Neugier-getriebenes Lernen via Prediction Error |
| 8 | **Neuro-Symbolische Brücke** | Logische Regeln modulieren Vorhersagen |
| 9 | **Multi-GPU Pipeline** | Sekundäre GPU für parallele Evaluation |
| 10 | **Streaming Data Pipeline** | Große Datasets ohne RAM-Limit |
| 11 | **Hebbian Transformer Hybrid** | Self-Attention mit Hebbian Learning |
| 12 | **Online Evaluation** | Automatische Quality-Metriken |
| 13 | **Gradient-Free Optimizer** | Evolution Strategies für Weight Updates |
| 14 | **Modularer Skill-Mechanismus** | Spezialisierte Sub-Netzwerke |
| 15 | **Efficiency Suite** | Mixed Precision, Async Loading, Dynamic Batching |
| 16 | **Pause/Resume/Stop** | Training kontrollierbar machen |
| 17 | **Resource Throttle** | CPU/GPU Drosselung für paralleles Surfen |
| 18 | **Code Scraper** | GitHub + StackOverflow als Trainingsdaten |
| 19 | **Code Tokenizer** | Spezieller Tokenizer für Programmiersprachen |
| 20 | **Multi-Source Pipeline** | Code + Text kombiniert |
| 21 | **Sparse Weight Updates** | Nur signifikante Gradienten updaten (60% sparen) |

---

## 🚀 Quick Start

### Training starten
```bash
wsl -d Ubuntu-24.04
cd /home/anima && source ~/venv/bin/activate
python3 coglang_evolve.py
```

### Training steuern (während es läuft)
```bash
python3 training_controller.py pause   # ⏸️ Pause
python3 training_controller.py resume  # ▶️ Fortsetzen
python3 training_controller.py stop    # 🛑 Stoppen
python3 training_controller.py status  # 📊 Status
```

### Code-Daten scrapen
```bash
python3 code_scraper.py  # Scrapt GitHub Repos + StackOverflow
```

### Mit dem Modell chatten
```bash
python3 chat.py  # Interaktive CLI
```

---

## 🏗️ Projektstruktur

```
anima/
├── coglang.py              # Core Architecture (CogLang v3)
├── coglang_evolve.py       # Autonomous Evolution Loop
├── coglang_train.py        # Single training run
├── coglang_v3.py           # Full AGI architecture source
├── training_controller.py  # Pause/Resume/Stop controller
├── code_scraper.py         # GitHub/StackOverflow scraper
├── code_tokenizer.py       # Code-specific tokenizer
├── data_loader.py          # Multi-source data pipeline
├── streaming_data.py       # Memory-mapped dataset streaming
├── multi_gpu_eval.py       # Multi-GPU evaluation pipeline
├── test_agent.py           # Automated code validation
├── guardian_daemon.py      # Process monitoring daemon
├── chat.py                 # Interactive CLI interface
└── data/                   # Training data directory
    ├── input.txt           # Shakespeare dataset
    └── code/               # Scraped code data
```

---

## ⚙️ Konfiguration

Die Evolution-Konfiguration wird in `evolution_config.json` gespeichert:

```json
{
    "d_model": 384,
    "d_sparse": 2048,
    "n_layers": 6,
    "d_state": 128,
    "d_context": 256,
    "lr": 0.05,
    "max_vram_mb": 4500,
    "generation_step": 50000,
    "use_code_data": false
}
```

---

## 📈 Performance

| Metrik | Wert |
|--------|------|
| Parameter | 324.2M |
| VRAM Usage | ~4.6GB / 8GB |
| Speed | 1-4 step/s |
| Precision | FP16/FP32 Mixed |
| CPU Threads | 4 (begrenzt) |

---

## 🛡️ Sicherheit

- **SafetyCore**: Immutable, keine Gewalt-Verherrlichung
- **Weight Clamping**: Verhindert Explosion der Gewichte
- **NaN Protection**: Automatische Erkennung und Recovery
- **VRAM Limits**: Konfigurierbare OOM-Prävention

---

## 📝 Lizenz

Private Repository — Alle Rechte vorbehalten.

---

*Entwickelt mit ❤️ für die Erforschung backprop-freier KI-Architekturen.*
