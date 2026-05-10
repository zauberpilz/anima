# Anima — Backprop-freies Predictive Coding Netzwerk

Ein selbstverbesserndes, ressourcenschonendes kognitives System jenseits traditioneller LLM-Architekturen. Anima verwendet Predictive Coding, Sparse Activation und lokale Lernregeln (Hebbian/Delta) — **kein Backprop**.

## Architektur

```
Input → Embed → SparseEncoder → PredictiveStack → OutputDecoder → Logits
                                        ↕
                               SparseAssociativeMemory
                                        ↕
                               MetaController + SelfMonitor
```

### Komponenten

| Komponente | Beschreibung |
|---|---|
| **SparseEncoder** | Wandelt dichte Embeddings in spärliche Repräsentationen (~2% Aktivierung) |
| **PredictiveCodingLayer** | Sagt Input aus State+Context voraus, lernt via NLMS (Normalized LMS) |
| **PredictiveStack** | Stapelt 4 PredictiveCodingLayer hierarchisch mit tanh-Zwischenschichten |
| **OutputDecoder** | Projiziert Prediction-Mix via tanh-Hidden → Vokabular, lernt via Delta-Regel |
| **SparseAssociativeMemory** | Inhaltsadressierbarer Speicher mit LSH-Hashing und Konsolidierung |
| **MetaController** | Steuert Compute-Allokation, Explore/Exploit, Temperatur |
| **SelfMonitor** | Überwacht Metriken (Error, Dichte, Speicher) und gibt Optimierungsvorschläge |
| **SafetyCore** | Unveränderbare Sicherheitsschicht (Violationserkennung, User Rules) |

### Lernverfahren

- **Kein Backpropagation!** Alle Gewichte lernen via lokale Hebbian/Delta-Regeln
- **Predictive Layer**: NLMS (Normalized Least Mean Squares) mit Momentum
- **Output Decoder**: Delta-Regel für Softmax-Cross-Entropy mit Label Smoothing
- **Gate**: Lernt Öffnungsrate proportional zum Prediction Error
- **Momentum**: Beschleunigt Konvergenz der Hebbian-Updates

## Training

```bash
python train_10k.py
```

Trainiert auf TinyShakespeare (~200K Zeichen) mit:
- Batch: 16 × 128 Tokens
- Learning Rate: 0.05 (NLMS) / 0.05 (Output Decoder)
- Momentum: 0.9
- Label Smoothing: 0.1
- ~96 steps/s auf RTX 2070 SUPER

### Ergebnisse (10K Steps)

| Metrik | Start | Ende |
|---|---|---|
| Loss | 4.13 (random) | 3.34 |
| VRAM | — | 223 MB |
| Speed | — | 96 step/s |

Loss-Ziel: ln(62) ≈ 4.13 (random) → ~2.0 (gutes Character-Level LM)

## Philosophie

Anima ist kein Transformer. Anima ist ein Versuch, 
kognitive Prinzipien aus den Neurowissenschaften in 
einer effizienten, backprop-freien Architektur 
umzusetzen:

1. **Predictive Coding**: Das Gehirn sagt ständig sensorischen Input voraus und lernt aus Überraschung (Prediction Error)
2. **Sparse Distributed Representations**: Nur ~2% der Neuronen sind aktiv — effizient und robust
3. **Lokale Lernregeln**: Synapsen lernen von lokalen Signalen, nicht von globalen Gradienten
4. **Persistenter Zustand**: Jede Schicht hat einen internen State, der über Zeit skaliert
5. **Meta-Lernen**: Das System beobachtet sich selbst und optimiert seine Hyperparameter

## Projektstruktur

```
anima/
├── __init__.py
├── network.py      # Hauptarchitektur
├── predictive.py   # Predictive Coding Layer mit Hebbian/Momentum
├── sparse.py       # Sparse Encoder (Top-k Aktivierung)
├── memory.py       # SparseAssociativeMemory
├── meta.py         # MetaController + SelfMonitor
├── safety.py       # SafetyCore (unveränderbar)
└── data.py         # Shakespeare-Datenloader
train_10k.py        # 10K-Step Training
self_improve.py     # Autonome Selbstverbesserung
```

## Lizenz

MIT
