# Anima — CogLang v3 AGI Architecture

**Backprop-freie kognitive Architektur — Predictive Coding + Hebbian Learning.**

> "Nicht Backprop. Nicht Autograd. Pure Hebbian Intelligence."

---

## Architecture Overview

**33 CogModule classes · 4440 lines · 43 phases**

CogLang v3 is a complete AGI architecture with zero backpropagation. All learning happens via Hebbian rules, predictive coding errors, and local plasticity. The system autonomously evolves through multi-domain BPE data.

| Metric | Value |
|--------|-------|
| Modules | 33 CogModule classes |
| Parameters | ~324M |
| VRAM | ~1939MB (45% of 8GB RTX 2070 SUPER) |
| Speed | 1.8 step/s (throttled, 2 threads) |
| Precision | FP16/FP32 Mixed |
| Training | Step 2722+/50000, ~7h ETA |
| Loss | ~54 (warmup phase, LR climbing) |
| Best Loss | 2.78 (from checkpoint) |

---

## All 43 Phases

### Core Architecture (1-15)
| # | Phase | Class | Description |
|---|-------|-------|-------------|
| 1 | EpisodicMemory | `EpisodicMemory` | Working memory across sequences, read/write/forget |
| 2 | Meta-Plasticity | `CogModule` base | Self-controlled learning rate from prediction error |
| 3 | Predictive Attention | `PredictiveAttention` | Hebbian attention mechanism over error signals |
| 4 | Continual Learning | `CogModule` base | EWC elastic weight consolidation against forgetting |
| 5 | Multi-Scale Hierarchy | `PredictiveStack` | 6 layers, each with different temporal scales |
| 6 | Self-Model | `SelfModel` | Meta-cognitive uncertainty + self-confidence estimation |
| 7 | Intrinsic Motivation | `ActiveInference` superseded | Curiosity via prediction error (→ Phase 33) |
| 8 | Neuro-Symbolic Bridge | `NeuroSymbolicBridge` | Logical rules modulate predictions |
| 9 | Multi-GPU Pipeline | *(external)* | Secondary GPU for parallel evaluation |
| 10 | Streaming Data Pipeline | *(external)* | Memory-mapped dataset streaming |
| 11 | Hebbian Transformer | `HebbianAttention` | Self-attention with Hebbian weight updates |
| 12 | Online Evaluation | *(in evolve)* | Automatic quality metrics |
| 13 | Gradient-Free Optimizer | `EvolutionStrategyOptimizer` | ES for population-based weight updates |
| 14 | Modular Skills | `SkillModule` | Specialized sub-networks for different tasks |
| 15 | Efficiency Suite | *(various)* | Mixed precision, async loading, dynamic batching |

### Training Control (16-24)
| # | Phase | Description |
|---|-------|-------------|
| 16 | Pause/Resume/Stop | `training_controller.py` — control training remotely |
| 17 | Resource Throttle | CPU/GPU throttling for parallel usability |
| 18 | Code Scraper | GitHub + StackOverflow data collection |
| 19 | Code Tokenizer | AST-aware code tokenization |
| 20 | Multi-Source Pipeline | Combined code + text data streams |
| 21 | Sparse Weight Updates | 60% sparsity in weight updates |
| 22 | NaN-Guards | NaN detection + fallback + weight decay |
| 23 | Cosine Annealing LR | Adaptive learning rate scheduling |
| 24 | CUDA Graphs | 2-3x GPU kernel replay optimization |

### Data & Tokenization (25-30)
| # | Phase | Class | Description |
|---|-------|-------|-------------|
| 25 | Multi-URL Dataset | *(external)* | Fallback chain: TinyStories → PG-19 → Shakespeare |
| 26 | BPE Tokenizer | `BPETokenizer` | Byte-pair encoding with 4096 vocab |
| 27 | Multi-Domain Data | *(in evolve)* | Text + Code + Math + Security domains |
| 28 | Domain Router | *(in evolve)* | Routes data by domain with curriculum weights |
| 29 | Multi-Task Learning | *(in evolve)* | Combined CE loss across domains |
| 30 | Code-Native Tokenizer | `CodeTokenizer` | Syntax-aware code tokenization (AST-level) |

### Security & Networking (31-32)
| # | Phase | Class | Description |
|---|-------|-------|-------------|
| 31 | Vulnerability Detection | `SecurityHead` | CWE-aware anomaly detection from prediction error |
| 32 | Network Traffic | `NetworkEncoder` | Protocol-aware network traffic encoding |

### Advanced Cognition (33-43)
| # | Phase | Class | Description |
|---|-------|-------|-------------|
| 33 | Active Inference | `ActiveInference` | Kalman filter per domain, epistemic value, free energy, curiosity-gated sampling |
| 34 | Sleep Replay | `SleepReplay` | Priority replay buffer, consolidation, pattern pruning |
| 35 | Hierarchical PC | `HierarchicalPC` | 3-level hierarchy (token/phrase/concept), bottom-up encoding, top-down prediction |
| 36 | Goal Encoder | `GoalEncoder` | Goal-directed generation with beam search, goal-conditioned sampling |
| 37 | Self-Reflection | `SelfReflection` | Meta-cognitive self-critique, confidence scoring, contradiction detection |
| 38 | Knowledge Graph | `KnowledgeGraph` | Entity-relation graph, Hebbian triple learning, context retrieval |
| 39 | Tool Use | `ToolUse` | External tool calling (calculator, python, search), `[TOOL:name:arg]` pattern |
| 40 | Multi-Agent | `MultiAgent` | Dual-persona (focus/creative), agreement scoring, synthesis |
| 41 | Transfer Learning | `TransferLearning` | LoRA adapters per domain, few-shot buffer, transfer matrix |
| 42 | Consciousness Glimpse | `ConsciousnessGlimpse` | Global Workspace: salience spotlight, broadcast, recurrent coherence |
| 43 | Auto-Curriculum | `AutoCurriculum` | ZPD-based difficulty adaptation, 5 levels, mastery tracking |

---

## Quick Start

### Start Training
```bash
wsl -d Ubuntu-24.04
cd /home/anima && source ~/venv/bin/activate
nice -n 19 python3 coglang_evolve.py
```

### Control Training
```bash
python3 training_controller.py pause    # Pause
python3 training_controller.py resume   # Resume
python3 training_controller.py stop     # Stop
python3 training_controller.py status   # Status
```

### Chat with Model
```bash
python3 chat.py
```

---

## Project Structure

```
anima/
├── coglang.py                  # 4440 lines — all 33 CogModule classes + CogLang controller
├── coglang_evolve.py           # 1058 lines — autonomous evolution loop (domain sampling, sleep, reflection)
├── coglang_train.py            # Single training run
├── training_controller.py      # Pause/resume/stop controller
├── code_scraper.py             # GitHub + StackOverflow scraper
├── code_tokenizer.py           # Code-specific tokenizer
├── data_loader.py              # Multi-source data pipeline
├── streaming_data.py           # Memory-mapped dataset streaming
├── multi_gpu_eval.py           # Multi-GPU evaluation
├── test_agent.py               # Automated validation
├── guardian_daemon.py          # Process monitor
├── chat.py                     # Interactive CLI
├── README.md                   # This file
├── evolution_config.json       # Training configuration
├── train_state.json            # Live training state
└── data/                       # Training data
    ├── input.txt               # Shakespeare + TinyStories
    └── code/                   # Scraped code data
```

---

## Architecture Flow

```
Input IDs
    │
    ▼
SensoryInput (BPE Embedding) ──────────────┐
    │                                       │
    ▼                                       │
SparseEncoder (d_model → d_sparse)          │
    │                                       │
    ▼                                       │
EpisodicMemory (retrieve context)           │
    │                                       │
    ▼                                       │
PredictiveStack / HierarchicalPC            │
  ├── Level 3 (concept, 16:1 compressed)    │
  ├── Level 2 (phrase, 4:1 compressed)      │
  └── Level 1 (token, full resolution)      │
    │                                       │
    ▼                                       │
ConsciousnessGlimpse (salience broadcast)   │
    │                                       │
    ▼                                       │
SkillModule (gate modulation)               │
    │                                       │
    ▼                                       │
KnowledgeGraph (retrieve + condition)        │
    │                                       │
    ▼                                       │
ActiveInference (free energy, curiosity)    │
    │                                       │
    ▼                                       │
GoalEncoder (goal conditioning)             │
    │                                       │
    ▼                                       │
MultiAgent (dual-persona synthesis)         │
    │                                       │
    ▼                                       │
TransferLearning (domain adapter)           │
    │                                       │
    ▼                                       │
AutoCurriculum (difficulty embedding)       │
    │                                       │
    ▼                                       │
OutputDecoder (d_sparse → vocab) ──────────► Logits
    │                                       │
    ▼                                       │
SelfReflection (confidence, consistency)    │
    │                                       │
    ▼                                       │
ToolUse (detect [TOOL:] calls)              │
    │                                       │
    ▼                                       │
NeuroSymbolicBridge (rule modulation)       │
    │                                       │
    ▼                                       │
SleepReplay (consolidation, pruning)        │
    │                                       │
    ▼                                       │
EvolutionStrategy (population update)       │
```

---

## Key Design Decisions

- **Zero backpropagation:** All updates via Hebbian rules, predictive coding, and local plasticity
- **No autograd:** `with torch.no_grad()` everywhere — pure Hebbian
- **Modular:** Each phase is a CogModule subclass, composable via factory methods
- **Autonomous:** `coglang_evolve.py` runs indefinitely with domain curriculum, sleep phases, self-reflection
- **Throttled:** 45% VRAM fraction, 2 CPU threads, sleeps between steps — PC remains usable
- **Resilient:** NaN guards, checkpoint fallback, strict=False loading, weight clamping
- **Composable:** All 33 modules share the same `d_model`/`d_sparse` interface

---

## Configuration (`evolution_config.json`)

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

## Monitoring

Check training progress in real-time:
```bash
# Live log
tail -f /home/anima/evolve.log

# Training state
cat /home/anima/train_state.json | python3 -m json.tool

# Checkpoints
ls -la /home/anima/checkpoints/
```

---

## Module Reference

All 33 CogModule subclasses (in order of appearance):

| # | Class | Line | Description |
|---|-------|------|-------------|
| 1 | `CodeTokenizer` | 14 | BPE tokenizer, 4096 vocab |
| 2 | `BPETokenizer` | 104 | Byte-pair encoding |
| 3 | `SensoryInput` | 203 | Embedding + BPE lookup |
| 4 | `SparseEncoder` | 254 | d_model → d_sparse projection |
| 5 | `PredictiveLayer` | 283 | Single PC layer: predict, encode, error |
| 6 | `PredictiveStack` | 454 | 6-layer PC stack |
| 7 | `HebbianAttention` | 576 | Hebbian self-attention |
| 8 | `PredictiveAttention` | 688 | Error-driven attention |
| 9 | `SelfModel` | 748 | Meta-cognitive confidence |
| 10 | `EpisodicMemory` | 808 | Working memory |
| 11 | `OutputDecoder` | 907 | d_sparse → vocab |
| 12 | `ActiveInference` | 1191 | Kalman filter + epistemic value |
| 13 | `SleepReplay` | 1399 | Priority replay + consolidation |
| 14 | `HierarchicalPC` | 1599 | 3-level PC hierarchy |
| 15 | `NeuroSymbolicBridge` | 1760 | Rule-based modulation |
| 16 | `EvolutionStrategyOptimizer` | 1874 | Population-based optimization |
| 17 | `SkillModule` | 1955 | Specialized sub-networks |
| 18 | `GoalEncoder` | 2010 | Goal-directed generation |
| 19 | `SelfReflection` | 2089 | Meta-cognitive self-critique |
| 20 | `SecurityHead` | 2192 | CWE anomaly detection |
| 21 | `NetworkEncoder` | 2267 | Protocol encoding |
| 22 | `KnowledgeGraph` | 2356 | Entity-relation graph |
| 23 | `ToolUse` | 2500 | External tool calling |
| 24 | `MultiAgent` | 2692 | Dual-persona debate |
| 25 | `TransferLearning` | 3020 | Domain adapters + few-shot |
| 26 | `ConsciousnessGlimpse` | 3309 | Global workspace broadcasting |
| 27 | `AutoCurriculum` | 3539 | ZPD difficulty adaptation |
| 28–33 | *(internal CogLang structures)* | | Stack, memory, targets, etc. |

---

## Training Status

The model is currently training autonomously:
- **Step:** ~2722/50000 (warmup phase)
- **Loss:** ~54 (expected increase during LR warmup)
- **VRAM:** 1939MB (stable)
- **LR:** 0.116 (climbing to 0.5 at step 5000)
- **Speed:** 1.8 step/s
- **Domain:** foundation (text only)
- **Phase Progress:** 4.8%

After warmup (step 5000), the loss should drop as cosine annealing kicks in.

---

## License

Private repository — All rights reserved.

---

*Developed for exploration of backprop-free AGI architectures.*
