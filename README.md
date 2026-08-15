# Anima — CogLang v3 AGI Architecture

**Backprop-freie kognitive Architektur — Predictive Coding + Hebbian Learning.**

> "Nicht Backprop. Nicht Autograd. Pure Hebbian Intelligence."

---

## Architecture Overview

**34 CogModule classes · 7635 lines (coglang.py) · 58 phases**

CogLang v3 is a complete AGI architecture with zero backpropagation. All learning happens via Hebbian rules, predictive coding errors, and local plasticity. The system autonomously evolves through multi-domain BPE data.

| Metric | Value |
|--------|-------|
| Modules | 34 CogModule classes |
| Parameters | ~385.5M |
| VRAM | ~1997MB (25% of 8GB RTX 2070 SUPER) |
| Speed | 2.6 step/s (throttled, 2 threads) |
| Precision | FP16/FP32 Mixed |
| Training | Step ~600+/50000, ~5.5h ETA |
| Loss | ~57 (foundation phase, warmup LR) |
| Best Loss | 2.78 (from checkpoint) |

---

## All 57 Phases

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

### Deep Cognition & Meta-Cognition (44-57)
| # | Phase | Class | Description |
|---|-------|-------|-------------|
| 44 | Causal Reasoning | `CausalReasoning` | Causal structure discovery, intervention inference, world-model cause-effect |
| 45 | System-2 Reasoning | `System2Reasoning` | Chain-of-thought, tree-of-thought, verification, slow deliberate reasoning |
| 46 | Imagination & Planning | `ImaginationPlanning` | Future simulation, plan generation, imagination vs. reality learning |
| 47 | Exploration Drive | `ExplorationDrive` | Active knowledge-gap search, novelty bonus, curiosity-driven data selection |
| 48 | MetaKognition | `MetaKognition` | Thinking about thinking: strategy selection, confidence calibration, resource allocation |
| 49 | Hierarchical Memory | `HierarchicalMemory` | 5-level memory hierarchy (sensory/working/episodic/semantic/procedural) with consolidation |
| 52 | Hierarchical Goal | `HierarchicalGoal` | Goal decomposer, subgoal tracker, goal adaptation to environment |
| 55 | Meta-Learning | `MetaLearning` | Learning to learn: strategy encoder, meta-predictor, hyperparameter controller |
| 56 | Active Learning | `ActiveLearning` | Self-directed learning: uncertainty sampler, query mechanism, curriculum on demand |
| 57 | Uncertainty-Coupled Meta-Learning | *(MetaLearning upgrade)* | ActiveLearning uncertainty modulates hyperparameters (conservative LR, more exploration); meta-HPs now applied to real optimizer LR |
| 58 | Knowledge-Gap Weighted Learning | *(ActiveLearning upgrade)* | Knowledge-gap tokens get stronger Hebbian updates (per-token boost in PredictiveLayer, up to 2x) — targeted learning of unknown vocabulary |

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
├── coglang.py                  # 7635 lines — all 34 CogModule classes + CogLang controller
├── coglang_evolve.py           # 1183 lines — autonomous evolution loop (domain sampling, sleep, reflection, stats)
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
MetaLearning (strategy + HP control)        │
    │                                       │
    ▼                                       │
ActiveLearning (uncertainty modulation)     │
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
- **Throttled:** 25% VRAM fraction, 2 CPU threads, sleeps between steps — PC remains usable
- **Resilient:** NaN guards, checkpoint fallback, strict=False loading, weight clamping
- **Self-directed:** ActiveLearning decides which data to sample; MetaLearning tunes its own hyperparameters from uncertainty (Phase 57)
- **Composable:** All 34 modules share the same `d_model`/`d_sparse` interface

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

All 34 CogModule subclasses (in order of appearance in `coglang.py`):

| # | Class | Line | Description |
|---|-------|------|-------------|
| 1 | `EpisodicMemory` | 361 | Working memory across sequences |
| 2 | `SensoryInput` | 458 | BPE embedding lookup |
| 3 | `SparseEncoder` | 477 | d_model → d_sparse projection |
| 4 | `HebbianAttention` | 501 | Hebbian self-attention |
| 5 | `PredictiveAttention` | 557 | Error-driven attention |
| 6 | `PredictiveLayer` | 625 | Single PC layer: predict, encode, error |
| 7 | `SelfModel` | 753 | Meta-cognitive confidence |
| 8 | `PredictiveStack` | 835 | Multi-layer PC stack |
| 9 | `HierarchicalPC` | 899 | 3-level PC hierarchy |
| 10 | `OutputDecoder` | 1162 | d_sparse → vocab |
| 11 | `ActiveInference` | 1192 | Kalman filter + epistemic value |
| 12 | `SleepReplay` | 1400 | Priority replay + consolidation |
| 13 | `NeuroSymbolicBridge` | 1598 | Rule-based modulation |
| 14 | `EvolutionStrategyOptimizer` | 1655 | Population-based optimization |
| 15 | `SkillModule` | 1828 | Specialized sub-networks |
| 16 | `SecurityHead` | 1886 | CWE anomaly detection |
| 17 | `NetworkEncoder` | 1952 | Protocol encoding |
| 18 | `GoalEncoder` | 2012 | Goal-directed generation |
| 19 | `SelfReflection` | 2091 | Meta-cognitive self-critique |
| 20 | `KnowledgeGraph` | 2275 | Entity-relation graph |
| 21 | `ToolUse` | 2547 | External tool calling |
| 22 | `MultiAgent` | 2820 | Dual-persona debate |
| 23 | `TransferLearning` | 3021 | Domain adapters + few-shot |
| 24 | `ConsciousnessGlimpse` | 3310 | Global workspace broadcasting |
| 25 | `AutoCurriculum` | 3542 | ZPD difficulty adaptation |
| 26 | `CausalReasoning` | 3773 | Causal structure discovery |
| 27 | `System2Reasoning` | 4005 | Chain/tree-of-thought reasoning |
| 28 | `ImaginationPlanning` | 4250 | Future simulation + planning |
| 29 | `ExplorationDrive` | 4507 | Knowledge-gap search |
| 30 | `MetaKognition` | 4767 | Thinking about thinking |
| 31 | `HierarchicalMemory` | 5090 | 5-level memory hierarchy |
| 32 | `HierarchicalGoal` | 5604 | Goal decomposition |
| 33 | `MetaLearning` | 6003 | Learning to learn |
| 34 | `ActiveLearning` | 6437 | Self-directed active learning |

---

## Training Status

The model is currently training autonomously (fresh run, Phases 55-57 active):
- **Step:** ~600+/50000 (foundation phase)
- **Loss:** ~57 (expected variance during LR warmup)
- **VRAM:** ~1997MB (25% of 8GB, stable)
- **LR:** ~0.12 (climbing to 0.5 at step 5000)
- **Speed:** 2.6 step/s
- **Domain:** foundation (text only)
- **Meta-Learning:** active — HP control coupled to ActiveLearning uncertainty (Phase 57)
- **ActiveLearning:** active — 1 query/step, knowledge-gap tracking, domain preference weights
- **train_state.json:** full module stats (meta_learning, active_learning, causal, reasoning, …)

After warmup (step 5000), the loss should drop as cosine annealing kicks in.

---

## License

Private repository — All rights reserved.

---

*Developed for exploration of backprop-free AGI architectures.*
