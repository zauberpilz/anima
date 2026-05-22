# CogLang v3 — Exhaustive Experiment & Improvement Catalog

> PhD-thesis-level outline covering every component's variations, hyperparameters, ablations, optimizations, research connections, and failure modes.

---

# Table of Contents

1. [Char-Level Tokenizer (91 vocab)](#1-char-level-tokenizer)
2. [SensoryInput Embedder](#2-sensoryinput-embedder)
3. [SparseEncoder](#3-sparseencoder)
4. [PredictiveLayer / PredictiveStack](#4-predictivelayer--predictivestack)
5. [HebbianAttention](#5-hebbianattention)
6. [PredictiveAttention](#6-predictiveattention)
7. [SelfModel](#7-selfmodel)
8. [EpisodicMemory](#8-episodicmemory)
9. [OutputDecoder](#9-outputdecoder)
10. [IntrinsicMotivation](#10-intrinsicmotivation)
11. [NeuroSymbolicBridge](#11-neurosymbolicbridge)
12. [EvolutionStrategyOptimizer](#12-evolutionstrategyoptimizer)
13. [SkillModule](#13-skillmodule)
14. [MixedPrecisionManager](#14-mixedprecisionmanager)
15. [DynamicBatchSizer](#15-dynamicbatchsizer)
16. [AsyncDataLoader](#16-asyncdataloader)
17. [CogModule Base (Meta-Plasticity + EWC + Hebbian Update)](#17-cogmodule-base)
18. [CogLang Controller / Architecture-Level](#18-coglang-controller)
19. [Cross-Cutting Concerns](#19-cross-cutting-concerns)

---

## 1. Char-Level Tokenizer

### 1.1 Variations / Alternatives

1. **BPE (Byte-Pair Encoding)**: Merge frequent character pairs iteratively. Standard in GPT-2/3/4. Variable-length tokens capture subword patterns. Better compression for natural language.
2. **Unigram LM Tokenizer (SentencePiece)**: Probabilistic subword segmentation. Trains with EM on a language model loss. Supports both raw text and pre-tokenized.
3. **WordPiece**: Like BPE but merges based on likelihood increase. Used by BERT.
4. **Morpheme-aware Tokenizer**: Linguistically motivated segmentation. Use morphological analyzers (e.g., spaCy, Stanza) to split into morphemes. Could improve generalization for agglutinative languages.
5. **MinHash / BBPE (Byte-level BPE)**: Operates on raw bytes. No fixed vocab. 256 tokens at byte level, fully lossless.
6. **Canonical Code Tokenizer**: For code, use AST-based tokenization that preserves syntax structure. Tokens = AST nodes.
7. **Adaptive / Online Tokenizer**: Tokenizer that adapts to data distribution during training. Add new tokens for frequent n-grams discovered online.
8. **Multiple Vocabularies**: Separate tokenizers for different modalities (code vs. text vs. math) combined via input embeddings.
9. **n-gram Hashing (Bloom Filter Tokenizer)**: Hash every n-gram into a fixed-size vocabulary. No explicit vocab table. Useful for OOV robustness.
10. **Different Base Vocab Sizes**: 91 is odd. Try 64, 128, 256, 512. Measure compression ratio vs. model perplexity trade-off.

### 1.2 Hyperparameter Sweeps

| Parameter | Range | Notes |
|-----------|-------|-------|
| vocab_size | 32, 64, 91, 128, 256, 512, 1024 | Larger = less compression, more expressivity |
| max_token_length | 1-8 (char), 1-64 (BPE) | Controls granularity |
| min_frequency | 1, 2, 5, 10 | For BPE/Unigram pruning |
| special_tokens | <unk>, <pad>, <bos>, <eos>, <mask> | Impact on generation |

### 1.3 Ablation Studies

- **Remove tokenizer entirely**: Feed raw bytes (vocab=256) — test if char-level bottleneck is helpful.
- **Replace with BPE of varying sizes**: Measure perplexity improvement vs. compute cost.
- **Remove special tokens <pad>, <bos>, <eos>**: Does model learn boundaries implicitly?
- **Train tokenizer on different data**: Code-only vs. text-only vs. mixed. Measure cross-domain transfer.

### 1.4 Optimization Tricks

- **Pre-tokenize and cache on disk**: Avoid re-tokenization on each epoch.
- **Use `torch.data.utils.Dataset` with pre-tokenized tensor**: Store as `.pt` file.
- **Multi-threaded tokenization**: `torch.utils.data.DataLoader(num_workers=...)`.
- **Hash-based dedup of frequent sequences**: Reduce redundancy in training data.
- **Dynamic vocab pruning**: Remove dead tokens (never used) during training.

### 1.5 Research Papers

- Sennrich et al. "Neural Machine Translation of Rare Words with Subword Units" (BPE, 2015)
- Kudo & Richardson "SentencePiece: A simple and language independent subword tokenizer" (2018)
- Radford et al. "Language Models are Unsupervised Multitask Learners" (GPT-2 byte-level BPE, 2019)
- Xue et al. "ByT5: Towards a token-free future with pre-trained byte-to-byte models" (2021)
- Tay et al. "Charformer: Fast character transformers via gradient-based subword tokenization" (2021)

### 1.6 Failure Modes

| Failure | Symptom | Fix |
|---------|---------|-----|
| OOV at inference | Unknown token → garbled | Add fallback to char-level |
| Fragile to typos | Single char changes break BPE segmentation | Use byte-level |
| Vocab imbalance | Some tokens dominate | Re-balance via frequency cap |
| Information loss | Rare chars merged into <unk> | Ensure full coverage |

---

## 2. SensoryInput Embedder

### 2.1 Variations / Alternatives

1. **Scaled Embeddings**: Multiply by `sqrt(d_model)` as in Vaswani et al. Improves gradient flow.
2. **Position-Adaptive Embeddings**: Learn separate embeddings per position. Replace context embedding.
3. **Factorized Embeddings (ALBERT-style)**: Decompose into `vocab_size * d` where `d << d_model`, then project up. Massive parameter reduction.
4. **Weight-Tied Embeddings**: Share weights with OutputDecoder's `out_head`. Common in Transformers. Forces embedding and output to align.
5. **Subword-aware Embeddings**: Sum of token + each sub-character. Like CharBERT.
6. **Hash Embeddings (Svenstrup et al.)**: Multiple hashing functions map token to embedding via feature hashing. No explicit lookup table. Good for OOV.
7. **Continuous Cache Embeddings**: Maintain a learnable cache of frequent token embeddings, updated via Hebbian rule.
8. **n-hot Embeddings**: Each token activates multiple embedding vectors (like product quantization). Increases representational capacity without increasing params.
9. **Fourier / Random Features Embeddings**: Use random projections (based on Rahimi & Recht 2007) instead of learned lookup. Theoretical guarantees for kernel approximation.
10. **Hypernetwork Embeddings**: Small NN that generates embedding weights conditioned on token frequency / recency.

### 2.2 Hyperparameter Sweeps

| Parameter | Range | Notes |
|-----------|-------|-------|
| d_model | 64, 128, 256, 384, 512, 768, 1024 | Core capacity knob |
| embedding_init | uniform(-0.1,0.1), normal(0,0.02), xavier | Impacts training stability |
| max_weight | 1.0, 2.0, 3.0, 5.0 | Clipping threshold |
| lr_scale | 0.01, 0.05, 0.1, 0.2 | Relative to module LR |

### 2.3 Ablation Studies

- **Remove learn_step entirely**: Frozen embeddings. Does model still learn? (Test random embeddings.)
- **Replace with one-hot**: No embedding layer. Direct sparse input.
- **Replace with learned position + content separation**: Remove context embed, add position directly to embedding.
- **Weight tie**: Tie embedding with decoder head. Measure perplexity change.
- **Double embedding**: Separate embeddings for input vs. context.

### 2.4 Optimization Tricks

- **Use `nn.EmbeddingBag` for variable-length sequences**: More memory efficient.
- **L2-normalize embeddings after each update**: Prevents embedding drift.
- **Embedding noise during training**: Add small Gaussian noise for regularization.
- **Adaptive embedding pruning**: Remove embeddings for tokens seen < threshold times.
- **Use `torch.sparse` for embedding gradient update**: Sparse scatter_add is slow — switch to embedding weights in dense.

### 2.5 Research Papers

- Press & Wolf "Using the Output Embedding to Improve Language Models" (2016) — weight tying
- Lan et al. "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations" (2019) — factorized embeddings
- Svenstrup et al. "Hash Embeddings for Efficient Word Representations" (2017)
- Baevski & Auli "Adaptive Input Representations for Neural Language Modeling" (2018)
- Grave et al. "Unbounded Cache Model for Online Language Modeling" (2016)

### 2.6 Failure Modes

| Failure | Symptom | Fix |
|---------|---------|-----|
| Embedding collapse | All tokens converge to same vector | Add regularization, lower LR |
| Embedding explosion | Weights grow unbounded | Tighten max_weight, use L2 norm |
| Stale embeddings for rare tokens | Rare tokens never update | Use separate LR for rare tokens |
| Learning interference | Token 0 (null) gets spurious updates | Ignore <pad> in learn_step |

---

## 3. SparseEncoder

### 3.1 Variations / Alternatives

1. **Top-k with entmax (Peters et al. 2019)**: Replace `torch.topk + sigmoid` with `entmax` or `sparsemax`. Entmax naturally produces sparse probability distributions with a learned alpha parameter. Better gradient properties.
2. **Beta-entmax**: `entmax` with learned beta per dimension. More flexible than top-k.
3. **Winners-Take-All (WTA)**: Only top-1 neuron active. Extreme sparsity. Used in competitive learning.
4. **k-Winners-Take-All (kWTA) with Inhibition**: Active neurons inhibit neighbors via lateral connections (already partially implemented with `lateral_weights` in `anima/sparse.py`).
5. **Gated Sparsity (MoE-style)**: Learnable router with noise + top-k. As in Mixture of Experts (Shazeer et al. 2017). Uses softmax over expert logits.
6. **Stick-breaking / Indian Buffet Process**: Bayesian nonparametric sparsity. Number of active dimensions is unbounded.
7. **Hard Concrete Gating (Louizos et al. 2017)**: Continuous relaxation of discrete masks. Gradients flow through concrete distribution.
8. **FISTA / ISTA Sparsity**: Iterative soft-thresholding. Enforces L1 penalty via proximal operator.
9. **Random Projection + Hashing (LSH)**: Use locality-sensitive hashing instead of top-k. Differentiable via soft approximations.
10. **N:M Structured Sparsity (NVIDIA A100)**: For every M contiguous neurons, exactly N are active. Hardware-efficient. Used in Ampere sparsity.
11. **Adaptive Sparsity via Gating Network**: Small network predicts k per sample. Current `dynamic_sparsity` is heuristic — replace with learned.
12. **ReLU + L1 regularization**: Simple, no top-k needed. `x = relu(Wx); loss += lambda * |x|_1`. Biologically plausible.

### 3.2 Hyperparameter Sweeps

| Parameter | Range | Notes |
|-----------|-------|-------|
| d_sparse | 1024, 2048, 4096, 8192, 16384 | Expansion ratio |
| sparsity (base) | 0.005, 0.01, 0.02, 0.05, 0.1 | Fraction of active neurons |
| k dynamic range | [1, max(1, d_sparse*0.1)] | Adaptive bounds |
| init scale | 0.01, 0.1, 0.5 | Weight initialization |
| ternary | True, False | Binary vs. {-1,0,1} |

### 3.3 Ablation Studies

- **Remove sparsity entirely**: Dense projection (ReLU). Is sparse coding essential?
- **Replace top-k with random subsampling**: Not learned. Controls for code benefit.
- **Replace sigmoid with hard binary (step function + STE)**: Straight-through estimator for binary.
- **Fix sparsity to constant (no adaptive)**: Does dynamic sparsity help?
- **Remove LayerNorm**: Does the encoder need normalization?

### 3.4 Optimization Tricks

- **Use `torch.topk` with `largest=False` for bottom-k**: Anti-sparsity (active inhibition).
- **Fused kernel for top-k + scatter**: Write custom CUDA kernel for `topk + scatter_` in single pass.
- **Use `KeOps` or `triton` for custom sparse kernels**: Avoid Python overhead.
- **Cached top-k indices**: Reuse indices across forward passes if input changes slowly.
- **Sigmoid temperature annealing**: Start with temperature=1.0, anneal to 0.1 for sharper sparsity.

### 3.5 Research Papers

- Olshausen & Field "Emergence of simple-cell receptive field properties by learning a sparse code" (1996) — foundational
- Peters et al. "Sparse Sequence-to-Sequence Models" (entmax, 2019)
- Shazeer et al. "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" (2017)
- Louizos et al. "Learning Sparse Neural Networks through L0 Regularization" (2017)
- Child et al. "Generating Long Sequences with Sparse Transformers" (2019)
- Hubara et al. "Binarized Neural Networks" (2016) — binary activations
- Ahmad & Hawkins "Properties of Sparse Distributed Representations and their Application to Hierarchical Temporal Memory" (2015)
- Nvidia "N:M Sparsity" — ASP (Automatic SParsity) (2020)

### 3.6 Failure Modes

| Failure | Symptom | Fix |
|---------|---------|-----|
| Dead neurons (never activate) | Zero gradient for entire dimensions | Use sigmoid, not hard; add noise; Kaiming init |
| Too sparse → no signal | Loss doesn't decrease | Lower sparsity, increase k |
| Top-k collapse | Same neurons always active | Add lateral inhibition, diversify |
| Adaptive sparsity oscillation | k bounces wildly per step | Smooth k with EMA |
| Gradient through top-k is zero | Top-k is non-differentiable | Use STE (straight-through estimator) or soft top-k |

---

## 4. PredictiveLayer / PredictiveStack

### 4.1 Variations / Alternatives

1. **GRU-style State Update**: Replace simple `(1-gate)*state + gate*delta` with full GRU: `reset_gate, update_gate, candidate_state`. More expressive temporal dynamics.
2. **LSTM-style Gating**: Add cell state `c`, forget gate, input gate, output gate. Better long-range memory within a layer.
3. **Mamba / State-Space Model (SSM) Replacement**: Replace entire predictive coding layer with an SSM (Gu & Dao 2023). Structured state space with selective scan. O(L log L) instead of O(L²).
4. **Multi-Head Prediction**: Multiple W_pred heads, each predicting different aspects of input. Outputs combined via learned mixing.
5. **Prediction of Future Latents**: Instead of predicting input `x`, predict the sparse code of the *next* step. Hierarchical predictive coding (Rao & Ballard 1999).
6. **Cross-Layer Skip Connections**: Connect error signal from layer i directly to prediction of layer i+2. Gradient highway.
7. **Bidirectional Predictive Layers**: Predict from both past and future (within a window). Like ELMo.
8. **Gated Residual Network (GRN)**: `prediction = W_pred(inp); gate = sigmoid(W_gate(inp)); output = gate * prediction + (1-gate) * x`. Skip connection with learned gating.
9. **DeltaNet-style**: Replace NLMS with a delta rule that only updates when prediction error is large. Sparse-in-time updates.
10. **Timescale as Learned Parameter**: Current timescale is fixed per layer. Make it learned via `nn.Parameter(torch.sigmoid(logit) * 0.9 + 0.1)`.
11. **Plasticity-Weighted State**: Each state dimension has learnable plasticity (how much it should change per step). Like AlphaZero's adaptive parameters.
12. **Multiple State Tracks**: Each layer maintains multiple state vectors (fast/slow). Like Hinton's GLOM.

### 4.2 Hyperparameter Sweeps

| Parameter | Range | Notes |
|-----------|-------|-------|
| d_state | 32, 64, 128, 256, 512 | State dimension |
| d_context | 64, 128, 256, 512, 1024 | Context dimension |
| n_layers | 2, 4, 6, 8, 12, 18 | Stack depth |
| timescale range | [0.05, 1.0], [0.1, 0.9], fixed per layer | How slow/fast |
| lr per layer | 0.01-0.1, exponential decay across layers | Layer-specific LR |
| pred_mixer init | uniform, 1/n_layers, softmax | Mixing weights init |

### 4.3 Ablation Studies

- **Remove state entirely**: State dimension = 0. Pure feedforward prediction. Test if recurrence matters.
- **Remove context entirely**: Predictive layers get no position info.
- **Replace state update with learned MLP**: Instead of gated linear.
- **Single layer vs. deep stack**: What depth is necessary?
- **Remove timescale variation**: All layers timescale=1.0.
- **Layer-wise output (no mixing)**: Use only last layer's prediction. Does mixing help?
- **Remove Hebbian learning**: Freeze all weights. Random predictions baseline.
- **Reverse order**: Fast at bottom, slow at top (vs. current).

### 4.4 Optimization Tricks

- **State normalization**: Apply LayerNorm/RMSNorm to state before each forward.
- **Gradient clipping on state update**: `clamp_(state, -max_state, max_state)`.
- **State reset on loss spike**: If loss explodes, reset all states.
- **Parallel forward across layers**: All layers can process independently in a single pass (they're not sequential).
- **State as running average**: `state = 0.999*state + 0.001*delta` for extreme stability.
- **Use `torch.vmap` for per-layer operations**: Vectorize across layers.

### 4.5 Research Papers

- Rao & Ballard "Predictive coding in the visual cortex: a functional interpretation" (1999) — foundational
- Friston "A theory of cortical responses" (2005) — free energy principle
- Ororbia & Kifer "The Neural Coding Framework for Learning Generative Models" (2022) — modern predictive coding
- Gu & Dao "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023)
- Hochreiter & Schmidhuber "Long Short-Term Memory" (1997) — LSTM gates
- Cho et al. "Learning Phrase Representations using RNN Encoder-Decoder" (GRU, 2014)
- Hinton "How to represent part-whole hierarchies in a neural network" (GLOM, 2021)
- Millidge et al. "Predictive Coding: A Theoretical and Experimental Review" (2021)

### 4.6 Failure Modes

| Failure | Symptom | Fix |
|---------|---------|-----|
| State explosion | State values grow unbounded | L2-normalize state, clamp |
| State collapse | State converges to zero | Add leaky integration, bias |
| Layer specialization failure | All layers learn same thing | Add noise, diversify d_state |
| Timescale saturation | All gate=1 or gate=0 | Initialize gates near 0.5 |
| Error vanishing | Bottom layers get no error | Add skip error connections |
| Context embedding overfitting | Context captures noise | Reduce d_context, add dropout |
| Catastrophic forgetting of state | Old context completely overwritten | Add forgetting factor, consolidate |

---

## 5. HebbianAttention

### 5.1 Variations / Alternatives

1. **Linear Attention (Katharopoulos et al. 2020)**: Replace softmax with `phi(Q) @ phi(K)^T @ V` where phi is ELU+1. O(L) instead of O(L²). More Hebbian-friendly: `delta = V @ softmax(K) for each Q`.
2. **Sliding Window Attention (Beltagy et al. 2020)**: Attend only to local window. O(L*w). Better for long sequences.
3. **Dilated / Strided Attention**: Attend to every k-th position. Captures long-range without full O(L²).
4. **Hash-based / Reformer Attention (Kitaev et al. 2020)**: LSH to find similar tokens. O(L log L).
5. **Performer / FAVOR+ (Choromanski et al. 2020)**: Orthogonal random features for kernelized attention. O(L).
6. **Flash Attention (Dao et al. 2022)**: IO-aware exact attention. Fused kernel, tiled. Not an algorithmic change but optimization.
7. **Delta Rule Attention (Schlag et al. 2021)**: Replace softmax with iterative delta rule. Each key updates value towards its target. More associative-memory-like.
8. **Hopfield Network Attention (Ramsauer et al. 2020)**: Modern Hopfield Networks for attention. Energy-based. Dense retrieval.
9. **Fixed Q/K/V (no learned projections)**: Use random Q/K/V. Only learn W_out. Test if learned Q/K/V are necessary.
10. **Q/K/V shared across layers**: Single set of projections. Reduces params, may improve training.
11. **Cross-attention variant**: Attend to memory instead of self. W_q attends to W_k from memory.
12. **Two-tower Hebbian**: Separate fast (current batch) and slow (running average) attention matrices.

### 5.2 Hyperparameter Sweeps

| Parameter | Range | Notes |
|-----------|-------|-------|
| n_heads | 1, 2, 4, 8 | Multi-head count |
| head_dim | 32, 64, 128, 256 | Per-head dimension |
| lr attention | 0.001, 0.005, 0.01, 0.05 | Hebbian learning rate |
| W_out init | 0.01, 0.1, xavier | Output projection init |
| temperature for softmax | 0.5, 1.0, 2.0 | Sharper vs. diffuse attention |

### 5.3 Ablation Studies

- **Remove HebbianAttention entirely**: Stack only uses PredictiveLayer errors.
- **Remove W_out update**: Only learn Q/K/V, freeze W_out.
- **Remove multi-head**: Single head attention.
- **Replace softmax with `torch.sigmoid`**: No normalization across positions.
- **Causal masking**: Should attention look at future tokens?

### 5.4 Optimization Tricks

- **Sparse attention mask**: Precompute mask for local + global attention.
- **Fused softmax + matmul**: Use `torch.nn.functional.scaled_dot_product_attention` (PyTorch 2.0).
- **Attention dropout**: Drop random attention weights for regularization.
- **Tiled computation for long sequences**: Process in chunks, accumulate.
- **Cached attention patterns**: Reuse if input doesn't change much.
- **W_out momentum with Nesterov**: Use Nesterov accelerated gradient for W_out.

### 5.5 Research Papers

- Vaswani et al. "Attention Is All You Need" (2017)
- Katharopoulos et al. "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention" (2020)
- Kitaev et al. "Reformer: The Efficient Transformer" (2020)
- Beltagy et al. "Longformer: The Long-Document Transformer" (2020)
- Choromanski et al. "Rethinking Attention with Performers" (2020)
- Schlag et al. "Linear Transformers Are Secretly Fast Weight Programmers" (2021) — delta rule
- Ramsauer et al. "Hopfield Networks is All You Need" (2020)
- Dao et al. "FlashAttention: Fast and Memory-Efficient Exact Attention" (2022)

### 5.6 Failure Modes

| Failure | Symptom | Fix |
|---------|---------|-----|
| Attention entropy collapse | All mass on one position | Add entropy regularization |
| Diffuse attention | Uniform distribution | Lower temperature, sharpening |
| Hebbian weight explosion | W_out grows unbounded | Weight decay, clamping |
| Multi-head redundancy | All heads same | Add diversity loss |
| Context ignoring | Attention distribution independent of input | Check Q/K projections |

---

## 6. PredictiveAttention

### 6.1 Variations / Alternatives

1. **Learnable Error Modulation**: Replace fixed `error_boost * 0.5` with learned `nn.Parameter` for per-head / per-position error scaling.
2. **Gated Error Integration**: Use sigmoid gate to blend standard attention with error-modulated attention.
3. **Normalized Error Boost**: `error_mag / (error_mag.mean() + eps)` to make relative.
4. **Excitatory-Inhibitory Balance**: Separate positive error (surprising) and negative error (expected) channels. Excitation for novelty, inhibition for familiarity.
5. **Temporal Difference Attention**: `error_t - gamma * error_{t-1}` — attend to where error is *increasing*.
6. **Error-Weighted Keys**: `V = V + error * W_ev`. Inject error into value directly (not just into scores).
7. **Layer-Specific Error Projection**: Separate error projection per layer, not shared across layers.
8. **Predictive Coding Attention (PCA)**: Compute attention using predicted vs. actual, then use error to update prediction. Iterative.
9. **Multi-Scale Error**: Combine errors from multiple layers to modulate a single attention head.
10. **Hierarchical Error Attention**: Bottom layers attend to fine-grained errors, top layers attend to abstract errors.

### 6.2 Hyperparameter Sweeps

| Parameter | Range | Notes |
|-----------|-------|-------|
| error_boost_scale | 0.0, 0.1, 0.5, 1.0, 2.0 | Error modulation strength |
| n_heads | 1, 2, 4, 8 | Same as HebbianAttention |
| temperature | 0.5, 1.0, 2.0 | Attention softmax temperature |

### 6.3 Ablation Studies

- **Disable error modulation**: `error_boost = 0`. Standard attention only.
- **Remove PredictiveAttention entirely**: Only HebbianAttention processes errors.
- **Replace with Kullback-Leibler (KL) attention**: Use KL divergence between predicted and actual as attention signal.
- **Error sign matters**: Test only positive error, only negative error.

### 6.4 Optimization Tricks

- **Error magnitude smoothing**: Use EMA of error magnitude to avoid jitter.
- **Attention gating with meta-plasticity**: High self-confidence → less error modulation.
- **Normalize error across layers**: Prevent one layer's error from dominating.

### 6.5 Research Papers

- Friston "The free-energy principle: a unified brain theory?" (2010)
- Ororbia et al. "The Neural Coding Framework for Learning Generative Models" (2022)
- Spratling "Predictive coding as a model of cognition" (2016)
- Bogacz "A tutorial on the free-energy framework for modelling perception and learning" (2017)

### 6.6 Failure Modes

| Failure | Symptom | Fix |
|---------|---------|-----|
| Error saturation | Error magnitudes dominate (all boost = huge) | Normalize error, cap boost |
| Zero error at start | No modulation signal | Add small constant error |
| Oscillation | Attention flips between positions | Smooth error with EMA |
| Error-attention feedback loop | High error → more attention → more error → ... | Dampen the loop with leaky integration |

---

## 7. SelfModel

### 7.1 Variations / Alternatives

1. **Bayesian Uncertainty**: Instead of heuristic EMA, maintain a Bayesian estimate: `p(error|state)`. Use variational inference to track posterior over errors.
2. **Meta-Cognitive Network (small NN)**: Replace hand-crafted uncertainty with a learned network that predicts confidence from internal states.
3. **Ensemble Uncertainty (Lakshminarayanan 2017)**: Maintain multiple copies of SelfModel, compute variance. Better calibration.
4. **Mutual Information / Epistemic Uncertainty**: Use mutual information between weights and predictions (via dropout or ensemble).
5. **Prediction Interval Network**: Output lower/upper bounds on prediction. Train with quantile regression.
6. **Confidence Calibration**: Add temperature scaling for well-calibrated confidence (Guo et al. 2017).
7. **Self-Attention over Error History**: SelfModel attends to past error traces to detect patterns. Uses EpisodicMemory as query.
8. **Evidential Deep Learning (Sensoy 2018)**: Output Dirichlet concentration parameters. Uncertainty = evidence.
9. **Hesitant Activation**: Replace single linear `W_uncertainty` with a small MLP (2-3 layers) with residual.
10. **Input-dependent modulation**: Learned per-layer weighted modulation (attention over layers).

### 7.2 Hyperparameter Sweeps

| Parameter | Range | Notes |
|-----------|-------|-------|
| n_layers tracked | 2, 4, 6, 8, 12 | SelfModel's capacity |
| EMA decay | 0.8, 0.9, 0.95, 0.99 | Error statistics smoothing |
| modulation scale | 0.05, 0.1, 0.2, 0.5 | How much uncertainty affects output |
| W_uncertainty init | 0.01, 0.1, xavier | Projection init |

### 7.3 Ablation Studies

- **Remove SelfModel entirely**: No meta-cognitive modulation.
- **Remove modulation**: SelfModel tracks statistics but doesn't affect output.
- **Replace with constant confidence**: Always 0.5.
- **Per-layer vs. global uncertainty**: Use layer-specific or overall uncertainty.

### 7.4 Optimization Tricks

- **Track error per position, not just mean**: Capture positional uncertainty.
- **Use log-variance**: Train `log(sigma^2)` for numerical stability.
- **Calibrate confidence on validation set**: Post-hoc temperature scaling.
- **Confidence annealing**: Start with high confidence, reduce as model learns.

### 7.5 Research Papers

- Guo et al. "On Calibration of Modern Neural Networks" (2017)
- Lakshminarayanan et al. "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles" (2017)
- Sensoy et al. "Evidential Deep Learning to Quantify Classification Uncertainty" (2018)
- Gal & Ghahramani "Dropout as a Bayesian Approximation" (2016)
- Kendall & Gal "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?" (2017)
- Depeweg et al. "Decomposition of Uncertainty in Bayesian Deep Learning for Efficient and Risk-sensitive Learning" (2018)

### 7.6 Failure Modes

| Failure | Symptom | Fix |
|---------|---------|-----|
| Overconfident on OOD | Confidence high on garbage | Use OOD detection |
| Underconfident always | Never modulates | Lower confidence threshold |
| Modulation oscillation | Output wavers | Smooth modulation with EMA |
| Layer error statistics stale | SelfModel ignores recent changes | Faster EMA decay |

---

## 8. EpisodicMemory

### 8.1 Variations / Alternatives

1. **Differentiable Neural Computer (DNC) (Graves et al. 2016)**: Full differentiable memory with read/write heads, temporal links, usage tracking.
2. **Neural Turing Machine (NTM) (Graves et al. 2014)**: Content + location-based addressing. Read/write heads with interpolation.
3. **Memory-Augmented Neural Network (MANN) (Santoro et al. 2016)**: Least recently used (LRU) access. Current implementation is LRU-like.
4. **Sparse Associative Memory (already in anima/memory.py)**: LSH-based. Fast O(1) retrieval.
5. **Slot Attention (Locatello et al. 2020)**: Iterative binding of input to slots. Good for object-centric representations.
6. **Transformer-XL / Compressive Memory (Dai et al. 2019)**: Segment-level recurrence with memory cache. Old states not forgotten but compressed.
7. **Fast Weight Programmer (Schlag et al. 2021)**: Memory = weight matrix updated by outer product. Read = matrix-vector product.
8. **Hyperdimensional Computing Memory**: Use HD vectors (10,000+ dims). Binding via circular convolution. Superposition of concepts.
9. **Microsoft's Memorizing Transformer (Wu et al. 2022)**: kNN-augmented attention over large non-differentiable memory.
10. **Titov's Differentiable Retrieval**: Dual-encoder retriever trained end-to-end with Hebbian rules.
11. **EM-set / Kanerva Machine**: Memory slots with EM-based read/write. Expectation-Maximization for sparse memory access.
12. **Temporal Memory Hierarchy**: Multiple memory modules at different timescales (milliseconds to hours). Like HTM (Hawkins).

### 8.2 Hyperparameter Sweeps

| Parameter | Range | Notes |
|-----------|-------|-------|
| memory_size | 16, 32, 64, 128, 256, 512, 4096 | Number of slots |
| d_memory | d/2, d, 2d | Memory vector dimension |
| aging_rate | 0.01, 0.05, 0.1 | How fast slots age |
| lr memory | 0.01, 0.05, 0.1 | Hebbian memory weights |
| target_dim | d_state, d_sparse | Memory projection dimension |
| retrieval temperature | 0.05, 0.1, 0.5 | For attention softmax |

### 8.3 Ablation Studies

- **Remove EpisodicMemory entirely**: No memory augmentation.
- **Disable write**: Only read from random/zero memory.
- **Disable read**: Write only, no retrieval.
- **Replace with simple FIFO queue**: No content-addressable.
- **Replace with random memory**: Random vectors for slots.
- **Single slot vs. many**: What size matters most?
- **Remove projection**: Direct memory access.

### 8.4 Optimization Tricks

- **Write gating**: Only write when error is high (novelty gating).
- **Read thresholding**: Only retrieve if similarity > threshold.
- **Memory compaction**: Cluster similar slots together.
- **Age-weighted retrieval**: Multiply similarity by recency-based weight.
- **Memory defragmentation**: Periodically merge duplicate similar slots.
- **Quantize memory slots**: Use 8-bit storage. 4x memory reduction.
- **Hierarchical memory**: First retrieve coarse, then fine.

### 8.5 Research Papers

- Graves et al. "Neural Turing Machines" (2014)
- Graves et al. "Hybrid computing using a neural network with dynamic external memory" (DNC, 2016)
- Santoro et al. "Meta-Learning with Memory-Augmented Neural Networks" (2016)
- Dai et al. "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context" (2019)
- Wu et al. "Memorizing Transformer" (2022)
- Locatello et al. "Object-Centric Learning with Slot Attention" (2020)
- Schlag et al. "Linear Transformers Are Secretly Fast Weight Programmers" (2021)
- Hawkins & Ahmad "Why Neurons Have Thousands of Synapses, a Theory of Sequence Memory in Neocortex" (2016) — HTM

### 8.6 Failure Modes

| Failure | Symptom | Fix |
|---------|---------|-----|
| Memory saturation | All slots filled, overwriting important | Increase memory_size, consolidate |
| Retrieval noise | Always returns same irrelevant memory | Add diversity penalty |
| Write starvation | Never writes | Lower write threshold |
| Read collapse | Attention degenerates | Temperature annealing |
| Memory slot interference | Similar states overwrite each other | Increase slot dimensionality |

---

## 9. OutputDecoder

### 9.1 Variations / Alternatives

1. **Mixture of Softmax (MoS)**: Multiple softmax heads, combined via learned gating. Better probability distribution modeling (Yang et al. 2017).
2. **Adaptive Softmax (Grave et al. 2017)**: Hierarchical softmax for large vocab. Group rare tokens under clusters. Faster and better probability estimates.
3. **Differentiable Sampling (Gumbel-Softmax)**: Replace argmax with Gumbel-softmax during training. Allows gradient flow through sampling.
4. **Top-k + Rejection Sampling**: During generation, sample from top-k, reject low-probability. Better quality.
5. **Temperature-based Decoder**: Learned temperature per token. More calibrated confidence.
6. **Character-Level Decoder**: Output next char directly instead of logits over full vocab.
7. **Beam Search Integration**: Support beam search in decoder for generation.
8. **Multi-Task Decoder Heads**: Separate heads for: next token prediction + binary classification (is this correct?) + auxiliary loss.
9. **Discrete Residual Output**: Instead of full softmax, predict residual of a base distribution.
10. **Contrastive Decoder**: Learn to distinguish correct next token from incorrect distractors.

### 9.2 Hyperparameter Sweeps

| Parameter | Range | Notes |
|-----------|-------|-------|
| d_model | 128, 256, 512 | Output projection size |
| label_smoothing | 0.0, 0.05, 0.1, 0.2 | Cross-entropy smoothing |
| temperature | 0.5, 0.7, 0.8, 1.0, 2.0 | Generation temperature |
| top_k | 10, 20, 30, 40, 100 | Top-k sampling |
| max_weight | 1.0, 2.0, 3.0 | Weight clipping |

### 9.3 Ablation Studies

- **Remove out_proj**: Direct sparse → vocab prediction.
- **Remove tanh activation**: Linear projection.
- **Replace with learned softmax temperature**: Single learned scalar.
- **Label smoothing off**: Test if needed.
- **Learn step with true gradient vs. Hebbian**: Compare with `torch.optim.SGD` on decoder.

### 9.4 Optimization Tricks

- **Logit normalization**: Normalize logits before softmax. Stabilizes training.
- **Output embedding weight tying**: Tie with input embedding (see 2.1.4).
- **Use `torch.nn.CrossEntropyLoss`** with `ignore_index` for padding.
- **Gradient clipping on decoder weights**: Prevent output head explosion.
- **Dynamic temperature**: Increase temperature when perplexity spikes.

### 9.5 Research Papers

- Yang et al. "Breaking the Softmax Bottleneck: A High-Rank RNN Language Model" (MoS, 2017)
- Grave et al. "Efficient softmax approximation for GPUs" (2017)
- Pereyra et al. "Regularizing Neural Networks by Penalizing Confident Output Distributions" (2017)
- Vaswani et al. "Tensor2Tensor for Neural Machine Translation" (label smoothing, 2018)
- Holtzman et al. "The Curious Case of Neural Text Degeneration" (top-k, nucleus sampling, 2019)

### 9.6 Failure Modes

| Failure | Symptom | Fix |
|---------|---------|-----|
| Output head collapse | All logits near zero | Increase weight init |
| Extreme confidence | One token gets 99% probability | Increase label smoothing |
| Mode collapse | Model generates same token repeatedly | Increase top-k, temperature |
| Rare token never generated | Mode-seeking behavior | Use top-k + repetition penalty |
| Logit explosion | Logits grow to ±inf | Tighter weight clamping, gradient scaling |

---

## 10. IntrinsicMotivation

### 10.1 Variations / Alternatives

1. **ICM (Intrinsic Curiosity Module) (Pathak et al. 2017)**: Two networks: forward dynamics (predict next state given action) + inverse dynamics (predict action from state transition). Curiosity = error in forward model.
2. **RND (Random Network Distillation) (Burda et al. 2018)**: Fixed random network predicts target features. Training a predictor network. Curiosity = prediction error on features. High error = novel state.
3. **Count-Based Exploration (Tang et al. 2017)**: Pseudo-count of state visits via hash. Curiosity = 1/sqrt(count(state)). Simple and effective.
4. **Disagreement-Based (Pathak et al. 2019)**: Ensemble of forward models. Curiosity = variance across ensemble predictions. Epistemic uncertainty.
5. **Information Gain (Houthooft et al. 2016)**: VIME — variational information maximizing exploration. Curiosity = information gain about model parameters.
6. **Empowerment (Klyubin et al. 2005)**: Maximize mutual information between actions and future states. "How much control do I have?"
7. **Predictive Variance**: Curiosity = variance of prediction across multiple forward passes (with dropout).
8. **Learning Progress (Oudeyer et al. 2007)**: Track loss derivative. Curiosity high when loss is decreasing fast (you're learning). Low when saturated.
9. **Meta-Cognitive Curiosity**: SelfModel uncertainty directly drives curiosity. High confidence → low curiosity.
10. **Competence-Based (Baranes & Oudeyer 2013)**: Curiosity for skills near competence frontier. Not too easy, not too hard.

### 10.2 Hyperparameter Sweeps

| Parameter | Range | Notes |
|-----------|-------|-------|
| novelty_threshold | 0.1, 0.5, 1.0, 2.0 | When does error = novel? |
| reward_history_len | 10, 50, 100, 500 | EMA window |
| curiosity_factor_range | [0.1, 2.0], [0.5, 3.0] | Modulation bounds |
| EMA decay | 0.9, 0.95, 0.99 | For curiosity_drive |
| curiosity_bonus_scale | 0.1, 0.5, 1.0 | Learning rate multiplier |

### 10.3 Ablation Studies

- **Remove IntrinsicMotivation entirely**: No curiosity modulation.
- **Constant curiosity**: Fixed factor = 1.0 (no modulation).
- **Fixed reward**: Always the same intrinsic reward.
- **Inverse curiosity**: Low error → more curiosity (satiety). Like boredom.
- **Replace with extrinsic reward**: Use validation loss as reward.

### 10.4 Optimization Tricks

- **Normalize reward**: Divide by running variance. Prevents reward scale drift.
- **Curiosity annealing**: Reduce curiosity over time to stabilize.
- **Hysteresis**: Only reward significant error drops, not noise.
- **Scheduled curiosity**: High at start, low later.
- **Layer-specific curiosity**: Different curiosity per layer (bottom = high, top = low).

### 10.5 Research Papers

- Pathak et al. "Curiosity-driven Exploration by Self-Supervised Prediction" (ICM, 2017)
- Burda et al. "Exploration by Random Network Distillation" (RND, 2018)
- Tang et al. "#Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning" (2017)
- Oudeyer et al. "Intrinsic Motivation Systems for Autonomous Mental Development" (2007)
- Houthooft et al. "VIME: Variational Information Maximizing Exploration" (2016)
- Pathak et al. "Self-Supervised Exploration via Disagreement" (2019)
- Schmidhuber "Formal Theory of Creativity, Fun, and Intrinsic Motivation" (2010)

### 10.6 Failure Modes

| Failure | Symptom | Fix |
|---------|---------|-----|
| Curiosity starvation | Never curious | Lower threshold |
| Curiosity addiction | Always curious, unstable learning | Raise threshold, cap reward |
| Reward hacking | Error stays high artificially | Lagging reward normalization |
| Stagnation | No longer curious about anything | Periodic resets of curiosity |
| Oscillation | Curiosity bounces between 0 and 2 | Smoother EMA, slower adaptation |

---

## 11. NeuroSymbolicBridge

### 11.1 Variations / Alternatives

1. **Logic Tensor Networks (LTN)**: Embed first-order logic formulas as differentiable operations. Rules have truth degrees. Gradient-based learning.
2. **Neural Theorem Provers (Rocktäschel & Riedel 2017)**: Differentiable backward chaining. Prove queries through learned knowledge base.
3. **Graph Neural Network Rule Propagation**: Rules = graph edges. Propagate rule effects through GNN layers.
4. **Differentiable Inductive Logic Programming (dILP) (Evans & Grefenstette 2018)**: Learn logic programs from examples. Neural + symbolic.
5. **Attention-based Rule Retrieval**: Use attention over a rule library. Current implementation is close to this.
6. **Rule Induction via Hebbian Learning**: Learn new rules from correlation patterns. Rules emerge from data.
7. **Probabilistic Soft Logic (PSL)**: Rules as soft constraints on probabilities. Weighted logical formulas.
8. **Hybrid Markov Logic Networks**: Weighted first-order logic. Inference = MCMC over both neural and symbolic vars.
9. **Neural-Symbolic Concept Learning**: Learn symbols from data (like the Concept Bottleneck Model). Then apply symbolic reasoning.
10. **Embedding-based Knowledge Graph Completion**: Use TransE, RotatE, or ComplEx to learn rule-like embeddings. Hebbian updates.
11. **Rule Attention with Prior Knowledge**: Seed rules with known logical constraints (e.g., symmetry, transitivity).
12. **Neuro-Symbolic Program Synthesis**: Generate and execute mini-programs. Think: Differentiable Forth / ∂P.

### 11.2 Hyperparameter Sweeps

| Parameter | Range | Notes |
|-----------|-------|-------|
| n_rules | 4, 8, 16, 32, 64 | Number of rule slots |
| rule_effect_scale | 0.01, 0.05, 0.1, 0.5 | How much rules modulate logits |
| d_rule | d/4, d/2, d | Rule key dimension |
| lr rules | 0.001, 0.005, 0.01 | Rule Hebbian learning rate |
| softmax temperature | 0.5, 1.0, 2.0 | Rule activation sharpness |

### 11.3 Ablation Studies

- **Remove NeuroSymbolicBridge entirely**: No symbolic modulation.
- **Remove learned rules**: Use fixed random rules.
- **Remove update step**: Rules never learn.
- **Vary number of rules**: What happens with 1 rule vs. 64?
- **Replace rule_values with one-hot**: Each rule always outputs the same token.

### 11.4 Optimization Tricks

- **L1 regularization on rule activation**: Encourage sparse rule usage.
- **Rule diversity loss**: Encourage different rules to have different keys.
- **Periodic rule pruning**: Remove rules with zero activation.
- **Rule mirroring**: Pair each rule with its negation.
- **Incorporate external knowledge**: Seed rules from WordNet, ConceptNet.

### 11.5 Research Papers

- Rocktäschel & Riedel "End-to-End Differentiable Proving" (NTP, 2017)
- Evans & Grefenstette "Learning Explanatory Rules from Noisy Data" (dILP, 2018)
- Serafini & Garcez "Logic Tensor Networks: Deep Learning and Logical Reasoning from Data and Knowledge" (2016)
- Kimmig et al. "A Short Introduction to Probabilistic Soft Logic" (PSL, 2012)
- Richardson & Domingos "Markov Logic Networks" (2006)
- Mao et al. "The Neuro-Symbolic Concept Learner: Interpreting Scenes, Words, and Sentences From Natural Supervision" (2019)
- Koh et al. "Concept Bottleneck Models" (2020)

### 11.6 Failure Modes

| Failure | Symptom | Fix |
|---------|---------|-----|
| Rule collapse | All rules converge to same | Add diversity loss |
| Rule never activates | Dead rules | Better init, reinitialize unused |
| Rule overrides learning | Rules dominate neural output | Reduce rule_effect_scale |
| Rule instability | Rules oscillate | Lower rule LR, add momentum |
| Context-independent rules | Rules ignore input | Check rule activation function |

---

## 12. EvolutionStrategyOptimizer

### 12.1 Variations / Alternatives

1. **CMA-ES (Covariance Matrix Adaptation ES)**: State-of-the-art for continuous optimization. Maintains full covariance of search distribution. Much more sample-efficient than vanilla ES.
2. **PGPE (Policy Gradients with Parameter-based Exploration)**: Alternative to ES. Uses reward-weighted regression. Better for high-dim.
3. **Augmented Random Search (ARS) (Mania et al. 2018)**: Simple ES variant. Only uses top-performing perturbations. Very fast.
4. **Guided ES (Maheswaranathan et al. 2018)**: Combine ES with gradient information. Faster convergence.
5. **OpenAI-ES (Salimans et al. 2017)**: Standard ES with mirror sampling, fitness shaping, and rank normalization.
6. **Persistent ES (Vicol et al. 2021)**: Maintain perturbation over multiple steps. Reduces variance.
7. **Genetic Algorithm + Hebbian Hybrid**: Use GA for architecture search, Hebbian for weight updates.
8. **Particle Swarm Optimization (PSO)**: Each particle has velocity + position. Social and cognitive components.
9. **Newton-like ES**: Use second-order information (curvature) from ES perturbations.
10. **Trust Region ES**: Constrain parameter updates to trust region. More stable.
11. **Evolution Strategies with Ray / Population-Based Training (PBT)**: Train population in parallel, exploit best performers.
12. **Bayesian Optimization for LL**: Use GP-UCB to select hyperparameters. Better sample efficiency.
13. **Gradient-free + Gradient hybrid**: Use ES for global exploration, Hebbian for local refinement.
14. **Species-based ES**: Multiple species with different hyperparameters. Compete and evolve.

### 12.2 Hyperparameter Sweeps

| Parameter | Range | Notes |
|-----------|-------|-------|
| population_size | 4, 8, 16, 32, 64, 128 | Bigger = more robust, more compute |
| sigma | 0.001, 0.005, 0.01, 0.05, 0.1 | Perturbation noise |
| update_alpha | 0.01, 0.05, 0.1, 0.2 | Blend factor to best |
| elite_ratio | 0.1, 0.2, 0.5 | Fraction kept in ARS |
| fitness_shaping | rank, raw, softmax | How to weight perturbations |
| mirrrored_sampling | True, False | Half perturbations mirrored |

### 12.3 Ablation Studies

- **Remove ES optimizer entirely**: Pure Hebbian learning baseline.
- **Replace with SGD/Adam**: Backpropagation baseline (requires gradients).
- **Disable best-weight tracking**: Always blend, no elite.
- **Population of 2 vs. 128**: Sample efficiency curve.
- **Different noise distributions**: Gaussian vs. Cauchy vs. uniform.
- **Weight decay in ES**: Add regularization to fitness.

### 12.4 Optimization Tricks

- **Mirrored sampling**: For each perturbation, also try its negative. Reduces variance by 2x.
- **Fitness shaping**: Rank-transform fitness before weighting. Robust to outliers.
- **Anti-ambiguity trick**: Average over forward pass seeds. More reliable fitness.
- **Virtual batch normalization**: Stabilize population fitness evaluations.
- **Subspace ES**: Only perturb a random subspace each iteration. Scales to higher dimensions.
- **Compressed ES**: Use random projection to lower dimension before perturbing.
- **Asynchronous population training**: Evaluate population members in parallel.

### 12.5 Research Papers

- Salimans et al. "Evolution Strategies as a Scalable Alternative to Reinforcement Learning" (OpenAI-ES, 2017)
- Mania et al. "Simple random search provides a competitive approach to reinforcement learning" (ARS, 2018)
- Hansen "The CMA Evolution Strategy: A Tutorial" (CMA-ES, 2016)
- Sehnke et al. "Parameter-exploring Policy Gradients" (PGPE, 2010)
- Maheswaranathan et al. "Guided Evolution Strategies for Black-Box Optimization" (2018)
- Vicol et al. "Persistent Evolution Strategies" (2021)
- Lehman et al. "Safe Mutations for Deep and Recurrent Neural Networks through Output Gradients" (2017)

### 12.6 Failure Modes

| Failure | Symptom | Fix |
|---------|---------|-----|
| High variance | Fitness scores randomly jump | Increase population, mirrored sampling |
| Premature convergence | Population stuck at local optimum | Increase sigma, add restart |
| Fitness plateaus | All members same fitness | Add diversity reward |
| Computation cost | 8x forward passes per step | Reduce population, use async |
| Destructive updates | ES overwrites good Hebbian weights | Reduce ES update alpha |
| Memory explosion | Storing all perturbations | Subspace ES |

---

## 13. SkillModule

### 13.1 Variations / Alternatives

1. **Mixture of Experts (MoE) (Shazeer et al. 2017)**: Each expert is an MLP. Router selects top-2 experts. Sparsely activated. Load balancing loss.
2. **ST-MoE (Zoph et al. 2022)**: Stable MoE. Expert capacity, auxiliary loss, z-loss for stability.
3. **Soft MoE (Puigcerver et al. 2023)**: Each token is a weighted average of ALL experts. No discrete routing.
4. **Competitive Learning Skills**: Skills compete via WTA. Only winning skill's weights update.
5. **Skill Chaining**: Skills can call other skills. Hierarchical skill composition.
6. **ProtoNet-style Skills (Snell et al. 2017)**: Skill prototypes as class centroids. Few-shot skill adaptation.
7. **Skill Modulation via Fast Weights**: Skill transform = outer product of fast/slow weights. Like Hypernetwork.
8. **Task-vector Skills**: Each skill = a direction in weight space. Add skill vector to base weights (Ilharco et al. 2022).
9. **Gated Linear Skill Transform**: `skill(x) = sigmoid(W_gate @ x) * (W_transform @ x)`.
10. **Skill Embedding + Attention**: Embed skills, attend to them dynamically per position (not just per batch).
11. **Continuous Skill Space**: Instead of discrete n_skills, a continuous latent skill vector. Interpolate between skills.
12. **Skill Discovery via Intrinsic Motivation**: Skills discovered automatically. Curiosity-driven emergence.

### 13.2 Hyperparameter Sweeps

| Parameter | Range | Notes |
|-----------|-------|-------|
| n_skills | 2, 4, 8, 16, 32 | Number of specialized sub-modules |
| d_skill_prototype | d/4, d/2, d | Prototype dimension |
| lr skills | 0.001, 0.005, 0.01 | Skill Hebbian learning |
| transform_init | 0.001, 0.01, 0.1 | Initial transform magnitude |
| usage_decay | 0.8, 0.9, 0.95 | Skill usage EMA |
| sparsity (skill activation) | 1, 2, all | How many skills active per token |

### 13.3 Ablation Studies

- **Remove SkillModule entirely**: No skill specialization.
- **Single skill**: n_skills=1. Equivalent to additional transform.
- **Remove transforms, keep prototypes**: Skills only gate without transform.
- **Remove Hebbian update**: Fixed random skills.
- **Vary skill capacity**: What happens with 32 skills vs. 4?

### 13.4 Optimization Tricks

- **Load balancing loss**: Encourage equal usage across skills.
- **Skill pruning**: Remove unused skills, add new ones.
- **Skill reset**: If skill not used for N steps, reinitialize.
- **Skill prototype normalization**: Keep prototypes on unit sphere.
- **Skill merging**: Similar skills automatically merged.
- **Transform decomposition**: Low-rank `d_model * d_model = d_model * r + r * d_model`. Fewer params.

### 13.5 Research Papers

- Shazeer et al. "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" (2017)
- Zoph et al. "Designing Effective Sparse Expert Models" (ST-MoE, 2022)
- Puigcerver et al. "Learning to Route with Soft Mixture of Experts" (2023)
- Snell et al. "Prototypical Networks for Few-shot Learning" (2017)
- Ilharco et al. "Editing Models with Task Arithmetic" (2022)
- Ebrahimi et al. "Adversarial Skill Networks: Generalize Better with Diverse Skills" (2021)
- Florensa et al. "Automatic Goal Generation for Reinforcement Learning Agents" (2018)

### 13.6 Failure Modes

| Failure | Symptom | Fix |
|---------|---------|-----|
| Skill collapse | One skill dominates all usage | Load balancing loss |
| Dead skills | Skills never activate | Better init, reinitialize unused |
| Skill interference | Skills cancel each other (learn opposite transforms) | Orthogonality regularization |
| Over-specialization | Skills can't generalize | Mix skills during training |
| Parameter explosion | n_skills * d_model^2 huge | Low-rank transforms |

---

## 14. MixedPrecisionManager

### 14.1 Variations / Alternatives

1. **BF16 (bfloat16)**: Google's brain float. Same exponent range as FP32. No loss scaling needed. Better than FP16.
2. **FP8 (E4M3 / E5M2)**: NVIDIA H100 support. 2x throughput over FP16. Requires careful scaling.
3. **INT8 Quantization**: Post-training quantization. 4x smaller weights. May lose accuracy.
4. **4-bit NormalFloat (QLoRA, Dettmers et al. 2023)**: Information-theoretically optimal 4-bit quantization. Double quantization.
5. **Dynamic Precision Scaling**: Use higher precision for weights with high variance, lower for saturated weights.
6. **Chunk-based Precision**: Different layers different precision. Bottom layers FP32, top layers FP16.
7. **Stochastic Rounding**: Random rounding during quantization. Unbiased expectation.
8. **Block-wise Quantization (GPTQ)**: Quantize weights in blocks. Post-training quantization aware.
9. **Per-tensor vs. Per-channel vs. Per-group quantization**: Finer granularity = more accurate.
10. **Loss-aware Quantization**: Quantize to minimize task loss, not MSE.

### 14.2 Hyperparameter Sweeps

| Parameter | Range | Notes |
|-----------|-------|-------|
| precision | FP32, FP16, BF16, FP8 | Default precision |
| scaler_enabled | True, False | Whether to use GradScaler |
| loss_scale_init | 2^16, 2^8, 2^32 | Initial scale factor |
| quantization_bits | 4, 8, 16 | For post-training |

### 14.3 Ablation Studies

- **FP32 only**: No mixed precision. Higher accuracy, slower.
- **FP16 all**: Full FP16 weights and activations.
- **Remove scaler**: No gradient scaling.
- **Compare BF16 vs FP16 vs FP32 accuracy**.

### 14.4 Optimization Tricks

- **Tensor core alignment**: Ensure dimensions are multiples of 8/16 for Tensor Core use.
- **Overlap compute and quantization**: Quantize next layer while computing current.
- **Smart casting**: Cast once, reuse in cache.
- **Per-layer precision profiling**: Find optimal precision per layer empirically.
- **Use `torch.autocast`**: Automatic mixed precision without manual management.

### 14.5 Research Papers

- Micikevicius et al. "Mixed Precision Training" (2017)
- Wang et al. "FP8 for Deep Learning" (2022)
- Dettmers et al. "QLoRA: Efficient Finetuning of Quantized Language Models" (2023)
- Frantar et al. "GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers" (2022)
- Jacob et al. "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (2017)

### 14.6 Failure Modes

| Failure | Symptom | Fix |
|---------|---------|-----|
| Underflow | Small gradients become 0 | Use BF16 instead of FP16 |
| Overflow | Large values become inf | Loss scaling |
| Accuracy drop | Model gets worse | Keep critical layers FP32 |
| NaN from mixed precision | Numerical instability | Check for inf/nan and fallback |

---

## 15. DynamicBatchSizer

### 15.1 Variations / Alternatives

1. **Gradient Accumulation**: Instead of halving batch, accumulate gradients over N micro-batches. More stable.
2. **Adaptive Sequence Length**: Vary sequence length per batch. Short for noisy data, long for coherent data.
3. **Batch + Sequence Co-optimization**: Grid search over (batch, seq) at runtime.
4. **Memory-aware Scheduling**: Track VRAM consumption per operation. Predict OOM before it happens.
5. **Profiling-based sizing**: Profile once, set optimal static size. Simpler.
6. **Safe backoff**: On OOM, reduce by half and retry without crashing.
7. **Bayesian optimization for batch size**: Model loss as function of batch size. Find optimum.
8. **Curriculum batch sizing**: Start small, increase as training progresses.
9. **Dynamic batch based on data complexity**: Complex sequences get smaller batches.
10. **Quantized batch sizing**: Always use power-of-2 for Tensor Core alignment.

### 15.2 Hyperparameter Sweeps

| Parameter | Range | Notes |
|-----------|-------|-------|
| initial_batch | 2, 4, 8, 16, 32 | Starting batch |
| initial_seq | 32, 64, 128, 256 | Starting seq length |
| max_vram_mb | 1000, 3000, 4500, 8000 | VRAM budget |
| growth_factor | 1.5, 2.0, 3.0 | How fast to increase |
| shrink_factor | 0.5, 0.75 | How fast to decrease |

### 15.3 Ablation Studies

- **Fixed batch/seq**: No dynamic sizing.
- **Only vary batch, fix seq**: Simpler.
- **Only vary seq, fix batch**: Simpler.
- **Remove OOM count logic**: Always allow increase.

### 15.4 Optimization Tricks

- **PyTorch memory snapshot**: Use `torch.cuda.memory_snapshot()` for fine-grained tracking.
- **Set `torch.backends.cudnn.benchmark = True`**: Optimize for static sizes.
- **Pre-allocate memory pool**: Reduce allocation overhead.
- **Use `torch.cuda.empty_cache()` strategically**: Between large size changes.
- **Memory estimation formula**: Empirical model of `batch * seq * dim * bytes_per_param`.

### 15.5 Research Papers

- Smith et al. "Don't Decay the Learning Rate, Increase the Batch Size" (2017)
- Masters & Luschi "Revisiting Small Batch Training for Deep Neural Networks" (2018)
- Keskar et al. "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima" (2016)
- Goyal et al. "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" (2017)

### 15.6 Failure Modes

| Failure | Symptom | Fix |
|---------|---------|-----|
| Thrashing | Batch/seq oscillates | Add hysteresis, slower adjustment |
| OOM despite sizing | VRAM allocation spike | More conservative threshold |
| Too conservative | Never uses full VRAM | Lower threshold |
| Performance cliff | Odd batch sizes slower | Round to multiples of 8/16 |

---

## 16. AsyncDataLoader

### 16.1 Variations / Alternatives

1. **`torch.utils.data.DataLoader` with `num_workers`**: Built-in multiprocessing data loading. Prefetch factor controllable.
2. **NVIDIA DALI**: GPU-accelerated data loading. Preprocessing on GPU. Much faster for large datasets.
3. **Memory-mapped dataset (`np.memmap`, `torch.from_dlpack`)**: Zero-copy data loading. No need for worker threads.
4. **WebDataset (tarball streaming)**: Stream large datasets from disk/network without index. Good for sharded data.
5. **Two-stage prefetch**: Separate queue for tokenization and batching.
6. **Priority-based loading**: Important (recently high-error) samples get priority.
7. **Data augmentation in async worker**: Apply noising, masking on CPU during prefetch.
8. **Streaming from remote**: HTTP / S3 streaming. No local storage needed.
9. **Multi-stream / interleaved**: Interleave multiple data sources (text, code, math).
10. **On-the-fly tokenization in worker**: Tokenize from raw text during load.

### 16.2 Hyperparameter Sweeps

| Parameter | Range | Notes |
|-----------|-------|-------|
| prefetch | 1, 2, 4, 8, 16 | How many batches pre-loaded |
| num_workers | 0, 1, 2, 4, 8 | CPU workers |
| queue_timeout | 0.5, 1.0, 2.0, 5.0 | Seconds before fallback |
| batch_size | 4, 8, 16, 32 | Async batch size |

### 16.3 Ablation Studies

- **Synchronous loading**: No async, compute and load interleaved.
- **Remove fallback**: If queue empty, wait (no sync generation).
- **Different queue types**: `queue.Queue` vs. `multiprocessing.Queue`.

### 16.4 Optimization Tricks

- **Pin memory**: Use `pin_memory=True` for faster GPU transfer.
- **Pre-allocate batch tensor**: Reuse same buffer.
- **Overlap CPU→GPU copy with compute**: Use CUDA streams.
- **Data shuffling at worker level**: Avoid index computation overhead.
- **Profiling data loading vs. compute**: Tune prefetch to hide latency.

### 16.5 Research Papers

- "torch.utils.data Documentation" — PyTorch DataLoader design
- NVIDIA DALI documentation
- Warden "Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition" (data loading best practices, 2018)

### 16.6 Failure Modes

| Failure | Symptom | Fix |
|---------|---------|-----|
| Queue deadlock | Worker and main process lock | Timeout + fallback |
| Memory bloat | Queue holds large batches | Reduce prefetch |
| Worker crash | Worker thread dies silently | Check exceptions, restart |
| Data poisoning | Corrupted batch | Add validation checksum |
| Thread-safety violation | Shared state corruption | Workers must be stateless |

---

## 17. CogModule Base

### 17.1 Variations / Alternatives

1. **Different Hebbian Rules**:
   - **Oja's Rule**: `dw = lr * (error * input - dw * state)`. Normalizes weights to unit norm. Prevents explosion.
   - **BCM Rule (Bienenstock-Cooper-Munro)**: `dw = lr * input * (output - threshold) * output`. Sliding threshold. Homeostatic plasticity.
   - **Krotov's Rule (Hopfield / Dense Associative Memory)**: `dw = lr * (input - weight * output)`. For modern Hopfield networks.
   - **ABC Rule (Adaptive Bayesian Covariance)**: Bayesian posterior update for weights.
   - **Subspace Hebbian**: Weight changes constrained to subspace of previous weights.
   - **Spike-Timing-Dependent Plasticity (STDP)**: `dw = lr * (pre_spike * post_spike - decay)`. Temporal Hebbian.
   - **Three-factor Hebbian**: `dw = lr * neuromodulator * pre * post`. Neuromodulation = attention/error signal.
   - **Heterosynaptic Plasticity**: Neighboring synapses also weaken/strengthen. Biological realism.

2. **Momentum Variants**: Nesterov, Adam-style (per-parameter adaptive), Polyak averaging.

3. **Adaptive LR per weight**: Each weight has own LR based on gradient variance (like Adagrad/Adam but Hebbian).

4. **Weight Normalization**: Replace raw weights with `g * v / ||v||`. Decouples magnitude and direction.

5. **Sparse Weight Updates**: Current threshold-based. Try Top-k percentage, random subsampling, Student-t threshold.

6. **EWC Variants**:
   - **Online EWC (Schwarz et al. 2018)**: Single Fisher estimate, updated online.
   - **SI (Synaptic Intelligence)**: Per-weight importance from path integral of gradients.
   - **MAS (Memory Aware Synapses)**: Importance from sensitivity of output to weight changes.
   - **Riemannian EWC**: EWC in natural gradient space.

7. **Meta-Plasticity Variants**:
   - **Self-Tuning Networks**: LR = sigmoid(network output). Fully learned meta-learning.
   - **Hypergradient Descent**: Differentiate through learning process. Second-order.
   - **MAML-style**: Meta-learn initialization such that one step of Hebbian works well.

### 17.2 Hyperparameter Sweeps

| Parameter | Range | Notes |
|-----------|-------|-------|
| base_lr | 0.001, 0.01, 0.05, 0.1, 0.2 | Across all modules |
| momentum | 0.0, 0.5, 0.9, 0.95, 0.99 | Momentum factor |
| max_weight | 1.0, 2.0, 3.0, 5.0, 10.0 | Global weight clamping |
| meta_lr_target_error | 0.1, 0.5, 1.0, 2.0 | Target for meta-LR |
| ewc_lambda | 0.0, 0.01, 0.1, 1.0 | EWC penalty strength |
| sparse_update_threshold | 0.01, 0.05, 0.1, 0.2, 0.5 | Fraction of weights updated |

### 17.3 Ablation Studies

- **Remove momentum**: Pure stochastic Hebbian.
- **Remove weight clamping**: Unlimited weight growth.
- **Remove meta-plasticity**: Fixed LR.
- **Remove EWC**: No consolidation.
- **Remove sparse updates**: All weights updated every step.
- **Replace NLMS with simple LMS**: No input power normalization.

### 17.4 Optimization Tricks

- **Weight decay with Hebbian**: Add `-lambda * w` to update.
- **Weight smoothing**: EMA of weights for inference (Polyak averaging).
- **Weight standard deviation tracking**: Detect when weights converge.
- **Periodic Fisher reset**: Recompute Fisher from scratch to avoid stale estimates.
- **Gradient whitening**: Decorrelate gradient dimensions (KFAC-style).

### 17.5 Research Papers

- Oja "Simplified neuron model as a principal component analyzer" (1982) — Oja's rule
- Bienenstock, Cooper, Munro "Theory for the development of neuron selectivity" (1982) — BCM
- Krotov & Hopfield "Dense Associative Memory for Pattern Recognition" (2016)
- Kirkpatrick et al. "Overcoming catastrophic forgetting in neural networks" (EWC, 2017)
- Zenke et al. "Continual Learning Through Synaptic Intelligence" (2017)
- Aljundi et al. "Memory Aware Synapses: Learning what (not) to forget" (MAS, 2018)
- Schwarz et al. "Progress & Compress: A scalable framework for continual learning" (Online EWC, 2018)
- Schaul et al. "No more pesky learning rates" (hypergradient, 2013)
- Finn et al. "Model-Agnostic Meta-Learning for Fast Adaptation" (MAML, 2017)

### 17.6 Failure Modes

| Failure | Symptom | Fix |
|---------|---------|-----|
| Weight explosion | Weights grow to max_weight | Tighter clamp, weight decay |
| Weight decay to zero | All weights = 0 | Remove decay, check LR |
| EWC over-consolidation | Model can't learn new things | Lower ewc_lambda |
| Meta-plasticity oscillation | LR bounces between min and max | Smoother adaptation |
| Momentum overshoot | Weights oscillate around optimal | Nesterov correction |
| Sparse update starvation | No weights meet threshold | Lower threshold adaptively |

---

## 18. CogLang Controller

### 18.1 Variations / Alternatives

1. **Forward pass with gradient checkpointing**: Trade memory for compute. Enable deeper stacks.
2. **Adaptive Compute Time (ACT)**: Each layer decides how many computation steps to take. Halting with accumulative probability.
3. **Universal Transformer-style**: Recurrent over layers. Same layer applied repeatedly with different timescale.
4. **Layer-skipping via router**: Learned router decides which layers to activate for each token. Conditional computation.
5. **Multi-scale temporal integration**: Forward at multiple timescales simultaneously (parallel streams).
6. **Memory-integrated forward**: Interleave memory retrieval during forward, not just at start.
7. **Bi-directional forward**: Process left-to-right and right-to-left. Combine via learned mixing.
8. **Causal masking**: In forward, prevent attention to future tokens. For autoregressive generation.
9. **Iterative refinement**: Multiple passes through the stack. Each pass refines prediction.
10. **Speculative decoding**: Use smaller model (shallower stack) for draft, full model for verification.

### 18.2 Hyperparameter Sweeps

| Parameter | Range | Notes |
|-----------|-------|-------|
| n_layers | 2, 4, 6, 8, 12, 18 | Stack depth |
| d_sparse | 1024, 2048, 4096, 8192 | Sparse expansion |
| d_state | 32, 64, 128, 256, 512 | State capacity |
| d_context | 64, 128, 256, 512 | Context dimension |
| max_seq_len | 256, 512, 1024, 2048, 8192 | Context window |
| memory_size | 16, 32, 64, 128, 256 | Episodic slots |

### 18.3 Ablation Studies

- **Remove context embedding entirely**: No position info.
- **Causal vs. bidirectional**: Compare autoregressive vs. BERT-style.
- **Single sample vs. batched**: Batch size effect.
- **Fixed random context vs. learned**: Context embedding quality.

### 18.4 Optimization Tricks

- **Compile forward pass with `torch.compile`**: Fuse operations, reduce Python overhead.
- **CUDA graphs for forward pass**: Capture static execution graph. ~2x speedup.
- **Layer fusion**: Combine consecutive linear layers into single kernel.
- **State caching across calls**: Don't recompute if input length extended.
- **Early exit on low error**: If prediction error is below threshold early in stack, skip remaining layers.

### 18.5 Research Papers

- Chen et al. "Learning to Learn without Gradient Descent by Gradient-Free Evolutionary Strategy" (2017) — related to ES
- Dehghani et al. "Universal Transformers" (2018)
- Graves "Adaptive Computation Time for Recurrent Neural Networks" (2016)
- Stern et al. "Blockwise Parallel Decoding for Deep Autoregressive Models" (2018)

### 18.6 Failure Modes

| Failure | Symptom | Fix |
|---------|---------|-----|
| State corruption across batches | States leak between unrelated batches | Reset states per batch |
| Memory retrieval mismatch | Target_dim != state dimension | Fix projection |
| Learn flag stuck | Always learning, never inference | Proper learn flag management |
| Device mismatch | Some modules on CPU, some on GPU | Ensure .to(device) is complete |

---

## 19. Cross-Cutting Concerns

### 19.1 Weight Initialization
- **Current**: Default PyTorch init (uniform).
- **Experiments**: Xavier uniform, Kaiming normal, orthogonal, spectral normalization, SIREN (sinusoidal).
- **Per-module init**: Different init for W_pred, W_error, W_gate, attention.

### 19.2 Activation Functions
- **Current**: tanh for error, sigmoid for gates.
- **Experiments**: GELU, Swish/SiLU, SwiGLU, GeGLU, ReGLU, PReLU, ELU, CELU, SELU, Mish.
- **GELU often best for transformer-like models**.

### 19.3 Normalization
- **Current**: LayerNorm in SparseEncoder only.
- **Experiments**: RMSNorm (faster), ScaleNorm (simpler), BatchNorm (across batch), GroupNorm, LayerNorm everywhere, Pre-LN vs. Post-LN.
- **Apply RMSNorm to states, errors, predictions**.

### 19.4 Residual / Skip Connections
- **Current**: HebbianAttention processes tanh(error) directly.
- **Experiments**: Add dense residual `output = x + attended`. Skip connections from early to late layers. Multi-scale skip (concat from multiple levels).

### 19.5 Positional Encoding
- **Current**: Learned context embedding.
- **Experiments**: Sinusoidal (Vaswani), RoPE (Rotary, Su et al. 2021), ALiBi (Press et al. 2021), T5 relative bias, XPOS (Sun et al. 2022), NoPE (no position).
- **RoPE recommended**: Relative position, better length generalization.

### 19.6 Regularization
- **Current**: Weight clamping + EWC.
- **Experiments**: Dropout (on states, attention, embeddings), DropConnect, Stochastic Depth (drop layers), Label Smoothing, Weight Decay, Spectral Normalization, Mixout (dropout for fine-tuning).

### 19.7 Optimization Schedule
- **Current**: Cosine annealing (coglang_evolve.py) + per-layer LR decay.
- **Experiments**: Warmup + cosine, OneCycleLR (Smith 2018), Warmup + constant, SGDR (restarts), Linear decay, Exponential decay, Cyclical LR.

### 19.8 Gradient-Free Optimization (Central)
- **Current**: ES + Hebbian separate.
- **Experiments**: Joint Hebbian + ES (Hebbian for local, ES for global), Alternating (Hebbian for N steps, ES for 1 step), Weighted combination.

### 19.9 Architecture Search
- **Experiments**: Neural Architecture Search (NAS), Differentiable NAS (DARTS), Random search over d_model, n_layers, d_sparse, d_state, n_skills, memory_size.
- **Use population-based training (PBT)** to evolve hyperparameters while training.

### 19.10 Loss Functions
- **Current**: Cross-entropy (between output and next token).
- **Experiments**: Label smoothing cross-entropy, Focal loss (focus on hard tokens), InfoNCE (contrastive), Reverse KL (for generation), Auxiliary losses: sparsity loss, diversity loss, consistency loss.

### 19.11 Evaluation Metrics
- **Current**: Loss, perplexity, generation quality (unique ratio, repeat score).
- **Experiments**: BLEU, ROUGE, perplexity on held-out, next-token accuracy, surprisal (per token), entropy of output distribution, calibration score (ECE), generation diversity (n-gram overlap), probing tasks (syntax, semantics).

### 19.12 Hardware Utilization
- **Current**: Basic CUDA, mixed precision.
- **Experiments**: Multi-GPU (model parallel), Tensor Parallelism, Pipeline Parallelism, CPU offload for memory, Flash Attention (fused kernel), Fused Layernorm, Fused Softmax.

### 19.13 Distillation & Compression
- **Variations**: Distill from a larger Hebbian model to smaller, Knowledge Distillation (Hinton 2015), Quantization-Aware Training, Pruning (remove low-magnitude weights), Lottery Ticket Hypothesis (find winning tickets).

### 19.14 Continual / Lifelong Learning
- **Variations**: Progress & Compress (Schwarz 2018), Generative Replay (Shin 2017), Dark Experience Replay (Buzzega 2020), Gradient Episodic Memory (Lopez-Paz 2017).
- **EWC is already an approach**. Combine with replay.

### 19.15 Biorhythms / Sleep Phases
- **Experiments**: Sleep phase: replay historical data at high LR to consolidate. Wake phase: learn new data with curiosity. REM-like: random noise replay for creativity. Slow-wave: prune weak connections.

### 19.16 Consciousness / Global Workspace Theory
- **Future**: Global Workspace (Baars 1988) where content in EpisodicMemory is "broadcast" to all modules. Attention as gateway to workspace.
- **Implement**: A global workspace buffer that the SelfModel can access. Information in workspace is available to all skills.

---

## Summary: Priority Experiments

| Priority | Experiment | Expected Impact |
|----------|-----------|-----------------|
| P0 | Replace top-k with entmax or soft top-k | Better gradients, smoother sparsity |
| P0 | Add residual connections throughout | Stable training, deeper models |
| P0 | RoPE positional encoding | Better length generalization |
| P0 | RMSNorm on states and predictions | Training stability |
| P1 | Replace EMA uncertainty with Bayesian | Better calibrated confidence |
| P1 | Implement Oja's rule (weight normalization) | Prevent weight explosion |
| P1 | Add load balancing loss to SkillModule | Better skill utilization |
| P1 | Add learning progress (not just error) to IntrinsicMotivation | More directed curiosity |
| P2 | Replace ES with CMA-ES or ARS | More sample-efficient evolution |
| P2 | Add LSH-based SparseAssociativeMemory from anima | Faster, more scalable memory |
| P2 | Implement Mamba-like SSM in PredictiveLayer | O(L) sequence modeling |
| P2 | Memory consolidation during "sleep" phases | Better long-term retention |
| P3 | Factorized embeddings (ALBERT-style) | Parameter reduction |
| P3 | GELU/SwiGLU activations | Modern activation functions |
| P3 | Multi-GPU / distributed training | Scale to larger models |
| P3 | Contrastive loss + cross-entropy | Better representations |

---

*End of Document — CogLang v3 Exhaustive Experiment Catalog*
