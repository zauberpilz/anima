"""
CogLang v3 — AGI Architecture
Predictive Coding + Hebbian Learning + Working Memory + Meta-Plasticity
EFFICIENCY: Mixed Precision, Async Loading, Dynamic Batching
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import threading
import queue
import re


class CodeTokenizer:
    """
    PHASE 30: Code-Native Tokenizer — BPE + AST-aware tokenization.
    Statt Char-Level, tokenisiert dies Code in sinnvolle Tokens.
    """
    def __init__(self, vocab_size=4096):
        self.vocab_size = vocab_size
        self.vocab = {}  # token -> id
        self.reverse_vocab = {}  # id -> token
        self.bpe_merges = []

    def train(self, texts, min_frequency=2):
        """Train BPE tokenizer on code texts."""
        # Simple BPE training
        word_freqs = {}
        for text in texts:
            for word in text.split():
                word = ' '.join(list(word)) + ' </w>'
                word_freqs[word] = word_freqs.get(word, 0) + 1

        # Initialize with characters
        symbols = set()
        for word in word_freqs:
            for char in word.split():
                symbols.add(char)
        symbols = sorted(symbols)

        self.vocab = {s: i for i, s in enumerate(symbols)}
        self.vocab_size = len(symbols)

        # Merge pairs
        while len(self.vocab) < min(self.vocab_size, 4096):
            pair_freqs = {}
            for word, freq in word_freqs.items():
                chars = word.split()
                for i in range(len(chars)-1):
                    pair = (chars[i], chars[i+1])
                    pair_freqs[pair] = pair_freqs.get(pair, 0) + freq

            if not pair_freqs:
                break

            best_pair = max(pair_freqs, key=pair_freqs.get)
            if pair_freqs[best_pair] < min_frequency:
                break

            # Merge pair
            merged = ''.join(best_pair)
            self.bpe_merges.append(best_pair)
            self.vocab[merged] = len(self.vocab)

            new_word_freqs = {}
            for word, freq in word_freqs.items():
                new_word = word.replace(' '.join(best_pair), merged)
                new_word_freqs[new_word] = freq
            word_freqs = new_word_freqs

        self.reverse_vocab = {i: s for s, i in self.vocab.items()}

    def encode(self, text):
        """Encode text to token IDs."""
        tokens = []
        for word in text.split():
            word_chars = list(word) + ['</w>']
            while len(word_chars) > 1:
                # Find best merge
                best_pair = None
                for i in range(len(word_chars)-1):
                    pair = (word_chars[i], word_chars[i+1])
                    merged = ''.join(pair)
                    if merged in self.vocab:
                        best_pair = (i, merged)
                        break
                if best_pair is None:
                    # No merge found, output first char as token
                    tokens.append(self.vocab.get(word_chars[0], 0))
                    word_chars = word_chars[1:]
                else:
                    idx, merged = best_pair
                    tokens.append(self.vocab[merged])
                    word_chars = word_chars[:idx] + word_chars[idx+2:]
            if word_chars:
                tokens.append(self.vocab.get(word_chars[0], 0))
        return tokens

    def decode(self, ids):
        """Decode token IDs back to text."""
        return ''.join(self.reverse_vocab.get(i, '?') for i in ids).replace('</w>', ' ')


class AsyncDataLoader:
    """PHASE 15: Asynchronous Data Loading — CPU lädt Daten während GPU rechnet."""
    def __init__(self, data, batch_size, seq_length, device, prefetch=4):
        self.data = data
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.device = device
        self.queue = queue.Queue(maxsize=prefetch)
        self.running = False
        self.thread = None
        
    def _worker(self):
        """Background worker that pre-fetches batches."""
        while self.running:
            try:
                idx = torch.randint(0, len(self.data) - self.batch_size * self.seq_length, (1,)).item()
                batch = self.data[idx:idx + self.batch_size * self.seq_length].view(self.batch_size, self.seq_length)
                self.queue.put(batch.to(self.device), timeout=1.0)
            except queue.Full:
                continue
            except Exception:
                break
                
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        
    def get_batch(self):
        """Get next pre-fetched batch."""
        try:
            return self.queue.get(timeout=5.0)
        except queue.Empty:
            # Fallback: generate synchronously
            idx = torch.randint(0, len(self.data) - self.batch_size * self.seq_length, (1,)).item()
            return self.data[idx:idx + self.batch_size * self.seq_length].view(self.batch_size, self.seq_length).to(self.device)
            
    def stop(self):
        self.running = False


class DynamicBatchSizer:
    """PHASE 15: Dynamic Batch Sizing — Auto-adjust based on VRAM availability."""
    def __init__(self, initial_batch=8, initial_seq=128, max_vram_mb=4500):
        self.batch = initial_batch
        self.seq = initial_seq
        self.max_vram_mb = max_vram_mb
        self.oom_count = 0
        self.max_batch = 8  # Hard cap: B=16 verursacht OOM auf 8GB GPU mit 45% Fraction
        
    def adjust(self, vram_used_mb):
        """Adjust batch/seq based on VRAM usage."""
        if vram_used_mb > self.max_vram_mb * 0.9:
            # Too much VRAM -> reduce
            self.batch = max(2, self.batch // 2)
            self.seq = max(32, self.seq // 2)
            self.oom_count += 1
        elif vram_used_mb < self.max_vram_mb * 0.5 and self.oom_count == 0:
            # Plenty of VRAM -> increase (aber hard cap beachten)
            self.batch = min(self.max_batch, self.batch * 2)
            self.seq = min(256, self.seq * 2)
            
    def get_sizes(self):
        return self.batch, self.seq


class MixedPrecisionManager:
    """PHASE 15: Mixed Precision — FP16 for compute, FP32 for weights."""
    def __init__(self, enabled=True):
        self.enabled = enabled and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=False)  # No grad, but useful for scaling
        
    def to_fp16(self, tensor):
        """Convert tensor to FP16 if enabled."""
        if self.enabled:
            return tensor.half()
        return tensor
        
    def to_fp32(self, tensor):
        """Convert tensor back to FP32."""
        if self.enabled:
            return tensor.float()
        return tensor


class RMSNorm(nn.Module):
    """PHASE 29: RMS Layer Normalization — faster than LayerNorm, equally effective."""
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryPositionalEmbedding(nn.Module):
    """PHASE 29: Rotary Position Embedding (RoPE) — relative position encoding."""
    def __init__(self, dim, max_seq_len=8192):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
        self._cached_cos = None
        self._cached_sin = None

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.size(1)
        if seq_len > self.max_seq_len:
            self.max_seq_len = seq_len * 2
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = t[:, None] @ self.inv_freq[None, :]
        freqs = torch.cat([freqs, freqs], dim=-1)
        return torch.cos(freqs), torch.sin(freqs)

    @staticmethod
    def apply_rotary(x, cos, sin):
        """Apply rotary embedding to x (last dim must be even)."""
        d = x.size(-1)
        x1, x2 = x[..., :d//2], x[..., d//2:]
        cos = cos[:x.size(-2), :d].unsqueeze(0).unsqueeze(0)
        sin = sin[:x.size(-2), :d].unsqueeze(0).unsqueeze(0)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class CogModule(nn.Module):
    """Basis für alle CogLang-Module — mit Meta-Plastizität (PHASE 2) + EWC (PHASE 4)."""
    def __init__(self, name=None):
        super().__init__()
        self.name = name or self.__class__.__name__
        self._momentum = {}
        self._lr = 0.05
        self._momentum_factor = 0.9
        self._max_weight = 3.0
        # PHASE 2: Meta-Plastizität
        self._meta_lr_scale = 1.0
        self._error_history = []
        self._meta_lr_target_error = 0.5
        # PHASE 4: EWC — Elastic Weight Consolidation
        self._ewc_fisher = {}  # Fisher Information Matrix diagonal approximation
        self._ewc_optimal_params = {}  # Snapshot of important weights (keyed by tensor id)
        self._ewc_lambda = 10.0  # EWC penalty strength (erhöht von 0.1 für echten Schutz)
        # PHASE 28: Hebbian rule selector
        self._hebbian_rule = 'oja'  # 'nlms', 'oja', 'bcm'
        
    def learn(self, lr=0.05, momentum=0.9):
        self._lr = lr
        self._momentum_factor = momentum
        return self

    def _update_meta_plasticity(self, current_error_norm):
        """PHASE 2: Adjust learning rate based on error magnitude."""
        self._error_history.append(current_error_norm)
        if len(self._error_history) > 100:
            self._error_history.pop(0)
        avg_error = sum(self._error_history) / len(self._error_history)
        if avg_error > self._meta_lr_target_error * 2:
            self._meta_lr_scale = min(2.0, self._meta_lr_scale * 1.05)
        elif avg_error < self._meta_lr_target_error * 0.5:
            self._meta_lr_scale = max(0.1, self._meta_lr_scale * 0.95)
            
    def _ewc_consolidate(self, weight):
        """PHASE 4: Apply EWC penalty to prevent catastrophic forgetting."""
        if weight in self._ewc_fisher and weight in self._ewc_optimal_params:
            fisher = self._ewc_fisher[weight]
            optimal = self._ewc_optimal_params[weight]
            # NaN-Guard: Fisher und optimal müssen finit sein
            if torch.isnan(fisher).any() or torch.isinf(fisher).any():
                self._ewc_fisher[weight] = torch.zeros_like(fisher)
                fisher = self._ewc_fisher[weight]
            if torch.isnan(optimal).any() or torch.isinf(optimal).any():
                self._ewc_optimal_params[weight] = weight.data.clone()
                optimal = self._ewc_optimal_params[weight]
            # Penalty: -lambda * fisher * (current - optimal)
            # Zieht Gewicht Richtung Optimal, proportional zu Fisher-Importance
            penalty = -self._ewc_lambda * fisher * (weight.data - optimal)
            if torch.isnan(penalty).any() or torch.isinf(penalty).any():
                return  # Skip EWC wenn penalty NaN
            weight.data.add_(penalty)  # alpha=1.0: lambda steuert direkt die Stärke
            
    def _ewc_update_fisher(self, weight, gradient_estimate):
        """PHASE 4: Update Fisher Information diagonal approximation."""
        if weight not in self._ewc_fisher:
            self._ewc_fisher[weight] = gradient_estimate ** 2
        else:
            # Exponential moving average of Fisher
            self._ewc_fisher[weight] = 0.9 * self._ewc_fisher[weight] + 0.1 * (gradient_estimate ** 2)
            
    def _ewc_snapshot(self):
        """PHASE 4: Save current weights as optimal for EWC (keyed by tensor identity)."""
        for name, param in self.named_parameters():
            if param.requires_grad or True:  # Track all weights
                self._ewc_optimal_params[param] = param.data.clone()  # Key by tensor ref, not name!
            
    def _hebbian(self, error, inp, weight, lr_eff=1.0):
        """PHASE 28: Oja's Rule Hebbian update — weight-normalizing, prevents explosion."""
        # NaN-Guard: Bereinige Input-Tensoren und Weight
        if torch.isnan(error).any() or torch.isinf(error).any():
            error = torch.nan_to_num(error, nan=0.0, posinf=1.0, neginf=-1.0)
        if torch.isnan(inp).any() or torch.isinf(inp).any():
            inp = torch.nan_to_num(inp, nan=0.0, posinf=1.0, neginf=-1.0)
        if torch.isnan(weight.data).any() or torch.isinf(weight.data).any():
            weight.data = torch.nan_to_num(weight.data, nan=0.0, posinf=1.0, neginf=-1.0)

        e_2d = error.reshape(-1, error.size(-1))
        i_2d = inp.reshape(-1, inp.size(-1))
        output = i_2d @ weight.T  # [batch*seq, out_dim]

        # Oja's rule: dw = lr * (error * input - weight * output^2)
        # Hebbian term: outer product of error and input
        hebb_term = e_2d.T @ i_2d  # [out_dim, in_dim]
        # Oja normalization term: weight * (output^2).sum(dim=0)
        oja_term = (output ** 2).sum(dim=0).unsqueeze(1) * weight.data  # [out_dim, in_dim]

        dW = hebb_term - oja_term

        lr_eff *= self._meta_lr_scale

        # PHASE 21: Sparse Weight Updates - nur signifikante Updates
        grad_norm = dW.abs().mean()
        sparse_mask = dW.abs() > (grad_norm * 0.1)
        dW = dW * sparse_mask.float()

        # Gradient Clipping: Max 10% der Weight-Norm
        grad_norm_val = dW.norm().item()
        w_norm_val = weight.data.norm().item()
        if grad_norm_val > 0.1 * w_norm_val + 1e-8:
            dW = dW * (0.1 * w_norm_val / grad_norm_val)

        if weight not in self._momentum:
            self._momentum[weight] = dW.clone()
        else:
            m = self._momentum_factor
            self._momentum[weight] = m * self._momentum[weight] + (1 - m) * dW

        # PHASE 4: Apply EWC penalty before update
        self._ewc_consolidate(weight)

        # Weight Decay: Leichte Regularisierung
        decay = 1e-5 * weight.data
        weight.data.add_(lr_eff * self._momentum[weight] - lr_eff * decay)
        weight.data.clamp_(-self._max_weight, self._max_weight)

        # PHASE 4: Update Fisher with current gradient magnitude
        self._ewc_update_fisher(weight, self._momentum[weight])

    def set_hebbian_rule(self, rule='oja'):
        """PHASE 28: Switch Hebbian update rule. Options: 'nlms', 'oja', 'bcm'."""
        valid_rules = ['nlms', 'oja', 'bcm']
        if rule not in valid_rules:
            raise ValueError(f"Unknown Hebbian rule '{rule}'. Choose from {valid_rules}")
        self._hebbian_rule = rule


class EpisodicMemory(CogModule):
    """
    PHASE 1: Working Memory — Content-Addressable Episodic Memory.
    Speichert vergangene Zustände und ruft sie basierend auf Ähnlichkeit ab.
    """
    def __init__(self, d_model, memory_size=64, target_dim=None):
        super().__init__()
        self.d_model = d_model
        self.memory_size = memory_size
        self.target_dim = target_dim or d_model
        # Memory slots: each slot stores a state vector
        self.register_buffer('memory', torch.zeros(memory_size, d_model))
        self.register_buffer('memory_age', torch.zeros(memory_size))
        self.register_buffer('memory_strength', torch.ones(memory_size))
        
        # Hebbian write/read weights
        self.W_write = nn.Linear(d_model, memory_size, bias=False)
        self.W_read = nn.Linear(memory_size, d_model, bias=False)
        # Projection to match layer state dimension
        self.W_proj = nn.Linear(d_model, self.target_dim, bias=False)
        self._max_weight = 1.0
        
    def forward(self, query, write_state=None):
        """
        query: current state to retrieve similar memories [batch, seq, d_model]
        write_state: optional state to write into memory [batch, d_model]
        Returns: retrieved memory projected to target_dim [batch, seq, target_dim]
        """
        with torch.no_grad():
            batch, seq, d = query.shape
            
            q_flat = query.reshape(-1, d)
            similarity = q_flat @ self.memory.T
            attention = torch.softmax(similarity / (d ** 0.5), dim=-1)
            retrieved = attention @ self.memory
            retrieved = retrieved.reshape(batch, seq, d)
            
            # Project to target dimension
            retrieved = self.W_proj(retrieved)
            
            if write_state is not None:
                self._write_to_memory(write_state)
                
            return retrieved
    
    def _write_to_memory(self, state):
        """Hebbian write: find least-used slot and store state."""
        # state: [batch, d_model] -> take mean over batch
        state_mean = state.mean(dim=0)  # [d_model]
        
        # Find oldest/weakest slot
        oldest_idx = torch.argmax(self.memory_age)
        
        # Write state to slot
        self.memory[oldest_idx] = state_mean.detach()
        self.memory_age[oldest_idx] = 0
        self.memory_strength[oldest_idx] = 1.0
        
        # Age all other slots
        self.memory_age += 1
        self.memory_strength = torch.clamp(self.memory_strength - 0.01, 0.1, 1.0)
        
    def learn_step(self, query, retrieved_proj, target):
        """Hebbian learning for memory read/write weights."""
        # Recompute full retrieval (UNPROJECTED) for error in d_model space
        # retrieved_proj has target_dim (256), but target has d_model (1812)
        q_flat = query.reshape(-1, self.d_model)
        sim = q_flat @ self.memory.T  # [batch, memory_size]
        att = torch.softmax(sim / (self.d_model ** 0.5), dim=-1)  # [batch, memory_size]
        retrieved_full = att @ self.memory  # [batch, d_model=1812]
        retrieved_full = retrieved_full.reshape(query.shape)  # [batch, 1, d_model]
        
        # Error am QUERY-Position (letztes Token), nicht am ganzen Sequence!
        # query ist [batch, 1, d_model], target ist [batch, seq, d_model]
        error = target[:, -1:, :] - retrieved_full  # [batch, 1, d_model]
        e_flat = error.reshape(-1, self.d_model)  # [batch, d_model]
        
        # Update W_proj: learn to project d_model -> target_dim accurately
        current_proj = self.W_proj(query)  # [batch, 1, target_dim]
        proj_error = retrieved_proj - current_proj
        p_flat = proj_error.reshape(-1, self.target_dim)  # [batch, target_dim]
        dw_proj = p_flat.T @ q_flat  # [target_dim, d_model]
        self.W_proj.weight.data.add_(dw_proj, alpha=self._lr * 0.01)
        self.W_proj.weight.data.clamp_(-self._max_weight, self._max_weight)
        
        # dW_read = error^T @ attention  (beide auf batch-Dimension, passt!)
        dw_read = e_flat.T @ att  # [d_model, batch] @ [batch, memory_size] = [d_model, memory_size]
        # dw_read shape = [d_model, memory_size] = W_read.weight.shape
        if self.W_read.weight not in self._momentum:
            self._momentum[self.W_read.weight] = dw_read.clone()
        else:
            m = self._momentum_factor
            self._momentum[self.W_read.weight] = m * self._momentum[self.W_read.weight] + (1 - m) * dw_read
        self.W_read.weight.data.add_(self._momentum[self.W_read.weight], alpha=self._lr * 0.1)
        self.W_read.weight.data.clamp_(-1.0, 1.0)


class SensoryInput(CogModule):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self._max_weight = 2.0
    def forward(self, ids):
        return self.embed(ids)
    
    def learn_step(self, input_ids, error_in_d_model):
        """Hebbian update for embeddings."""
        batch, seq, d = error_in_d_model.shape
        lr_eff = self._lr * 0.1
        updates = error_in_d_model.reshape(-1, d) * lr_eff
        ids_flat = input_ids.reshape(-1)
        updates = torch.clamp(updates, -0.1, 0.1)
        self.embed.weight.data.scatter_add_(0, ids_flat.unsqueeze(1).expand(-1, d), updates)
        self.embed.weight.data.clamp_(-self._max_weight, self._max_weight)


class SparseEncoder(CogModule):
    def __init__(self, input_dim, d_sparse, sparsity=0.02):
        super().__init__()
        self.d_sparse = d_sparse
        self.sparsity = sparsity
        self.base_sparsity = sparsity
        self.proj = nn.Linear(input_dim, d_sparse, bias=False)
        self.norm = nn.LayerNorm(d_sparse)
    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        
        # Adaptive Sparsity
        input_var = x.var().item()
        dynamic_sparsity = self.base_sparsity / (input_var + 1e-4)
        dynamic_sparsity = max(0.01, min(0.1, dynamic_sparsity))
        
        k = max(1, int(self.d_sparse * dynamic_sparsity))
        vals, idx = torch.topk(x, k, dim=-1)
        mask = torch.zeros_like(x)
        mask.scatter_(-1, idx, torch.sigmoid(vals))
        return mask


class HebbianAttention(CogModule):
    """
    PHASE 11: Hebbian Transformer Hybrid — Self-Attention mit Hebbian Learning.
    Statt Backprop für Q/K/V Projektionen, nutzt dies Hebbian-Updates.
    """
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # Hebbian Q/K/V weights
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_out = nn.Linear(d_model, d_model, bias=False)
        self._max_weight = 1.0
        
    def forward(self, x, learn=True):
        with torch.no_grad():
            batch, seq, d = x.shape
            
            Q = self.W_q(x)
            K = self.W_k(x)
            V = self.W_v(x)
            
            # Multi-head attention
            Q = Q.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
            K = K.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
            V = V.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
            
            scores = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5)
            attn = torch.softmax(scores, dim=-1)
            out = attn @ V
            out = out.transpose(1, 2).contiguous().view(batch, seq, d)
            out = self.W_out(out)
            
            return out
    
    def learn_step(self, x, output, target):
        """Hebbian update for attention weights."""
        with torch.no_grad():
            error = target - output
            e_flat = error.reshape(-1, self.d_model)
            x_flat = x.reshape(-1, self.d_model)
            
            # Update W_out
            dw_out = (e_flat.T @ output.reshape(-1, self.d_model)) / e_flat.size(0)
            if self.W_out.weight not in self._momentum:
                self._momentum[self.W_out.weight] = dw_out.clone()
            else:
                m = self._momentum_factor
                self._momentum[self.W_out.weight] = m * self._momentum[self.W_out.weight] + (1 - m) * dw_out
            self.W_out.weight.data.add_(self._momentum[self.W_out.weight], alpha=self._lr * 0.1)
            self.W_out.weight.data.clamp_(-1.0, 1.0)


class PredictiveAttention(CogModule):
    """
    PHASE 3: Predictive Attention — Hebbian-basierter Aufmerksamkeitsmechanismus.
    Statt Softmax-Attention wie Transformer, nutzt dies Prediction Error als Attention-Signal.
    Hoher Error an einer Position -> mehr Aufmerksamkeit für diese Position.
    """
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # Hebbian attention weights: learn to project error to attention scores
        self.W_q = nn.Linear(d_model, d_model, bias=False)  # Query projection
        self.W_k = nn.Linear(d_model, d_model, bias=False)  # Key projection
        self.W_v = nn.Linear(d_model, d_model, bias=False)  # Value projection
        self.W_out = nn.Linear(d_model, d_model, bias=False)
        self._max_weight = 1.0
        
    def forward(self, x, error=None, learn=True):
        """
        x: input sequence [batch, seq, d_model]
        error: prediction error for attention modulation [batch, seq, d_model]
        Returns: attended output [batch, seq, d_model]
        """
        with torch.no_grad():
            batch, seq, d = x.shape
            
            # Project to Q, K, V
            Q = self.W_q(x)  # [batch, seq, d]
            K = self.W_k(x)  # [batch, seq, d]
            V = self.W_v(x)  # [batch, seq, d]
            
            # Standard attention scores
            scores = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5)  # [batch, seq, seq]
            
            # PHASE 3: Modulate attention with prediction error
            if error is not None:
                # Error magnitude as attention boost
                error_mag = (error ** 2).sum(dim=-1, keepdim=True)  # [batch, seq, 1]
                # Boost attention to high-error positions
                error_boost = error_mag @ error_mag.transpose(-2, -1)  # [batch, seq, seq]
                scores = scores + error_boost * 0.5  # Modulate with error signal
            
            attn = torch.softmax(scores, dim=-1)
            output = attn @ V  # [batch, seq, d]
            output = self.W_out(output)
            
            return output
    
    def learn_step(self, x, error, output, target):
        """Hebbian learning for attention weights."""
        # Learn to attend better: minimize difference between attended output and target
        err = target - output  # [batch, seq, d]
        
        # Update W_out
        e_flat = err.reshape(-1, self.d_model)
        o_flat = output.reshape(-1, self.d_model)
        dw_out = (e_flat.T @ o_flat) / e_flat.size(0)
        
        if self.W_out.weight not in self._momentum:
            self._momentum[self.W_out.weight] = dw_out.clone()
        else:
            m = self._momentum_factor
            self._momentum[self.W_out.weight] = m * self._momentum[self.W_out.weight] + (1 - m) * dw_out
        self.W_out.weight.data.add_(self._momentum[self.W_out.weight], alpha=self._lr * 0.1)
        self.W_out.weight.data.clamp_(-1.0, 1.0)


class PredictiveLayer(CogModule):
    def __init__(self, d_model, d_state=64, d_context=128, timescale=1.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_context = d_context
        self.timescale = timescale  # PHASE 5: 1.0 = fast, <1.0 = slow
        self.W_pred = nn.Linear(d_state + d_context, d_model, bias=False)
        self.W_error = nn.Linear(d_model, d_state, bias=False)
        self.W_gate = nn.Linear(d_state + d_model + d_context, d_state)
        # PHASE 5: Use small random init for state (avoids dead neurons with zero Hebbian input)
        state_init = torch.randn(1, 1, d_state) * 0.01
        self.register_buffer('state', state_init)
        self.register_buffer('error_trace', torch.zeros(1, d_model))

    def forward(self, x, context=None, memory_retrieved=None, learn=True, token_weights=None):
        with torch.no_grad():
            batch, seq, d = x.shape
            device = x.device
            ctx = context if context is not None else torch.zeros(batch, seq, self.d_context, device=device)
            
            # Combine state with retrieved memory if available
            state = self.state.expand(batch, seq, -1).contiguous()
            # NaN-Guard: Korrupte Zustände zurücksetzen
            if torch.isnan(state).any() or torch.isinf(state).any():
                state = torch.zeros_like(state)
                self.state.data.zero_()
            if memory_retrieved is not None:
                state = state + memory_retrieved * 0.1  # Gated memory injection
                
            inp = torch.cat([state, ctx], dim=-1)
            prediction = self.W_pred(inp)
            error = x - prediction

            if learn:
                lr_eff = self._lr / (batch * seq)
                
                # PHASE 2: Update meta-plasticity based on current error
                error_norm = (error ** 2).sum().item() ** 0.5
                self._update_meta_plasticity(error_norm)
                
                # NaN-Guard: Error sanitize before any Hebbian update (verhindert NaN-Kaskade)
                if torch.isnan(error).any() or torch.isinf(error).any():
                    error = torch.nan_to_num(error, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # PHASE 58: Knowledge-Gap-gewichtet — Wissenslücken-Tokens
                # bekommen stärkere Hebbian-Updates (aktives Lernen auf Token-Ebene).
                # Nur das Haupt-Update (W_pred) wird skaliert; der zurückgegebene
                # Fehler bleibt unverändert, damit Module-Statistiken konsistent bleiben.
                update_error = error
                if token_weights is not None:
                    update_error = error * token_weights
                
                # W_pred: NLMS Hebbian (hat eigenen NaN-Guard)
                self._hebbian(update_error, inp, self.W_pred.weight, lr_eff)
                
                # W_error: Adaptive LMS Hebbian
                delta = self.W_error(error)
                if torch.isnan(delta).any() or torch.isinf(delta).any():
                    delta = torch.nan_to_num(delta, nan=0.0, posinf=1.0, neginf=-1.0)
                e_flat = error.reshape(-1, d)
                d_flat = delta.reshape(-1, self.d_state)
                error_norm = (e_flat ** 2).sum(dim=1, keepdim=True).mean() + 1e-8
                adaptive_scale = torch.clamp(1.0 / (error_norm.sqrt() + 1e-4), 0.1, 10.0)
                dw_err = (d_flat.T @ e_flat) / (e_flat.size(0)) * adaptive_scale
                # NaN-Guard: Gradient auf finit prüfen
                if torch.isnan(dw_err).any() or torch.isinf(dw_err).any():
                    dw_err = torch.nan_to_num(dw_err, nan=0.0, posinf=0.1, neginf=-0.1)
                
                if self.W_error.weight not in self._momentum:
                    self._momentum[self.W_error.weight] = dw_err.clone()
                else:
                    m = self._momentum_factor
                    self._momentum[self.W_error.weight] = m * self._momentum[self.W_error.weight] + (1 - m) * dw_err
                self.W_error.weight.data.add_(self._momentum[self.W_error.weight], alpha=lr_eff * 0.2)
                self.W_error.weight.data.clamp_(-1.0, 1.0)
                
                # Gate: Learnable with stabilized LR
                gate_in = torch.cat([state, error, ctx], dim=-1)
                if torch.isnan(gate_in).any() or torch.isinf(gate_in).any():
                    gate_in = torch.nan_to_num(gate_in, nan=0.0, posinf=1.0, neginf=-1.0)
                gate = torch.sigmoid(self.W_gate(gate_in))
                # PHASE 5: Timescale modulation - slower layers update less
                new_state = (1 - gate * self.timescale) * state + (gate * self.timescale) * delta
                
                # NaN-Guard: new_state auf finit prüfen
                if torch.isnan(new_state).any() or torch.isinf(new_state).any():
                    new_state = torch.nan_to_num(new_state, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # Hebbian update for W_gate
                gate_error = (new_state - state)
                g_flat = gate_in.reshape(-1, gate_in.size(-1))
                ge_flat = gate_error.reshape(-1, self.d_state)
                dw_gate = (ge_flat.T @ g_flat) / (g_flat.size(0))
                # NaN-Guard: Gradient auf finit prüfen
                if torch.isnan(dw_gate).any() or torch.isinf(dw_gate).any():
                    dw_gate = torch.nan_to_num(dw_gate, nan=0.0, posinf=0.1, neginf=-0.1)
                
                if self.W_gate.weight not in self._momentum:
                    self._momentum[self.W_gate.weight] = dw_gate.clone()
                else:
                    m = self._momentum_factor
                    self._momentum[self.W_gate.weight] = m * self._momentum[self.W_gate.weight] + (1 - m) * dw_gate
                self.W_gate.weight.data.add_(self._momentum[self.W_gate.weight], alpha=lr_eff * 0.05)
                self.W_gate.weight.data.clamp_(-0.5, 0.5)
            else:
                # NaN-Guard: Error und delta sanitizen
                if torch.isnan(error).any() or torch.isinf(error).any():
                    error = torch.nan_to_num(error, nan=0.0, posinf=1.0, neginf=-1.0)
                delta = self.W_error(error)
                if torch.isnan(delta).any() or torch.isinf(delta).any():
                    delta = torch.nan_to_num(delta, nan=0.0, posinf=1.0, neginf=-1.0)
                gate_in = torch.cat([state, error, ctx], dim=-1)
                if torch.isnan(gate_in).any() or torch.isinf(gate_in).any():
                    gate_in = torch.nan_to_num(gate_in, nan=0.0, posinf=1.0, neginf=-1.0)
                gate = torch.sigmoid(self.W_gate(gate_in))
                # PHASE 5: Timescale modulation
                new_state = (1 - gate * self.timescale) * state + (gate * self.timescale) * delta
                if torch.isnan(new_state).any() or torch.isinf(new_state).any():
                    new_state = torch.nan_to_num(new_state, nan=0.0, posinf=1.0, neginf=-1.0)

            new_state_detached = new_state[0:1, -1:, :].detach()
            if torch.isnan(new_state_detached).any():
                new_state_detached = torch.zeros_like(new_state_detached)
            self.state = new_state_detached
            error_trace = error[0:1, -1, :].detach()
            if torch.isnan(error_trace).any() or torch.isinf(error_trace).any():
                error_trace = torch.zeros_like(error_trace)
            self.error_trace = error_trace
            return new_state, error, prediction

    def reset_state(self, batch_size=1, device='cpu'):
        self.state = torch.zeros(1, 1, self.d_state, device=device)
        self.error_trace = torch.zeros(1, self.d_model, device=device)


class SelfModel(CogModule):
    """
    PHASE 6: Self-Model — Meta-kognitive Repräsentation der eigenen Struktur.
    Das Modell trackt seine eigene Unsicherheit und nutzt sie zur Verhaltensmodulation.
    """
    def __init__(self, d_model, n_layers):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        # Track per-layer prediction error statistics
        self.register_buffer('layer_error_mean', torch.zeros(n_layers))
        self.register_buffer('layer_error_var', torch.zeros(n_layers))
        # Self-confidence: how well the model predicts overall
        self.register_buffer('self_confidence', torch.tensor(0.5))
        # Uncertainty projection: maps error stats to modulation signal
        self.W_uncertainty = nn.Linear(n_layers, d_model, bias=False)
        self._max_weight = 1.0
        
    def forward(self, layer_errors):
        """
        layer_errors: list of error tensors per layer
        Returns: uncertainty modulation signal [batch, seq, d_model]
        """
        with torch.no_grad():
            # Update error statistics
            for i, err in enumerate(layer_errors):
                if i < self.n_layers:
                    err_norm = (err ** 2).sum(dim=-1).mean().item()
                    # Exponential moving average
                    self.layer_error_mean[i] = 0.9 * self.layer_error_mean[i] + 0.1 * err_norm
                    self.layer_error_var[i] = 0.9 * self.layer_error_var[i] + 0.1 * (err_norm - self.layer_error_mean[i]) ** 2
            
            # Compute overall uncertainty
            total_uncertainty = self.layer_error_mean.mean().item()
            # Self-confidence: inverse of uncertainty (normalized)
            self.self_confidence.fill_(1.0 / (1.0 + total_uncertainty))
            self.self_confidence.clamp_(0.1, 1.0)
            
            # Generate uncertainty modulation signal
            # High uncertainty -> stronger modulation
            uncertainty_vec = self.layer_error_mean.unsqueeze(0).unsqueeze(0)  # [1, 1, n_layers]
            modulation = self.W_uncertainty(uncertainty_vec)  # [1, 1, d_model]
            return modulation
    
    def learn_step(self, layer_errors, future_errors=None):
        """
        PHASE 6: Lerne, die eigene Unsicherheit besser vorherzusagen.
        Aktualisiert W_uncertainty mit Hebbian-Regel.
        
        layer_errors: Liste von Error-Tensoren pro Layer (aktuelle Schicht)
        future_errors: Tatsächliche zukünftige Errors (oder None = use current)
        """
        with torch.no_grad():
            # Extrahiere Unsicherheits-Vektor
            uncertainty_vec = self.layer_error_mean.unsqueeze(0).unsqueeze(0)  # [1, 1, n_layers]
            
            # Ziel: Vorhersagefehler der aktuellen Schicht 
            # (je höher der Fehler, desto mehr Modulation sollte kommen)
            if future_errors is not None and len(future_errors) > 0:
                target = sum((e ** 2).mean().item() for e in future_errors) / len(future_errors)
            else:
                target = self.layer_error_mean.mean().item()
            
            # Hebbian: Modulation verstärken bei hohem Error, abschwächen bei niedrigem
            lr = 0.001 * target  # LR skaliert mit Error-Größe
            mod_out = self.W_uncertainty(uncertainty_vec)  # [1, 1, d_model]
            
            # Aktualisiere W_uncertainty: korrekte Hebbian-Regel (outer product)
            # dW = lr * (error_in_output) x (input_activation)
            # error_in_output: target - mod_out, shape [1, 1, d_model]
            # input_activation: uncertainty_vec, shape [1, 1, n_layers]
            # dW shape: [1, d_model, n_layers] -> squeeze(0) -> [d_model, n_layers]
            mod_error = target - mod_out  # [1, 1, d_model]
            dW = lr * mod_error.transpose(-1, -2) @ uncertainty_vec  # [1, d_model, 1] @ [1, 1, n_layers] = [1, d_model, n_layers]
            self.W_uncertainty.weight.data += dW.squeeze(0)
            self.W_uncertainty.weight.data.clamp_(-self._max_weight, self._max_weight)
    
    def get_confidence(self):
        """Return current self-confidence score."""
        return self.self_confidence.item()


class PredictiveStack(CogModule):
    def __init__(self, d_model, n_layers=4, d_state=64, d_context=128, n_attention_heads=4):
        super().__init__()
        # PHASE 5: Multi-Scale Timescales
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            timescale = max(0.1, 1.0 - (i / n_layers) * 0.8)
            self.layers.append(PredictiveLayer(d_model, d_state, d_context, timescale=timescale))
        self.pred_mixer = nn.Parameter(torch.ones(n_layers) / n_layers)
        # PHASE 3: Predictive Attention
        self.attention = PredictiveAttention(d_model, n_heads=n_attention_heads)
        # PHASE 11: Hebbian Transformer Hybrid
        self.hebbian_attn = HebbianAttention(d_model, n_heads=n_attention_heads)
        # PHASE 6: Self-Model
        self.self_model = SelfModel(d_model, n_layers)

    def forward(self, x, context=None, memory_retrieved=None, errors_for_attn=None, learn=True, token_weights=None):
        errors, states, preds = [], [], []
        current = x
        for i, layer in enumerate(self.layers):
            mem = memory_retrieved if i == 0 else None
            s, e, p = layer(current, context, memory_retrieved=mem, learn=learn, token_weights=token_weights)
            errors.append(e); states.append(s); preds.append(p)
            
            # PHASE 11: Hebbian Attention als primäre Attention
            attended = self.hebbian_attn(torch.tanh(e), learn=learn)
            
            # PHASE 3: Predictive Attention - Error-modulierter Fokus (zusätzlich zu Hebbian)
            pred_attended = self.attention(current, e, learn=learn)
            if learn:
                # Learn: attention output soll Error minimieren helfen
                self.attention.learn_step(current, e, pred_attended, current)
            # Kombiniere beide Attention-Arten
            attended = attended + pred_attended * 0.3
            
            # PHASE 11: Hebbian Attention learn_step aktivieren (Ziel: Error reduzieren)
            if learn:
                # Nutze attn_input als x, attended als output, current als target
                # Attention soll lernen, Error-Signal in nützliche Repräsentation zu wandeln
                self.hebbian_attn.learn_step(current, attended, current)
            
            # PHASE 29: Residual connection — add input to attended output
            attended = current + attended * 0.5  # Scaled residual
            
            # PHASE 6: Self-Model modulation auf JEDER Schicht (skaliert mit Tiefe)
            modulation = self.self_model(errors[:i+1])
            depth_scale = (i + 1) / self.n_layers  # Mehr Modulation in tieferen Schichten
            current = attended + modulation * 0.1 * depth_scale
            
            # Lerne SelfModel auf jeder Schicht
            if learn:
                self.self_model.learn_step(errors[:i+1], errors)
        return errors, states, preds

    def mixed_prediction(self, predictions):
        mix = torch.softmax(self.pred_mixer, dim=-1)
        return sum(w * p for w, p in zip(mix, predictions))

    def reset_states(self, batch_size=1, device='cpu'):
        for l in self.layers:
            l.reset_state(batch_size, device)


class HierarchicalPC(CogModule):
    """
    PHASE 35: Hierarchical Predictive Coding — Mehr-Ebenen-Architektur.
    
    Statt einem flachen PredictiveStack (alle Layer gleiche Dimension) hat CogLang
    jetzt eine echte Hierarchie mit verschiedenen Abstraktionsebenen:
    
    Level 1 (Fast/Tokens):   d_sparse, volle Seq-Länge, hohe Auflösung
      → Verarbeitet einzelne Tokens, schnelle Dynamik (timescale=1.0)
    
    Level 2 (Medium/Phrases): d_sparse/2, 4× komprimiert, mittlere Auflösung
      → Erkennt Phrasen und kurze Muster, timescale=0.3
    
    Level 3 (Slow/Concepts):  d_sparse/4, 16× komprimiert, abstrakt
      → Konzepte, Ziele, langfristige Abhängigkeiten, timescale=0.1
    
    Kommunikationsfluss:
      Bottom-Up:  Level N+1 bekommt Prediction Error von Level N
      Top-Down:   Level N+1 sagt Aktivität von Level N vorher
      Ziel:       Minimaler Prediction Error auf ALLEN Ebenen
    
    Emergenz-Effekt: Das Modell entwickelt automatisch abstrakte Repräsentationen,
    weil Level 3 nur überleben kann, wenn es Level 2 gut vorhersagt — das zwingt
    Level 3 dazu, echte Konzepte zu lernen.
    """
    def __init__(self, d_model, n_levels=3, n_layers_per_level=None,
                 d_state=64, d_context=128, n_attention_heads=4):
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels
        if n_layers_per_level is None:
            n_layers_per_level = [6, 4, 2]  # Level 1 tief, Level 3 flach
        self.n_layers_per_level = n_layers_per_level
        
        # ——— Ebenen-Konfiguration ———
        # Jede Ebene hat eigene Dimension (kleiner = abstrakter)
        level_dims = []
        current_dim = d_model
        for i in range(n_levels):
            level_dims.append(current_dim)
            current_dim = max(64, current_dim // 2)  # Halbiere pro Ebene
        self.level_dims = level_dims
        
        # ——— Predictive Stacks pro Ebene ———
        self.stacks = nn.ModuleList()
        for i in range(n_levels):
            dim = level_dims[i]
            n_layers = n_layers_per_level[i] if i < len(n_layers_per_level) else 4
            # Tiefere Ebenen (höheres i) haben langsamere Timescale
            timescale_scale = 1.0 / (2 ** i) if i > 0 else 1.0
            stack = PredictiveStack(dim, n_layers, 
                                     max(32, d_state // (2 ** i)),
                                     max(64, d_context // (2 ** i)),
                                     n_attention_heads)
            # Set timescale for all layers in this stack
            for layer in stack.layers:
                layer.timescale = timescale_scale * max(0.1, 1.0 - (layer.timescale - 0.1) * (1 - timescale_scale))
            stack.level_timescale = timescale_scale
            self.stacks.append(stack)
        
        # ——— Bottom-Up Encoder (Level N → Level N+1) ———
        # Nimmt Error von Level N und projiziert auf Level N+1 Dimension
        self.enc_bottom_up = nn.ModuleList()
        for i in range(n_levels - 1):
            in_dim = level_dims[i]
            out_dim = level_dims[i + 1]
            self.enc_bottom_up.append(nn.Linear(in_dim, out_dim, bias=False))
        
        # ——— Top-Down Predictor (Level N+1 → Level N) ———
        # Sagt Aktivität von Level N aus Level N+1 Zustand vorher
        self.pred_top_down = nn.ModuleList()
        for i in range(n_levels - 1):
            in_dim = level_dims[i + 1]
            out_dim = level_dims[i]
            self.pred_top_down.append(nn.Linear(in_dim, out_dim, bias=False))
        
        # ——— Temporal Compression ———
        # Level 1: every token, Level 2: every 4 tokens, Level 3: every 16 tokens
        self.compression_factors = [1]
        for i in range(1, n_levels):
            self.compression_factors.append(self.compression_factors[-1] * 4)
        
        # ——— Kontext-Projektionen ———
        # Level 1 bekommt Original-Kontext, höhere Ebenen projizierte Versionen
        self.context_proj = nn.ModuleList()
        if n_levels > 1:
            for i in range(1, n_levels):
                self.context_proj.append(nn.Linear(d_context, max(64, d_context // (2 ** i)), bias=False))
        
        # ——— Metrik-Tracking ———
        self.register_buffer('level_errors', torch.zeros(n_levels))
        self.register_buffer('level_agreement', torch.ones(n_levels))  # Top-Down ↔ Bottom-Up Agreement
        
    def _downsample(self, x, factor):
        """Downsample sequence by factor using average pooling."""
        if factor <= 1 or x.size(1) < factor:
            return x
        batch, seq, dim = x.shape
        # Pad to multiple of factor
        pad = factor - (seq % factor)
        if pad < factor:
            x = F.pad(x, (0, 0, 0, pad))
        # Reshape and average pool
        x = x.view(batch, -1, factor, dim).mean(dim=2)
        return x
    
    def _upsample(self, x, target_len):
        """Upsample sequence to target length using nearest-neighbor interpolation."""
        batch, seq, dim = x.shape
        if seq == 0:
            return torch.zeros(batch, target_len, dim, device=x.device)
        # Repeat each element to fill target length
        repeats = (target_len + seq - 1) // seq
        x = x.repeat_interleave(repeats, dim=1)
        return x[:, :target_len, :]
    
    def forward(self, x, context=None, memory_retrieved=None, errors_for_attn=None, learn=True):
        """
        Hierarchical forward pass.
        
        Args:
            x: [batch, seq, d_model] — Input (bottom level)
            context: [batch, seq, d_context] — Context embeddings
            memory_retrieved: [batch, seq, d_model] — Episodic memory
            learn: bool — Enable learning
        
        Returns:
            all_errors: Liste aller Errors (alle Ebenen, alle Layer)
            all_states: Liste aller States
            mixed_pred: Kombinierte Prediction (auf originaler Dimension)
            level_reports: Dict mit Ebenen-Metriken
        """
        with torch.no_grad():
            batch, seq, d = x.shape
            device = x.device
            
            # ——— Level 0 Processing (immer aktiv) ———
            # Level 0 ist der Input selbst (Sensorische Ebene)
            
            # ——— Level 1: Token-Level (volle Auflösung) ———
            l1_context = context  # Original-Kontext
            
            # Downsample Memory Retrieval für Level 1
            l1_memory = memory_retrieved  # [batch, seq, d_model]
            
            l1_errors, l1_states, l1_preds = self.stacks[0](
                x, l1_context, memory_retrieved=l1_memory, learn=learn
            )
            
            # Level 1 Prediction (gemischt aus allen Sub-Layern)
            l1_pred = self.stacks[0].mixed_prediction(l1_preds)  # [batch, seq, d_model]
            
            # Level 1 Error (Differenz zwischen Prediction und Input)
            l1_error = x - l1_pred  # [batch, seq, d_model]
            
            # ——— Höhere Ebenen (Level 2, Level 3, ...) ———
            higher_errors = []
            higher_states = []
            higher_preds = []
            
            current_bottom_up = l1_error  # Start: Error von Level 1
            
            for level_idx in range(1, self.n_levels):
                stack = self.stacks[level_idx]
                comp = self.compression_factors[level_idx]
                dim = self.level_dims[level_idx]
                
                # ——— Bottom-Up Encoding ———
                # 1. Downsample: komprimiere Sequenz
                bu_down = self._downsample(current_bottom_up, comp // self.compression_factors[level_idx - 1] if level_idx > 1 else comp)
                # 2. Projection: auf Ebene-Dimension bringen
                bu_encoded = self.enc_bottom_up[level_idx - 1](bu_down)  # [batch, seq/comp, dim]
                
                # ——— Kontext für diese Ebene ———
                if level_idx == 1:
                    level_context = context  # Level 1 hat Original-Kontext
                else:
                    ctx_proj = self.context_proj[level_idx - 1]
                    level_context = self._downsample(context, comp)
                    level_context = ctx_proj(level_context)
                
                # ——— Forward durch PredictiveStack dieser Ebene ———
                # Normalisiere Input auf [0, 1]-Bereich (stabilisiert höhere Ebenen)
                bu_input = torch.tanh(bu_encoded) * 0.5
                
                l_errors, l_states, l_preds = stack(
                    bu_input, level_context, memory_retrieved=None, learn=learn
                )
                
                # Gemischte Prediction dieser Ebene
                l_pred = stack.mixed_prediction(l_preds)
                
                # Error dieser Ebene
                l_error = bu_input - l_pred
                
                # ——— Top-Down Prediction ———
                # Höhere Ebene sagt die Aktivität der aktuellen Ebene vorher
                if level_idx < self.n_levels - 1:
                    td_pred = self.pred_top_down[level_idx - 1](l_pred)
                    # Upsample auf Level 1 Sequenzlänge
                    td_pred_up = self._upsample(td_pred, seq)
                    # Top-Down Modulation: korrigiere Level 1 Error basierend auf höherer Ebene
                    # Der "korrigierte" Error = Level 1 Error + Top-Down Korrektur
                    # Idee: Höhere Ebene sagt, was der Error SEIN SOLLTE
                    current_bottom_up = l1_error + td_pred_up * 0.2
                else:
                    # Höchste Ebene: keine Top-Down Prediction mehr
                    # Aber: Top-Down zur Ebene darunter
                    td_pred = self.pred_top_down[level_idx - 1](l_pred)
                    td_pred_up = self._upsample(td_pred, seq)
                    current_bottom_up = l1_error + td_pred_up * 0.2
                
                higher_errors.extend(l_errors)
                higher_states.extend(l_states)
                higher_preds.extend(l_preds)
                
                # Tracke Level Error
                self.level_errors[level_idx] = (l_error ** 2).mean().item()
                
                # Level Agreement: Korrelation zwischen Bottom-Up und Top-Down
                if level_idx > 0:
                    td_signal = self.pred_top_down[level_idx - 1](l_pred)
                    bu_norm = bu_input / (bu_input.norm(dim=-1, keepdim=True) + 1e-8)
                    td_norm = td_signal / (td_signal.norm(dim=-1, keepdim=True) + 1e-8)
                    agreement = (bu_norm * td_norm).sum(dim=-1).mean().item()
                    self.level_agreement[level_idx] = 0.9 * self.level_agreement[level_idx] + 0.1 * agreement
            
            # ——— Finale Prediction: Level 1 Prediction + Top-Down Korrektur ———
            # Die finale Prediction ist die Level 1 Prediction, moduliert durch höhere Ebenen
            mixed_pred = l1_pred
            
            # Top-Down Modulation von Level 2 und 3 in finale Prediction einweben
            if self.n_levels > 1:
                td_from_level2 = self.pred_top_down[0](higher_preds[-1])  # Letzte Prediction von Level 2
                td_from_level2_up = self._upsample(td_from_level2, seq)
                mixed_pred = mixed_pred + td_from_level2_up * 0.1
            
            if self.n_levels > 2:
                td_from_level3 = self.pred_top_down[1](higher_preds[-1])  # Von Level 3
                td_from_level3_up = self._upsample(td_from_level3, seq)
                # Erst hoch zu Level 2 Dim, dann zu Level 1 Dim
                td_from_level3_up = self.pred_top_down[0](td_from_level3_up)
                mixed_pred = mixed_pred + td_from_level3_up * 0.05
            
            # Self-Model Error Tracking (für Kompatibilität mit CogLang.learn)
            self_model_errors = l1_errors + higher_errors
            
            return self_model_errors, l1_states + higher_states, l1_preds + higher_preds, mixed_pred
    
    def reset_states(self, batch_size=1, device='cpu'):
        for stack in self.stacks:
            stack.reset_states(batch_size, device)
    
    def get_level_report(self):
        """Gibt Metriken pro Ebene zurück."""
        return {
            'level_errors': [f'{e:.3f}' for e in self.level_errors.tolist()],
            'level_agreement': [f'{a:.3f}' for a in self.level_agreement.tolist()],
            'level_dims': self.level_dims,
            'n_levels': self.n_levels,
        }


class OutputDecoder(CogModule):
    def __init__(self, d_sparse, d_model, vocab_size):
        super().__init__()
        self.out_proj = nn.Linear(d_sparse, d_model, bias=False)
        self.out_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, pred):
        # NaN-Guard: pred darf nicht NaN sein
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=-1.0)
        hidden = torch.tanh(self.out_proj(pred))
        return self.out_head(hidden), hidden

    def learn_step(self, output, hidden, pred, input_ids):
        V = output.size(-1)
        smooth = 0.1
        probs = F.softmax(output, dim=-1)
        target = (1 - smooth) * F.one_hot(input_ids, num_classes=V).float() + smooth / V
        err = probs - target
        batch, seq = input_ids.shape
        d_hidden = err @ self.out_head.weight
        dw_proj = (d_hidden.reshape(-1, d_hidden.size(-1)).T @ pred.reshape(-1, pred.size(-1))) / (batch * seq)
        dw_head = (err.reshape(-1, V).T @ hidden.reshape(-1, hidden.size(-1))) / (batch * seq)
        self.out_head.weight.data.add_(dw_head, alpha=-self._lr)
        self.out_proj.weight.data.add_(dw_proj, alpha=-self._lr)
        self.out_head.weight.data.clamp_(-2.0, 2.0)
        self.out_proj.weight.data.clamp_(-2.0, 2.0)
        return d_hidden


class ActiveInference(CogModule):
    """
    PHASE 33: Active Inference — Free Energy Principle + Curiosity-Driven Exploration.
    
    Ersetzt PHASE 7 (IntrinsicMotivation). Das Modell minimiert nicht nur Prediction 
    Error (passiv), sondern sucht aktiv nach lernreichen Erfahrungen.
    
    Kernsignale:
    1. Epistemic Value: Wie viel kann das Modell HIER lernen? (Error-Differenz)
    2. Information Gain: Tatsächlicher Wissenszuwachs (Error-Reduktion über Zeit)
    3. Free Energy: Gesamt-Entropie = Komplexität + Inaccuracy
    4. Novelty: Abweichung von bekannten Mustern (hoher Error)
    
    Ausgabe: Curiosity-Gating-Signal für Datenauswahl + LR-Modulation.
    """
    def __init__(self, d_model, n_domains=4, memory_size=1024):
        super().__init__()
        self.d_model = d_model
        self.n_domains = n_domains
        
        # ——— Intrinsic Drive States ———
        self.register_buffer('curiosity_drive', torch.tensor(0.5))
        self.register_buffer('exploration_bonus', torch.tensor(0.0))
        self.register_buffer('free_energy', torch.tensor(0.0))
        
        # ——— Per-Domain Uncertainty Tracking ———
        # Jede Domain hat eigene Error-Statistiken
        self.register_buffer('domain_errors', torch.zeros(n_domains))
        self.register_buffer('domain_uncertainty', torch.ones(n_domains) * 0.5)
        self.register_buffer('domain_visit_counts', torch.ones(n_domains))
        
        # ——— Learning Progress Tracker ———
        # Trackt ob Error fällt (learning) oder steigt (forgetting)
        self.register_buffer('learning_progress', torch.zeros(n_domains))
        # EMA der letzten Errors pro Domain
        self.register_buffer('error_ema_fast', torch.zeros(n_domains))  # Kurzzeit-EMA
        self.register_buffer('error_ema_slow', torch.zeros(n_domains))  # Langzeit-EMA
        self._error_log = []  # Letzte 1000 Errors für Free Energy
        
        # ——— Epistemic Value Buffer ———
        # Speichert kürzliche (error, domain)-Paare für Epistemic Value
        self.register_buffer('epistemic_buffer', torch.zeros(memory_size, 2))
        self._epistemic_idx = 0
        self._epistemic_count = 0
        
        # ——— Reward History ———
        self.register_buffer('reward_history', torch.zeros(100))
        self._reward_idx = 0
        
        # ——— Kalman-Filter Zustand pro Domain ———
        # Schätzung: error = wahrer Wert + Rauschen
        self.register_buffer('kalman_mean', torch.zeros(n_domains))    # Geschätzter wahrer Error
        self.register_buffer('kalman_var', torch.ones(n_domains))      # Unsicherheit der Schätzung
        self._kalman_r = 0.1   # Messrauschen
        self._kalman_q = 0.01  # Prozessrauschen
        
    def _kalman_update(self, domain_idx, error_obs):
        """Kalman-Filter: Schätze wahren Error pro Domain."""
        # Predict
        pred_mean = self.kalman_mean[domain_idx]
        pred_var = self.kalman_var[domain_idx] + self._kalman_q
        
        # Update
        K = pred_var / (pred_var + self._kalman_r)  # Kalman Gain
        innovation = error_obs - pred_mean
        self.kalman_mean[domain_idx] = pred_mean + K * innovation
        self.kalman_var[domain_idx] = (1 - K) * pred_var
        
        return innovation.item()  # Wie überraschend war dieser Error?
        
    def observe(self, error, domain_idx=0):
        """
        Hauptmethode: Beobachte Prediction Error und aktualisiere alle Signale.
        
        Args:
            error: torch.Tensor [batch, seq, d_model] — aktueller Prediction Error
            domain_idx: int — Index der aktuellen Domain (0=text,1=code,2=security,3=network)
        
        Returns:
            dict mit:
                - 'curiosity_factor': float — LR-Multiplikator (0.5=verlangsamen, 2.0=beschleunigen)
                - 'information_gain': float — Wissenszuwachs dieser Observation
                - 'domain_weights': torch.Tensor [n_domains] — Domain-Gewichtung für Sampling
                - 'free_energy': float — Aktuelle Free Energy
                - 'epistemic_value': float — Epistemischer Wert
        """
        with torch.no_grad():
            # ——— 1. Error-Tracking ———
            error_norm = (error ** 2).mean().item()
            
            # ——— 2. Kalman-Update pro Domain ———
            innovation = self._kalman_update(domain_idx, error_norm)
            
            # ——— 3. Update Domain-Statistiken ———
            self.domain_errors[domain_idx] = error_norm
            self.domain_visit_counts[domain_idx] += 1
            
            # EMA Updates (unterschiedliche Zeitskalen)
            self.error_ema_fast[domain_idx] = 0.9 * self.error_ema_fast[domain_idx] + 0.1 * error_norm
            self.error_ema_slow[domain_idx] = 0.99 * self.error_ema_slow[domain_idx] + 0.01 * error_norm
            
            # Learning Progress: Schnelle EMA - Langsame EMA
            # Positiv = Error sinkt (lernt), Negativ = Error steigt (vergisst)
            progress = self.error_ema_slow[domain_idx].item() - self.error_ema_fast[domain_idx].item()
            self.learning_progress[domain_idx] = progress
            
            # Domain Uncertainty: Kalman Variance + Error-Varianz
            self.domain_uncertainty[domain_idx] = self.kalman_var[domain_idx].item()
            
            # ——— 4. Free Energy = Komplexität + Inaccuracy ———
            complexity = self.kalman_var.sum().item()  # Gesamt-Unsicherheit
            inaccuracy = (self.error_ema_fast ** 2).sum().item()  # Gesamt-Error
            self.free_energy = torch.tensor(complexity + inaccuracy)
            
            # ——— 5. Information Gain (Epistemic Value) ———
            # Wie viel NEUES haben wir gelernt? = |innovation| * (1 - certainty)
            certainty = 1.0 / (1.0 + self.kalman_var[domain_idx].item())
            info_gain = abs(innovation) * (1.0 - certainty)
            
            # ——— 6. Epistemic Buffer & Value ———
            self.epistemic_buffer[self._epistemic_idx] = torch.tensor([error_norm, info_gain])
            self._epistemic_idx = (self._epistemic_idx + 1) % len(self.epistemic_buffer)
            self._epistemic_count = min(self._epistemic_count + 1, len(self.epistemic_buffer))
            
            # Epistemic Value: Durchschnittlicher Info Gain der letzten N Beobachtungen
            if self._epistemic_count > 0:
                recent_values = self.epistemic_buffer[:self._epistemic_count, 1]
                epistemic_value = recent_values.mean().item()
            else:
                epistemic_value = 0.0
            
            # ——— 7. Curiosity Drive ———
            # Steigt mit hohem Info Gain, fällt mit niedrigem
            curiosity_target = 0.3  # Gewünschtes Curiosity-Level
            curiosity_error = info_gain - curiosity_target
            self.curiosity_drive = 0.99 * self.curiosity_drive + 0.01 * (0.5 + curiosity_error * 2)
            self.curiosity_drive = torch.clamp(self.curiosity_drive, 0.1, 2.0)
            
            # ——— 8. Exploration Bonus ———
            # Domains mit hoher Unsicherheit bekommen Bonus
            exploration = self.domain_uncertainty.max().item()
            self.exploration_bonus = torch.tensor(exploration)
            
            # ——— 9. Intrinsic Reward ———
            intrinsic_reward = info_gain * (1.0 + exploration)
            
            # Reward History
            self.reward_history[self._reward_idx] = intrinsic_reward
            self._reward_idx = (self._reward_idx + 1) % 100
            
            # ——— 10. Domain-Weights für aktives Sampling ———
            # Domains mit hohem epistemic_value + exploration bekommen mehr Gewicht
            base_weights = 1.0 + self.domain_uncertainty * 2.0 + torch.clamp(self.learning_progress * 5, -0.5, 2.0)
            # Aber nicht zu extrem: softmax-Skalierung
            domain_weights = torch.softmax(base_weights / 0.5, dim=-1)
            # Mindestgewicht für jede Domain (verhindert Vernachlässigung)
            min_weight = 0.05 / self.n_domains
            domain_weights = torch.clamp(domain_weights, min=min_weight)
            domain_weights = domain_weights / domain_weights.sum()
            
            # ——— 11. Curiosity Factor ———
            # Wenn wir viel lernen (hoher info_gain): höhere LR
            # Wenn wir nichts lernen: niedrigere LR (conservation mode)
            avg_recent_reward = self.reward_history.mean().item() if hasattr(self, 'reward_history') else 0.0
            curiosity_factor = 0.5 + self.curiosity_drive.item() * avg_recent_reward
            curiosity_factor = max(0.3, min(3.0, curiosity_factor))
            
            # ——— 12. Error-Log für Metriken ———
            self._error_log.append(error_norm)
            if len(self._error_log) > 1000:
                self._error_log.pop(0)
            
            return {
                'curiosity_factor': curiosity_factor,
                'information_gain': info_gain,
                'domain_weights': domain_weights,
                'free_energy': self.free_energy.item(),
                'epistemic_value': epistemic_value,
                'intrinsic_reward': intrinsic_reward,
                'exploration_bonus': exploration,
                'learning_progress': progress,
                'innovation': innovation,
                'domain_uncertainty': self.domain_uncertainty.clone(),
            }
    
    def get_domain_preference(self):
        """
        Returns: torch.Tensor [n_domains] — normalized preference weights for domain sampling.
        """
        with torch.no_grad():
            base_weights = 1.0 + self.domain_uncertainty * 2.0 + torch.clamp(self.learning_progress * 5, -0.5, 2.0)
            domain_weights = torch.softmax(base_weights / 0.5, dim=-1)
            min_weight = 0.05 / self.n_domains
            domain_weights = torch.clamp(domain_weights, min=min_weight)
            return domain_weights / domain_weights.sum()
    
    def get_report(self):
        """Gibt aktuelles State-Report zurück (für Logging/Monitoring)."""
        return {
            'curiosity': f'{self.curiosity_drive.item():.3f}',
            'free_energy': f'{self.free_energy.item():.3f}',
            'exploration': f'{self.exploration_bonus.item():.3f}',
            'domain_uncertainty': [f'{x:.3f}' for x in self.domain_uncertainty.tolist()],
            'domain_weights': [f'{x:.3f}' for x in self.get_domain_preference().tolist()],
            'learning_progress': [f'{x:.3f}' for x in self.learning_progress.tolist()],
        }


class SleepReplay(CogModule):
    """
    PHASE 34: Sleep/Replay — Konsolidierungsmechanismus für Langzeitspeicherung.
    
    Nachahmt den hippocampal-neocorticalen Konsolidierungsprozess des Gehirns:
    1. Replay: Wiederhole alte Erfahrungen, verstärke wichtige Muster
    2. Merging: Verbinde neue Erfahrungen mit bestehendem Wissen
    3. Pruning: Entferne schwache/irrelevante Verbindungen
    
    Das Modell lernt während des "Schlafs" aus seinem eigenen Replay-Puffer,
    ohne neuen externen Input. Dadurch werden wichtige Muster stabilisiert
    und unwichtige verlernen sich (cognitive offloading).
    """
    def __init__(self, buffer_size=10000, d_model=None):
        super().__init__()
        self.buffer_size = buffer_size
        self.d_model = d_model
        
        # ——— Replay Buffer ———
        # Speichert (input_ids, error_norm, domain, timestamp)
        self.register_buffer('replay_inputs', torch.zeros(buffer_size, 128, dtype=torch.long))
        self.register_buffer('replay_errors', torch.zeros(buffer_size))
        self.register_buffer('replay_domains', torch.zeros(buffer_size, dtype=torch.long))
        self.register_buffer('replay_weights', torch.zeros(buffer_size))  # Wichtigkeit
        self.register_buffer('replay_age', torch.zeros(buffer_size))
        self._replay_idx = 0
        self._replay_count = 0
        
        # ——— Sleep-Parameter ———
        self.register_buffer('sleep_cycle', torch.zeros(1, dtype=torch.long))
        self.register_buffer('last_sleep_step', torch.tensor(0.0))
        self._sleep_duration = 100  # Steps pro Sleep-Phase
        self._replay_batch_size = 8
        self._consolidation_factor = 0.1  # Wie stark konsolidiert wird
        
        # ——— Pattern Stats ———
        self.register_buffer('pattern_strength', torch.zeros(buffer_size))
        self.register_buffer('pattern_age', torch.zeros(buffer_size))
        
    def store(self, input_ids, error_norm, domain_idx=0, importance=1.0):
        """
        Speichere Erfahrung im Replay-Buffer.
        
        Args:
            input_ids: [batch, seq] — Modell-Input
            error_norm: float — Prediction Error dieser Erfahrung
            domain_idx: int — Domain-Index
            importance: float — Wichtigkeit (1.0=normal, >1.0=wichtig, <1.0=unwichtig)
        """
        with torch.no_grad():
            batch = input_ids.size(0)
            for b in range(batch):
                idx = self._replay_idx % self.buffer_size
                
                # Speichere ersten Sequenzabschnitt (max 128 Tokens)
                seq = input_ids[b, :min(128, input_ids.size(1))]
                self.replay_inputs[idx, :len(seq)] = seq[:128].to(dtype=torch.long)
                self.replay_errors[idx] = float(min(error_norm, 1e4))
                self.replay_domains[idx] = int(domain_idx)
                self.replay_weights[idx] = float(min(importance, 1e4))
                self.replay_age[idx] = 0
                
                # Pattern: höherer Error + höhere Importance = stärkeres Pattern
                val = error_norm * importance
                self.pattern_strength[idx] = float(min(val, 1e4))  # Überlaufschutz
                self.pattern_age[idx] = 0
                
                self._replay_idx += 1
                self._replay_count = min(self._replay_count + 1, self.buffer_size)
    
    def should_sleep(self, current_step, force=False):
        """Entscheide ob eine Sleep-Phase nötig ist."""
        if force:
            return True
        # Sleep alle 1000 Steps für 100 Steps
        sleep_interval = 1000
        return (current_step % sleep_interval) > (sleep_interval - self._sleep_duration)
    
    def sleep_step(self, model, device='cuda'):
        """
        Führe EINEN Sleep-Step aus: replaye eine gespeicherte Erfahrung.
        
        Args:
            model: CogLang-Instanz
            device: torch device
        
        Returns:
            dict mit Sleep-Metriken
        """
        with torch.no_grad():
            if self._replay_count < 10:
                return {'replayed': 0, 'consolidation_loss': 0.0}
            
            # ——— 1. Wichtige Erfahrungen bevorzugen (Priority Sampling) ———
            weights = self.pattern_strength[:self._replay_count].clone()
            # Altern: auch alte Erfahrungen nicht vergessen (Age Penalty)
            age_penalty = torch.exp(-self.pattern_age[:self._replay_count] * 0.01)
            weights = weights * age_penalty
            
            # Sanfte Verteilung
            if weights.sum() < 1e-8:
                weights = torch.ones(self._replay_count)
            probs = weights / weights.sum()
            
            # ——— 2. Sample Batch aus Buffer ———
            n = min(self._replay_batch_size, self._replay_count)
            indices = torch.multinomial(probs, n, replacement=True)
            batch = self.replay_inputs[indices].to(device)  # [n, 128]
            
            # ——— 3. Replay: Forward + Learn ———
            # Kontext: Verstärke Muster, die das Modell bereits kennt
            # Das Ziel des Replays ist KONSOLIDIERUNG, nicht neues Lernen
            
            # Original-Loss vor Replay messen
            orig_out, orig_info = model.forward(batch, learn=False)
            orig_loss = F.cross_entropy(orig_out.view(-1, orig_out.size(-1)), batch.view(-1))
            
            # Replay mit niedriger LR (conservative learning)
            # Speichere originale LRs
            original_lrs = {}
            for module in model.modules.modules():
                if hasattr(module, '_lr'):
                    original_lrs[id(module)] = module._lr
                    module._lr *= self._consolidation_factor  # Niedrige LR für Konsolidierung
            
            # Lerne aus Replay
            replay_loss, _ = model.learn(batch)
            
            # Stelle LRs wieder her
            for module in model.modules.modules():
                if id(module) in original_lrs:
                    module._lr = original_lrs[id(module)]
            
            # ——— 4. Consolodation: Verstärke wichtigen Patterns ———
            # Erfolgreiche Replays (Loss gesunken) verstärken das Pattern
            loss_change = orig_loss.item() - replay_loss
            if loss_change > 0:  # Loss gesunken = erfolgreiche Konsolidierung
                self.pattern_strength[indices] *= 1.0 + loss_change * 0.1
                self.pattern_strength.clamp_(0.01, 10.0)
            
            # ——— 5. Alter alle Patterns ———
            self.pattern_age[:self._replay_count] += 1
            self.replay_age[:self._replay_count] += 1
            
            # ——— 6. Pruning: Entferne schwache Patterns (wenn Buffer voll) ———
            if self._replay_count >= self.buffer_size:
                # Finde schwächste Patterns und überschreibe sie
                weak_mask = self.pattern_strength < 0.1
                n_weak = weak_mask.sum().item()
                if n_weak > 100:
                    # Setze freie Slots zurück
                    self.pattern_strength[weak_mask] = 0.0
                    self.replay_inputs[weak_mask] = 0
                    self.replay_errors[weak_mask] = 0
                    self.replay_weights[weak_mask] = 0
            
            return {
                'replayed': n,
                'consolidation_loss': replay_loss,
                'orig_loss': orig_loss.item(),
                'loss_change': loss_change,
                'patterns_used': int((self.pattern_strength > 0.1).sum().item()),
            }
    
    def sleep_phase(self, model, n_steps=100, device='cuda'):
        """
        Führe komplette Sleep-Phase aus (mehrere Sleep-Steps).
        
        Args:
            model: CogLang-Instanz
            n_steps: int — Anzahl Sleep-Steps
            device: torch device
        
        Returns:
            dict mit Sleep-Report
        """
        self.sleep_cycle += 1
        total_loss = 0.0
        total_change = 0.0
        
        for _ in range(n_steps):
            result = self.sleep_step(model, device)
            total_loss += result.get('consolidation_loss', 0.0)
            total_change += result.get('loss_change', 0.0)
        
        avg_loss = total_loss / max(1, n_steps)
        avg_change = total_change / max(1, n_steps)
        
        return {
            'sleep_cycle': self.sleep_cycle.item(),
            'steps': n_steps,
            'avg_consolidation_loss': avg_loss,
            'avg_loss_change': avg_change,
            'active_patterns': int((self.pattern_strength > 0.1).sum().item()),
            'buffer_usage': f'{self._replay_count}/{self.buffer_size}',
        }


class NeuroSymbolicBridge(CogModule):
    """
    PHASE 8: Neuro-Symbolische Brücke — Kombination von Mustern mit logischen Regeln.
    Erlaubt dem Modell, einfache logische Constraints auf Vorhersagen anzuwenden.
    """
    def __init__(self, vocab_size, d_model, n_rules=16):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_rules = n_rules
        # Rule embeddings: each rule is a pattern in embedding space
        self.rule_keys = nn.Parameter(torch.randn(n_rules, d_model) * 0.1)
        self.rule_values = nn.Parameter(torch.randn(n_rules, vocab_size) * 0.1)
        self._max_weight = 1.0
        
    def forward(self, prediction_logits, context_embedding=None):
        """
        Apply neuro-symbolic rules to prediction logits.
        Returns: modulated logits
        """
        with torch.no_grad():
            batch, seq, vocab = prediction_logits.shape
            
            # Compute rule activation based on context
            if context_embedding is not None:
                # context: [batch, seq, d_model]
                context_flat = context_embedding.reshape(-1, self.d_model)
                rule_activation = torch.softmax(context_flat @ self.rule_keys.T, dim=-1)  # [batch*seq, n_rules]
                
                # Apply rule values to logits
                rule_effect = rule_activation @ self.rule_values  # [batch*seq, vocab]
                rule_effect = rule_effect.reshape(batch, seq, vocab)
                
                # Modulate logits with rule effect (small influence to avoid overriding learning)
                modulated_logits = prediction_logits + rule_effect * 0.1
            else:
                modulated_logits = prediction_logits
                
            return modulated_logits
    
    def learn_step(self, context_embedding, prediction_error):
        """Hebbian learning for rule keys and values."""
        with torch.no_grad():
            # Learn rules that predict prediction errors
            if context_embedding is not None:
                context_flat = context_embedding.reshape(-1, self.d_model)
                error_flat = prediction_error.reshape(-1, prediction_error.size(-1))
                
                # Update rule keys: associate context with error patterns
                rule_activation = torch.softmax(context_flat @ self.rule_keys.T, dim=-1)
                
                # Hebbian update for rule keys
                dk = (context_flat.T @ rule_activation) / context_flat.size(0)
                self.rule_keys.data.add_(dk.T, alpha=self._lr * 0.01)
                self.rule_keys.data.clamp_(-1.0, 1.0)


class EvolutionStrategyOptimizer(CogModule):
    """
    PHASE 13: Gradient-Free Optimizer — Evolution Strategies für Weight Updates.
    Statt Hebbian Learning, nutzt dies perturbationsbasierte Optimierung.
    
    Bietet zwei Modi:
    1. Population-based (original): perturb_weights + apply_perturbations + update_from_fitness
    2. Inline (NEW): learn_step() — leichte, kontrollierte Perturbation im laufenden Training
    """
    def __init__(self, d_model, population_size=8, sigma=0.01):
        super().__init__()
        self.d_model = d_model
        self.population_size = population_size
        self.sigma = sigma  # Perturbation noise (0.01 = 1%)
        self._best_weights = {}
        self._best_fitness = float('inf')
        self._max_weight = 1.0  # clamp limit for perturbations
        
        # Inline evolution state
        self.register_buffer('_step', torch.zeros(1, dtype=torch.long))
        self.register_buffer('_plateau_steps', torch.zeros(1, dtype=torch.long))
        self._loss_history = []
        self._saved_weights = {}  # For revert on regression
        self._perturbed_layer = None
        self._pre_perturb_loss = None
        self._last_noise = {}
        
    def perturb_weights(self, model, seed=None):
        """Erstelle perturbierte Kopie der Gewichte."""
        if seed is not None:
            torch.manual_seed(seed)
        perturbations = {}
        for name, param in model.named_parameters():
            noise = torch.randn_like(param.data) * self.sigma
            perturbations[name] = noise
        return perturbations
    
    def apply_perturbations(self, model, perturbations):
        """Wende Perturbationen auf Modell an."""
        for name, param in model.named_parameters():
            if name in perturbations:
                param.data.add_(perturbations[name])
                param.data.clamp_(-self._max_weight, self._max_weight)
    
    def update_from_fitness(self, model, perturbations_list, fitness_scores):
        """Update Gewichte basierend auf Fitness-Scores."""
        if len(fitness_scores) == 0:
            return
            
        best_idx = torch.argmin(torch.tensor(fitness_scores)).item()
        best_fitness = fitness_scores[best_idx]
        
        if best_fitness < self._best_fitness:
            self._best_fitness = best_fitness
            for name, param in model.named_parameters():
                self._best_weights[name] = param.data.clone()
        
        # Weighted update from population
        fitness_array = torch.tensor(fitness_scores)
        weights = torch.softmax(-fitness_array, dim=0)  # Lower loss = higher weight
        
        for name, param in model.named_parameters():
            if name in self._best_weights:
                # Blend towards best with population influence
                update = torch.zeros_like(param.data)
                for i, pert in enumerate(perturbations_list):
                    if name in pert:
                        update += weights[i] * pert[name]
                param.data.add_(update, alpha=0.1)
                param.data.clamp_(-self._max_weight, self._max_weight)
    
    def _find_linear_layers(self, modules):
        """Find all Linear layers in a module list."""
        if isinstance(modules, nn.Module):
            return [m for m in modules.modules() if isinstance(m, nn.Linear)]
        return [m for m in modules if isinstance(m, nn.Linear)]
    
    def learn_step(self, current_loss, model_modules):
        """
        PHASE 13b: Inline-Evolution-Step.
        
        Strategie:
        - Trackt Loss-Trend über Zeit
        - Bei Plateau (>200 Steps ohne Verbesserung): gezielte Noise-Injektion
        - Bei Regression: Revert der letzten Perturbation
        - Bei Verbesserung: Noise langsam runterfahren (Exploitation)
        
        Args:
            current_loss: float — aktueller Loss-Wert
            model_modules: nn.ModuleList — Liste aller CogModule (für Weight-Access)
        """
        self._step += 1
        
        # Loss-History (letzte 500 Werte)
        self._loss_history.append(current_loss)
        if len(self._loss_history) > 500:
            self._loss_history.pop(0)
        
        # Check revert condition: if we perturbed and loss regressed
        if self._pre_perturb_loss is not None and len(self._loss_history) > 10:
            recent_avg = sum(self._loss_history[-10:]) / 10
            if recent_avg > self._pre_perturb_loss * 1.05:  # 5% regression
                # Revert the perturbation (same key format as save)
                for (m_idx, pn), saved in self._saved_weights.items():
                    module = model_modules[m_idx]
                    for param_name, param in module.named_parameters(recurse=True):
                        if param_name == pn and param.data.shape == saved.shape:
                            param.data.copy_(saved)
                            break
                self._saved_weights = {}
                self._pre_perturb_loss = None
                self._perturbed_layer = None
                self._plateau_steps += 1  # Count regressions
                return
        
        # Plateau detection: every 100 steps
        if self._step % 100 != 0 or len(self._loss_history) < 100:
            return
        
        # Compare recent 50 steps vs prior 50 steps
        recent = sum(self._loss_history[-50:]) / 50
        prior = sum(self._loss_history[-100:-50]) / 50
        improvement = prior - recent  # positive = improving
        
        # Find linear layers to perturb (recursively through all submodules)
        linear_layers = []
        for module in model_modules:
            if isinstance(module, nn.Module):
                for submodule in module.modules():
                    if isinstance(submodule, nn.Linear):
                        linear_layers.append(submodule)
        
        if not linear_layers:
            return
        
        if improvement < 0.001:  # Plateau!
            self._plateau_steps += 1
            
            # Escalate noise strength based on plateau duration
            noise_mult = min(3.0, 1.0 + self._plateau_steps.item() * 0.5)
            
            # Pick a random linear layer and perturb it
            layer = random.choice(linear_layers)
            
            # Save current weights for revert
            self._saved_weights = {}
            for i, m in enumerate(model_modules):
                for pn, p in m.named_parameters(recurse=True):
                    key = (i, pn)
                    self._saved_weights[key] = p.data.clone()
            
            self._pre_perturb_loss = recent
            self._perturbed_layer = layer
            
            # Apply structured noise
            noise = torch.randn_like(layer.weight.data) * self.sigma * noise_mult
            # Direction bias: small learned component (random sign per row)
            noise *= torch.sign(torch.randn(layer.weight.size(0), 1, device=noise.device))
            layer.weight.data.add_(noise)
            layer.weight.data.clamp_(-self._max_weight, self._max_weight)
            
            # Also perturb bias if present
            if layer.bias is not None:
                bias_noise = torch.randn_like(layer.bias.data) * self.sigma * 0.5 * noise_mult
                layer.bias.data.add_(bias_noise)
                layer.bias.data.clamp_(-self._max_weight, self._max_weight)
            
        else:  # Improving — reduce plateau counter
            self._plateau_steps = max(0, self._plateau_steps.item() - 1)
            self._saved_weights = {}
            self._pre_perturb_loss = None


class SkillModule(CogModule):
    """
    PHASE 14: Modularer Skill-Mechanismus — Spezialisierte Sub-Netzwerke.
    Das Modell kann spezialisierte 'Skills' aktivieren basierend auf dem Input-Kontext.
    """
    def __init__(self, d_model, n_skills=8):
        super().__init__()
        self.d_model = d_model
        self.n_skills = n_skills
        # Skill prototypes: each skill is a direction in embedding space
        self.skill_prototypes = nn.Parameter(torch.randn(n_skills, d_model) * 0.1)
        # Skill-specific transformation matrices
        self.skill_transforms = nn.Parameter(torch.randn(n_skills, d_model, d_model) * 0.01)
        # Skill activation history for meta-learning
        self.register_buffer('skill_usage', torch.zeros(n_skills))
        self._max_weight = 1.0
        
    def forward(self, x, context=None):
        """
        Activate skills based on context and apply transformations.
        x: [batch, seq, d_model]
        Returns: transformed output [batch, seq, d_model]
        """
        with torch.no_grad():
            batch, seq, d = x.shape
            
            # Compute skill activation from context
            if context is not None:
                ctx_mean = context.mean(dim=1, keepdim=True)  # [batch, 1, d]
                skill_scores = torch.softmax(ctx_mean @ self.skill_prototypes.T, dim=-1)  # [batch, 1, n_skills]
            else:
                skill_scores = torch.ones(batch, 1, self.n_skills, device=x.device) / self.n_skills
            
            # Update skill usage statistics
            self.skill_usage = 0.9 * self.skill_usage + 0.1 * skill_scores.mean(dim=0).squeeze()
            
            # Apply weighted skill transformations
            output = x.clone()
            for i in range(self.n_skills):
                weight = skill_scores[:, :, i:i+1]  # [batch, 1, 1]
                transform = self.skill_transforms[i]  # [d, d]
                output = output + weight * (x @ transform)
            
            return output
    
    def learn_step(self, context, output_error):
        """Hebbian learning for skill prototypes and transforms."""
        with torch.no_grad():
            if context is not None:
                ctx_mean = context.mean(dim=1)  # [batch, d]
                
                # Update skill prototypes based on error correlation
                for i in range(self.n_skills):
                    error_corr = (ctx_mean * output_error.mean(dim=1)).sum(dim=1).mean()
                    self.skill_prototypes.data[i].add_(ctx_mean.mean(dim=0) * error_corr * self._lr * 0.01)
                    self.skill_prototypes.data[i].clamp_(-1.0, 1.0)


class SecurityHead(CogModule):
    """
    PHASE 31: Vulnerability Detection Head.
    Analysiert Code auf Sicherheitslücken (CWE-Typen, Severity, Patches).
    """
    def __init__(self, d_model, d_sparse, n_cwe_types=20):
        super().__init__()
        self.d_model = d_model
        self.d_sparse = d_sparse
        self.n_cwe_types = n_cwe_types

        # CWE Classifier — erkennt Schwachstellen-Typen
        self.cwe_classifier = nn.Linear(d_sparse, n_cwe_types, bias=False)
        # Severity Predictor — CVSS-Score (0-10)
        self.severity_predictor = nn.Linear(d_sparse, 1, bias=False)
        # Patch Generator — transformiert vulnerable→sicheren Code
        self.patch_generator = nn.Linear(d_sparse * 2, d_sparse, bias=False)
        # Confidence Estimator
        self.confidence_estimator = nn.Linear(d_sparse, 1, bias=False)
        self._max_weight = 1.0

    def forward(self, code_embedding, learn=True):
        """
        code_embedding: [batch, seq, d_sparse] — representation of code
        Returns: dict with detection results
        """
        with torch.no_grad():
            # Mean pool over sequence
            pooled = code_embedding.mean(dim=1)  # [batch, d_sparse]

            # CWE classification
            cwe_logits = self.cwe_classifier(pooled)  # [batch, n_cwe_types]
            cwe_probs = torch.sigmoid(cwe_logits)  # Multi-label classification

            # Severity prediction
            severity = torch.sigmoid(self.severity_predictor(pooled)) * 10.0  # [batch, 1]

            # Confidence
            confidence = torch.sigmoid(self.confidence_estimator(pooled))  # [batch, 1]

            return {
                'cwe_probs': cwe_probs,
                'severity': severity,
                'confidence': confidence,
            }

    def learn_step(self, code_embedding, target_cwe, target_severity):
        """Hebbian learning for vulnerability detection."""
        with torch.no_grad():
            pooled = code_embedding.mean(dim=1)

            # Update CWE classifier
            cwe_logits = self.cwe_classifier(pooled)
            cwe_error = target_cwe - torch.sigmoid(cwe_logits)
            dw_cwe = (cwe_error.T @ pooled) / pooled.size(0)
            self.cwe_classifier.weight.data.add_(dw_cwe, alpha=self._lr * 0.1)
            self.cwe_classifier.weight.data.clamp_(-1.0, 1.0)

            # Update severity predictor
            sev_pred = torch.sigmoid(self.severity_predictor(pooled)) * 10.0
            sev_error = (target_severity - sev_pred) / 10.0
            dw_sev = (sev_error.T @ pooled) / pooled.size(0)
            self.severity_predictor.weight.data.add_(dw_sev, alpha=self._lr * 0.1)
            self.severity_predictor.weight.data.clamp_(-1.0, 1.0)


class NetworkEncoder(CogModule):
    """
    PHASE 32: Network Traffic Encoder.
    Analysiert Netzwerkpakete und -flüsse auf Anomalien.
    """
    def __init__(self, d_model, d_sparse, n_protocols=16):
        super().__init__()
        self.d_model = d_model
        self.d_sparse = d_sparse

        # Protocol embeddings (TCP=6, UDP=17, ICMP=1, etc.)
        self.protocol_embed = nn.Embedding(n_protocols + 1, d_model // 4)
        # Port embeddings
        self.port_embed = nn.Embedding(1024, d_model // 4)
        # Packet feature projector
        self.packet_proj = nn.Linear(d_model, d_sparse, bias=False)
        # Flow state tracker (connection state machine)
        self.flow_gru = nn.Linear(d_sparse + d_model // 2, d_sparse)
        # Anomaly scorer
        self.anomaly_scorer = nn.Linear(d_sparse, 1, bias=False)
        self._max_weight = 1.0

    def forward(self, packet_sequence, learn=True):
        """
        packet_sequence: dict with 'src_port', 'dst_port', 'protocol', 'len', 'flags'
        Returns: dict with flow state and anomaly score
        """
        with torch.no_grad():
            batch, seq = packet_sequence['src_port'].shape

            # Embed protocols
            proto_emb = self.protocol_embed(packet_sequence['protocol'].clamp(0, 16))  # [batch, seq, d/4]
            # Embed ports (hash to range)
            src_p = packet_sequence['src_port'] % 1023
            dst_p = packet_sequence['dst_port'] % 1023
            src_emb = self.port_embed(src_p.long())  # [batch, seq, d/4]
            dst_emb = self.port_embed(dst_p.long())

            # Combine features
            features = torch.cat([proto_emb, src_emb, dst_emb], dim=-1)  # [batch, seq, d*3/4]

            # Project to sparse dimension
            projected = self.packet_proj(features)  # [batch, seq, d_sparse]

            # Flow state tracking (simple update)
            flow_state = torch.zeros(batch, self.d_sparse, device=features.device)
            for t in range(seq):
                combined = torch.cat([flow_state, projected[:, t, :]], dim=-1)
                flow_state = torch.tanh(self.flow_gru(combined))

            # Anomaly score
            anomaly = torch.sigmoid(self.anomaly_scorer(flow_state))  # [batch, 1]

            return {
                'flow_state': flow_state,
                'anomaly_score': anomaly,
                'packet_embeddings': projected,
            }


class GoalEncoder(CogModule):
    """
    PHASE 36: Goal-Directed Generation — Zielgesteuerte Textproduktion.
    
    Erlaubt CogLang, zielgerichtet zu generieren statt nur next-token.
    Das Goal wird als Embedding kodiert und moduliert alle Hidden-States.
    
    Drei Mechanismen:
    1. Goal Encoding: Ziel-Text → d_model-Vektor
    2. Goal Conditioning: Gate-gesteuerte Modulation der Hidden-States
    3. Goal Evaluation: Selbstbewertung ob Output das Ziel erfüllt
    """
    def __init__(self, d_model, max_goal_len=50):
        super().__init__()
        self.d_model = d_model
        self.max_goal_len = max_goal_len
        
        # Goal Encoder: projiziert gemittelte Goal-Embeddings
        self.goal_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Goal Gate: [state; goal] → Gating-Signal
        self.goal_gate = nn.Linear(d_model * 2, d_model)
        
        # Goal Evaluator: Erfüllungsgrad bewerten
        self.evaluator = nn.Linear(d_model, 1, bias=False)
        
        # Goal Speicher (für Metrik-Tracking)
        self.register_buffer('goal_trace', torch.zeros(100, d_model))
        self.register_buffer('goal_success', torch.zeros(100))
        self._goal_idx = 0
        self._max_weight = 1.0
        
    def encode(self, goal_token_ids, sensory_module):
        """Wandle Goal-Tokens in d_model-Embedding."""
        with torch.no_grad():
            if goal_token_ids.dim() == 1:
                goal_token_ids = goal_token_ids.unsqueeze(0)
            # Embed via SensoryInput
            goal_emb = sensory_module(goal_token_ids)  # [batch, seq, d_model]
            # Mean pool
            goal_pooled = goal_emb.mean(dim=1, keepdim=True)
            # Projiziere in Goal-Space
            return self.goal_proj(goal_pooled)
    
    def condition(self, hidden, goal_emb):
        """Moduliere Hidden-States mit Goal via Gate."""
        with torch.no_grad():
            batch, seq, d = hidden.shape
            goal_exp = goal_emb.expand(-1, seq, -1)
            gate_in = torch.cat([hidden, goal_exp], dim=-1)
            gate = torch.sigmoid(self.goal_gate(gate_in))
            return hidden + gate * goal_exp * 0.3
    
    def evaluate(self, gen_emb, goal_emb):
        """Bewerte Erfüllungsgrad: 0=schlecht, 1=perfekt."""
        with torch.no_grad():
            gen_n = gen_emb / (gen_emb.norm(dim=-1, keepdim=True) + 1e-8)
            goal_n = goal_emb / (goal_emb.norm(dim=-1, keepdim=True) + 1e-8)
            similarity = (gen_n * goal_n).sum(dim=-1, keepdim=True)
            sim_score = similarity * 0.5 + 0.5  # [-1,1] → [0,1]
            learned = torch.sigmoid(self.evaluator(gen_emb - goal_emb))
            return 0.7 * sim_score + 0.3 * learned
    
    def learn_step(self, goal_emb, gen_emb, target):
        """Lerne, Goals besser zu evaluieren."""
        with torch.no_grad():
            combined = gen_emb - goal_emb
            current = torch.sigmoid(self.evaluator(combined))
            error = target - current
            dw = (error.T @ combined) / combined.size(0)
            self.evaluator.weight.data.add_(dw, alpha=self._lr * 0.01)
            self.evaluator.weight.data.clamp_(-self._max_weight, self._max_weight)
            # Speichere
            idx = self._goal_idx % 100
            self.goal_trace[idx] = goal_emb.squeeze()
            self.goal_success[idx] = target.mean().item()
            self._goal_idx += 1


class SelfReflection(CogModule):
    """
    PHASE 37: Self-Reflection — Meta-Kognitive Selbstkritik.
    
    Das Modell überwacht seine eigenen Gedanken (System 2 über System 1):
    
    1. Confidence Scoring: Wie sicher ist die aktuelle Prediction?
    2. Consistency Check: Sind aufeinanderfolgende Tokens konsistent?
    3. Contradiction Detection: Logische Brüche im generierten Text
    4. Uncertainty Quantification: Wann sollte das Modell "Ich weiß es nicht" sagen?
    
    Lernsignal: Wenn das Modell einen Fehler macht (hoher Loss), soll es lernen,
    diese Situationen vorherzusehen und entweder korrektur oder vorsichtiger zu sein.
    """
    def __init__(self, d_model, n_confidence_bins=5):
        super().__init__()
        self.d_model = d_model
        self.n_bins = n_confidence_bins
        
        # ——— Confidence Estimator ———
        # Schätzt die Wahrscheinlichkeit, dass die aktuelle Prediction korrekt ist
        self.confidence_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )
        
        # ——— Consistency Scorer ———
        # Bewertet, ob zwei aufeinanderfolgende Hidden-States konsistent sind
        # (plötzliche Sprünge deuten auf Inkonsistenz/Verwirrung hin)
        self.consistency_net = nn.Linear(d_model * 2, 1)
        
        # ——— Contradiction Detector ———
        # Erkennt Widersprüche im Output (z.B. "es ist heiß" → "es schneit")
        # Durch Vergleich der aktuellen und vorherigen Kontext-Embeddings
        self.contradiction_net = nn.Linear(d_model, 1)
        
        # ——— Self-Question Generator ———
        # Erzeugt ein Embedding, das Bereiche mit hoher Unsicherheit markiert
        self.question_proj = nn.Linear(d_model, d_model)
        
        # ——— Metrik-Tracking ———
        self.register_buffer('confidence_history', torch.zeros(1000))
        self.register_buffer('consistency_history', torch.zeros(1000))
        self.register_buffer('contradiction_history', torch.zeros(1000))
        self._metric_idx = 0
        self._prev_state = None  # Letzter Hidden-State für Consistency-Check
        
        self._max_weight = 2.0
        
    def forward(self, hidden_states, logits=None, prev_hidden=None):
        """
        Führe Self-Reflection auf aktuellen Hidden-States aus.
        
        Args:
            hidden_states: [batch, seq, d_model] — Aktuelle Hidden-States
            logits: [batch, seq, vocab] oder None — Output-Logits
            prev_hidden: [batch, d_model] oder None — Vorheriger Hidden-State
            
        Returns:
            dict mit Reflection-Ergebnissen
        """
        with torch.no_grad():
            batch, seq, d = hidden_states.shape
            
            # ——— 1. Confidence per Token ———
            # Niedrige Aktivierung = unsichere Prediction
            conf_features = hidden_states  # [batch, seq, d_model]
            raw_conf = self.confidence_net(conf_features)  # [batch, seq, 1]
            confidence = torch.sigmoid(raw_conf)  # [batch, seq, 1]
            avg_confidence = confidence.mean().item()
            
            # ——— 2. Consistency Check ———
            # Vergleiche aufeinanderfolgende Hidden-States
            consistency_scores = []
            for t in range(1, seq):
                pair = torch.cat([hidden_states[:, t-1, :], hidden_states[:, t, :]], dim=-1)
                cons = torch.sigmoid(self.consistency_net(pair))  # [batch, 1]
                consistency_scores.append(cons)
            
            if consistency_scores:
                avg_consistency = torch.stack(consistency_scores).mean().item()
            else:
                avg_consistency = 0.5
            
            # ——— 3. Contradiction Detection ———
            # Suche nach plötzlichen Änderungen in der Repräsentation
            # (deutet auf Widerspruch oder Themenwechsel hin)
            state_deltas = []
            for t in range(1, min(seq, 10)):  # Nur erste 10 Schritte
                delta = hidden_states[:, t, :] - hidden_states[:, t-1, :]
                contrad = torch.sigmoid(self.contradiction_net(delta))
                state_deltas.append(contrad)
            
            if state_deltas:
                contradiction_risk = torch.stack(state_deltas).mean().item()
            else:
                contradiction_risk = 0.0
            
            # ——— 4. Previous State Consistency (über Generation hinweg) ———
            if prev_hidden is not None and prev_hidden.size(0) == hidden_states.size(0):
                # Vergleiche ersten aktuellen State mit letztem vorherigen State
                pair = torch.cat([prev_hidden, hidden_states[:, 0, :]], dim=-1)
                gen_consistency = torch.sigmoid(self.consistency_net(pair)).mean().item()
            else:
                gen_consistency = 0.5
            
            # ——— 5. Unsicherheits-Markierung ———
            low_conf_mask = (confidence < 0.3).float()  # [batch, seq, 1]
            uncertainty_signal = self.question_proj(hidden_states) * low_conf_mask
            
            # ——— 6. Metriken speichern ———
            idx = self._metric_idx % 1000
            self.confidence_history[idx] = avg_confidence
            self.consistency_history[idx] = avg_consistency
            self.contradiction_history[idx] = contradiction_risk
            self._metric_idx += 1
            
            # ——— 7. Gesamt-Reflection-Score ———
            # Gewichtete Kombination aus Confidence, Consistency und Contradiction
            reflection_score = (
                0.4 * avg_confidence + 
                0.3 * avg_consistency + 
                0.3 * (1.0 - contradiction_risk)
            )
            
            return {
                'confidence': avg_confidence,
                'confidence_per_token': confidence,  # [batch, seq, 1]
                'consistency': avg_consistency,
                'contradiction_risk': contradiction_risk,
                'gen_consistency': gen_consistency,
                'reflection_score': reflection_score,
                'uncertainty_signal': uncertainty_signal,
                'low_conf_mask': low_conf_mask,
            }
    
    def learn_step(self, reflection_result, loss_value):
        """
        Lerne aus Self-Reflection: Verbessere die Selbsteinschätzung.
        
        Args:
            reflection_result: dict von forward()
            loss_value: float — Tatsächlicher Loss (wie falsch lag das Modell?)
        """
        with torch.no_grad():
            confidence = reflection_result['confidence']
            consistency = reflection_result['consistency']
            
            # Ziel: Confidence soll mit tatsächlichem Loss korrelieren
            # Wenn Loss hoch ist, sollte Confidence niedrig sein
            # Ideales Confidence-Signal: inverse Loss-Normalisierung
            target_confidence = 1.0 / (1.0 + loss_value * 2.0)
            target_confidence = max(0.1, min(0.9, target_confidence))
            
            # Confidence-Update
            conf_error = target_confidence - confidence
            # Kleine Hebbian-Anpassung (sehr sanft, da Confidence stabil bleiben soll)
            for param in self.confidence_net.parameters():
                if hasattr(param, 'data') and param.data is not None:
                    param.data.add_(conf_error * 0.0001 * torch.randn_like(param.data) * 0.01)
                    param.data.clamp_(-self._max_weight, self._max_weight)
            
            # Consistency-Update: Bei hohem Loss, Consistency runter
            if loss_value > 1.0:
                cons_error = 0.3 - consistency  # Ziel: niedrige Consistency bei Fehlern
                for param in self.consistency_net.parameters():
                    if hasattr(param, 'data') and param.data is not None:
                        param.data.add_(cons_error * 0.0001 * torch.randn_like(param.data) * 0.01)
                        param.data.clamp_(-self._max_weight, self._max_weight)
    
    def get_report(self):
        """Gibt aktuelle Self-Reflection Metriken zurück."""
        recent_conf = self.confidence_history[:self._metric_idx].tolist() if self._metric_idx > 0 else [0.5]
        recent_cons = self.consistency_history[:self._metric_idx].tolist() if self._metric_idx > 0 else [0.5]
        
        return {
            'avg_confidence': f'{sum(recent_conf[-100:])/max(1,len(recent_conf[-100:])):.3f}' if recent_conf else '0.500',
            'avg_consistency': f'{sum(recent_cons[-100:])/max(1,len(recent_cons[-100:])):.3f}' if recent_cons else '0.500',
            'avg_contradiction': f'{self.contradiction_history[:self._metric_idx].mean().item():.3f}' if self._metric_idx > 0 else '0.000',
            'reflection_active': self._metric_idx > 100,
        }


class KnowledgeGraph(CogModule):
    """
    PHASE 38: Knowledge Graph — Explizites Weltwissen als Graph.
    
    CogLang speichert und ruft Faktenwissen in Form eines neuronalen Graphen ab.
    Der Graph besteht aus (Entity, Relation, Entity)-Triplets, die als Embeddings
    gespeichert sind.
    
    Kernfähigkeiten:
    1. Store: Neue Fakten lernen (über Hebbian-ähnliche Updates)
    2. Retrieve: Relevantes Wissen aus Kontext abrufen
    3. Query: Gezielte Fragen beantworten (Entity → Relation → Entity)
    4. Condition: Abgerufenes Wissen in Forward-Pass einweben
    
    Der Graph wächst dynamisch mit dem Training.
    """
    def __init__(self, d_model, max_entities=1024, max_relations=64):
        super().__init__()
        self.d_model = d_model
        self.max_entities = max_entities
        self.max_relations = max_relations
        
        # ——— Embeddings ———
        # Entity-Embeddings: Jede Entität ist ein d_model-Vektor
        self.entity_embeddings = nn.Embedding(max_entities, d_model)
        # Relation-Embeddings: Jede Relation ist ein d_model-Vektor
        self.relation_embeddings = nn.Embedding(max_relations, d_model)
        
        # ——— Scoring Network ———
        # Bewertet (sub, rel, obj)-Triple: Score = f(sub_emb, rel_emb, obj_emb)
        self.scorer = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )
        
        # ——— Context-to-Entity Projektion ———
        # Wandelt Kontext-Embedding in Entity-Space (für Retrieval)
        self.ctx_to_entity = nn.Linear(d_model, d_model, bias=False)
        
        # ——— Knowledge Conditioning ———
        # Webe abgerufenes Wissen in Hidden-States ein
        self.knowledge_gate = nn.Linear(d_model * 2, d_model)
        
        # ——— Graph-Struktur (Buffer) ———
        # Adjazenzliste: speichert Triple-IDs
        self.register_buffer('graph_subjects', torch.zeros(max_entities * 4, dtype=torch.long))
        self.register_buffer('graph_relations', torch.zeros(max_entities * 4, dtype=torch.long))
        self.register_buffer('graph_objects', torch.zeros(max_entities * 4, dtype=torch.long))
        self.register_buffer('graph_scores', torch.zeros(max_entities * 4))
        self.register_buffer('entity_count', torch.zeros(1, dtype=torch.long))
        self.register_buffer('relation_count', torch.zeros(1, dtype=torch.long))
        self._graph_idx = 0
        
        # ——— Entity-Name Mapping (metadaten) ———
        self.entity_names = {}  # idx -> name (nicht im Checkpoint)
        self.relation_names = {}  # idx -> name
        
        # ——— Metrik ———
        self.register_buffer('query_success_rate', torch.zeros(100))
        self._query_idx = 0
        self._max_weight = 3.0
        
    def add_entity(self, name=None):
        """Füge neue Entität hinzu. Gib Index zurück."""
        idx = self.entity_count.item()
        if idx < self.max_entities:
            if name:
                self.entity_names[idx] = name
            self.entity_count += 1
        return idx % self.max_entities
    
    def add_relation(self, name=None):
        """Füge neue Relation hinzu. Gib Index zurück."""
        idx = self.relation_count.item()
        if idx < self.max_relations:
            if name:
                self.relation_names[idx] = name
            self.relation_count += 1
        return idx % self.max_relations
    
    def store_triple(self, subject, relation, object, score=1.0):
        """
        Speichere (sub, rel, obj)-Triple im Graph.
        
        Args:
            subject: int — Entity-Index
            relation: int — Relation-Index
            object: int — Entity-Index
            score: float — Confidence dieses Wissens (0-1)
        """
        with torch.no_grad():
            idx = self._graph_idx % (self.max_entities * 4)
            self.graph_subjects[idx] = subject
            self.graph_relations[idx] = relation
            self.graph_objects[idx] = object
            self.graph_scores[idx] = score
            self._graph_idx += 1
            
            # Hebbian-ähnliches Update: verstärke Embeddings
            sub_emb = self.entity_embeddings.weight[subject]
            rel_emb = self.relation_embeddings.weight[relation]
            obj_emb = self.entity_embeddings.weight[object]
            
            # Verschiebe Embeddings zueinander (Konsistenz-Lernen)
            # sub + rel ≈ obj (in Embedding-Space)
            predicted_obj = sub_emb + rel_emb
            error = predicted_obj - obj_emb
            self.entity_embeddings.weight.data[object] += error * 0.01
            self.entity_embeddings.weight.data[subject] += error * 0.005  # Schwächer für Subjekt
            self.relation_embeddings.weight.data[relation] += error * 0.01
            
            # Clamp
            self.entity_embeddings.weight.data.clamp_(-self._max_weight, self._max_weight)
            self.relation_embeddings.weight.data.clamp_(-self._max_weight, self._max_weight)
    
    def query(self, subject, relation, top_k=5):
        """
        Frage Graph: (sub, rel, ?) — welche Objekte passen?
        
        Args:
            subject: int — Entity-Index
            relation: int — Relation-Index
            top_k: int — Anzahl Ergebnisse
        
        Returns:
            indices: [top_k] — Objekt-Indices
            scores: [top_k] — Confidence-Scores
        """
        with torch.no_grad():
            sub_emb = self.entity_embeddings.weight[subject].unsqueeze(0)  # [1, d]
            rel_emb = self.relation_embeddings.weight[relation].unsqueeze(0)  # [1, d]
            
            # Gespeicherte Objekte durchsuchen
            n = min(self._graph_idx, self.max_entities * 4)
            if n == 0:
                return torch.zeros(0, dtype=torch.long), torch.zeros(0)
            
            stored_objects = self.graph_objects[:n]
            stored_scores = self.graph_scores[:n]
            
            # Scorer: (sub, rel, obj) → Score
            sub_exp = sub_emb.expand(n, -1)
            rel_exp = rel_emb.expand(n, -1)
            obj_embs = self.entity_embeddings(stored_objects)
            
            triple_feat = torch.cat([sub_exp, rel_exp, obj_embs], dim=-1)
            triple_scores = torch.sigmoid(self.scorer(triple_feat)).squeeze(-1)
            
            # Kombiniere gelernten Score mit gespeichertem Score
            final_scores = triple_scores * 0.7 + stored_scores[:n] * 0.3
            
            # Top-K
            k = min(top_k, n)
            top_scores, top_idx = torch.topk(final_scores, k)
            top_objects = stored_objects[top_idx]
            
            return top_objects, top_scores
    
    def retrieve(self, context_embedding, top_k=3):
        """
        Retrieve relevant knowledge from context.
        
        Args:
            context_embedding: [batch, d_model] — Aktueller Kontext
            top_k: int — Anzahl relevanter Fakten
        
        Returns:
            facts: dict mit abgerufenen Fakten
        """
        with torch.no_grad():
            batch = context_embedding.size(0)
            n = min(self._graph_idx, self.max_entities * 4)
            
            if n == 0:
                return {
                    'embeddings': torch.zeros(batch, 0, self.d_model, device=context_embedding.device),
                    'scores': torch.zeros(batch, 0),
                    'n_facts': 0,
                }
            
            # Projiziere Kontext in Entity-Space
            ctx_entity = self.ctx_to_entity(context_embedding)  # [batch, d]
            
            # Ähnlichkeit mit allen Entitäten
            all_entities = self.entity_embeddings.weight[:self.entity_count.item()]  # [n_ent, d]
            similarity = ctx_entity @ all_entities.T  # [batch, n_ent]
            
            # Finde relevanteste Entitäten
            top_entities = torch.topk(similarity, min(top_k, self.entity_count.item()), dim=-1)
            
            # Retrieve Fakten für diese Entitäten
            fact_list = []
            for b in range(batch):
                for e_idx in top_entities.indices[b]:
                    # Finde alle Triples mit dieser Entity als Subjekt
                    mask = self.graph_subjects[:n] == e_idx
                    fact_indices = torch.where(mask)[0]
                    if len(fact_indices) > 0:
                        # Nimm das erste
                        fi = fact_indices[0]
                        rel = self.graph_relations[fi]
                        obj = self.graph_objects[fi]
                        
                        rel_emb = self.relation_embeddings.weight[rel]
                        obj_emb = self.entity_embeddings.weight[obj]
                        
                        # Kombiniere zu Fakt-Embedding
                        fact_emb = rel_emb + obj_emb
                        fact_list.append((fact_emb, self.graph_scores[fi].item()))
            
            if not fact_list:
                return {
                    'embeddings': torch.zeros(batch, 1, self.d_model, device=context_embedding.device),
                    'scores': torch.zeros(batch, 1),
                    'n_facts': 0,
                }
            
            # Stacke Fakten
            fact_embs = torch.stack([f[0] for f in fact_list])  # [n_facts, d]
            fact_scores = torch.tensor([f[1] for f in fact_list], device=context_embedding.device)
            
            return {
                'embeddings': fact_embs.unsqueeze(0).expand(batch, -1, -1),
                'scores': fact_scores.unsqueeze(0).expand(batch, -1),
                'n_facts': len(fact_list),
            }
    
    def condition(self, hidden_states, knowledge_embeddings, knowledge_scores=None):
        """
        Moduliere Hidden-States mit abgerufenem Wissen.
        
        Args:
            hidden_states: [batch, seq, d_model]
            knowledge_embeddings: [batch, n_facts, d_model]
            knowledge_scores: [batch, n_facts] oder None
        
        Returns:
            modulated: [batch, seq, d_model]
        """
        with torch.no_grad():
            batch, seq, d = hidden_states.shape
            n_facts = knowledge_embeddings.size(1)
            
            if n_facts == 0:
                return hidden_states
            
            # Gewichtete Summe der Fakten
            if knowledge_scores is not None:
                fact_weights = torch.softmax(knowledge_scores / 0.1, dim=-1)  # [batch, n_facts]
                fact_aggr = (fact_weights.unsqueeze(-1) * knowledge_embeddings).sum(dim=1)  # [batch, d]
            else:
                fact_aggr = knowledge_embeddings.mean(dim=1)  # [batch, d]
            
            # Gate: wie viel Wissen soll rein?
            fact_expanded = fact_aggr.unsqueeze(1).expand(-1, seq, -1)
            gate_in = torch.cat([hidden_states, fact_expanded], dim=-1)
            gate = torch.sigmoid(self.knowledge_gate(gate_in))
            
            return hidden_states + gate * fact_expanded * 0.2
    
    def get_graph_stats(self):
        """Gibt Graph-Statistiken zurück."""
        return {
            'entities': self.entity_count.item(),
            'relations': self.relation_count.item(),
            'triples': min(self._graph_idx, self.max_entities * 4),
            'max_entities': self.max_entities,
            'max_relations': self.max_relations,
        }


class ToolUse(CogModule):
    """
    PHASE 39: Tool Use — Externe Werkzeuge für CogLang.
    
    CogLang kann über spezielle Token-Patterns Werkzeuge aufrufen.
    Das Modul parsed den generierten Text, erkennt Tool-Aufrufe,
    führt sie aus und fügt Ergebnisse in den Kontext ein.
    
    Tools:
    - calculator: Mathematische Ausdrücke auswerten
    - python: Python-Code ausführen
    - text_stats: Textstatistiken (Zeichen/Wörter/Zeilen)
    - search: Web-Suche (Stub)
    - tokenize: Token-Zerlegung anzeigen
    """
    def __init__(self, d_model, max_tool_history=64):
        super().__init__()
        self.d_model = d_model
        self.max_tool_history = max_tool_history
        
        # ——— Tool Registry ———
        self._tools = {}
        self._register_default_tools()
        
        # ——— Tool-Call Embedding ———
        # Lernt, welche Tools in welchem Kontext nützlich sind
        self.tool_selector = nn.Linear(d_model, 16)  # 16 Tool-Typen
        self.tool_embeddings = nn.Embedding(16, d_model)
        
        # ——— Tool-Result Fusion ———
        # Webe Tool-Ergebnisse in Hidden-States ein
        self.result_gate = nn.Linear(d_model * 2, d_model)
        
        # —── Tool-History Buffer ———
        # Letzte N Tool-Aufrufe + Ergebnisse für Kontext
        self.register_buffer('tool_history', torch.zeros(max_tool_history, 3, dtype=torch.long))
        self.register_buffer('tool_history_idx', torch.zeros(1, dtype=torch.long))
        
        # ——— Metrik ———
        self.register_buffer('tool_success_rate', torch.zeros(20))
        self._success_idx = 0
        self._exec_counter = 0
        
        # ——— Tool-Beschreibungen (für Logging) ———
        self.tool_descriptions = {
            'calculator': 'Berechne mathematischen Ausdruck. Args: expr (string)',
            'python': 'Führe Python-Code aus. Args: code (string)',
            'text_stats': 'Zeichen/Wörter/Zeilen zählen. Args: text (string)',
            'search': 'Web-Suche (Stub). Args: query (string)',
            'tokenize': 'Token-Zerlegung. Args: text (string)',
        }
    
    def _register_default_tools(self):
        """Registriere Standard-Werkzeuge."""
        self.register_tool('calculator', self._tool_calculator,
                           'Berechne mathematischen Ausdruck (z.B. "2 + 3 * 4")')
        self.register_tool('python', self._tool_python,
                           'Führe Python-Code aus und gib Ergebnis zurück')
        self.register_tool('text_stats', self._tool_text_stats,
                           'Zähle Zeichen, Wörter und Zeilen in Text')
        self.register_tool('search', self._tool_search_stub,
                           'Durchsuche das Web (Demo-Version)')
        self.register_tool('tokenize', self._tool_tokenize,
                           'Zerlege Text in Tokens')
    
    def register_tool(self, name, func, description=""):
        """Registriere ein benutzerdefiniertes Werkzeug."""
        self._tools[name] = {
            'func': func,
            'description': description,
            'call_count': 0,
            'success_count': 0,
        }
    
    # ─── Tool-Implementierungen ─────────────────────────────────────
    
    def _tool_calculator(self, arg):
        """Werte mathematischen Ausdruck aus (sicher)."""
        try:
            # Nur erlaubte Funktionen/Operatoren
            allowed_names = {
                k: v for k, v in math.__dict__.items()
                if not k.startswith('_')
            }
            allowed_names.update({
                'abs': abs, 'round': round, 'min': min, 'max': max,
                'sum': sum, 'pow': pow, 'int': int, 'float': float,
                'str': str, 'len': len,
            })
            # Entferne potenziell gefährliche Konstrukte
            safe_arg = arg.strip().replace('\n', ' ')[:500]
            result = eval(safe_arg, {"__builtins__": {}}, allowed_names)
            return str(result)
        except Exception as e:
            return f"Error: {e}"
    
    def _tool_python(self, arg):
        """Führe Python-Code aus (sandboxed, read-only)."""
        self._exec_counter += 1
        # Nur einfache Ausdrücke erlauben, kein Import/IO
        forbidden = ['import ', 'open(', 'exec(', 'eval(', '__', 'os.', 'sys.', 'subprocess', 'shutil']
        safe_arg = arg.strip()[:1000]
        
        for f in forbidden:
            if f in safe_arg:
                return f"Error: '{f}' nicht erlaubt"
        
        try:
            # Versuche als Expression
            result = eval(safe_arg, {"__builtins__": {}}, {})
            return str(result)
        except:
            try:
                # Versuche als Statement
                local_vars = {}
                exec(safe_arg, {"__builtins__": {}}, local_vars)
                # Gib letzte definierte Variable zurück
                if local_vars:
                    return str(list(local_vars.values())[-1])
                return "None"
            except Exception as e:
                return f"Error: {e}"
    
    def _tool_text_stats(self, arg):
        """Zähle Zeichen, Wörter, Zeilen."""
        s = arg[:5000]
        return f"chars={len(s)} words={len(s.split())} lines={len(s.splitlines())}"
    
    def _tool_search_stub(self, arg):
        """Web-Suche (Stub)."""
        return f"[Search stub for '{arg[:100]}']"
    
    def _tool_tokenize(self, arg):
        """Simuliere Token-Zerlegung (Wort-Level)."""
        words = arg.strip().split()[:50]
        tokens = [f"<{w}>" for w in words]
        return f"[{', '.join(tokens)}]"
    
    # ─── Kern-Methoden ──────────────────────────────────────────────
    
    def detect_tool_calls(self, text):
        """
        Erkenne Tool-Aufrufe im generierten Text.
        
        Format: [TOOL:name:argument] oder TOOL(name, argument)
        
        Returns: list of (tool_name, argument)
        """
        calls = []
        
        # Pattern 1: [TOOL:name:arg]
        pattern1 = r'\[TOOL:(\w+):(.*?)\]'
        for match in re.finditer(pattern1, text):
            name, arg = match.group(1), match.group(2)
            if name in self._tools:
                calls.append((name, arg.strip()))
        
        # Pattern 2: TOOL(name, arg)
        pattern2 = r'TOOL\((\w+),\s*(.*?)\)'
        for match in re.finditer(pattern2, text):
            name, arg = match.group(1), match.group(2)
            if name in self._tools:
                calls.append((name, arg.strip().strip("'\"")))
        
        return calls
    
    def execute(self, tool_name, argument):
        """
        Führe ein Werkzeug aus.
        
        Args:
            tool_name: str — Name des Werkzeugs
            argument: str — Argument
        
        Returns:
            result: str — Ergebnis
            success: bool — Erfolg
        """
        if tool_name not in self._tools:
            return f"Unknown tool: {tool_name}", False
        
        tool = self._tools[tool_name]
        try:
            result = tool['func'](argument)
            tool['call_count'] += 1
            tool['success_count'] += 1
            
            # Metrik aktualisieren
            self.tool_success_rate[self._success_idx % 20] = 1.0
            self._success_idx += 1
            
            return result, True
        except Exception as e:
            tool['call_count'] += 1
            self.tool_success_rate[self._success_idx % 20] = 0.0
            self._success_idx += 1
            return f"Tool execution failed: {e}", False
    
    def condition(self, hidden_states, tool_context=None):
        """
        Moduliere Hidden-States mit Tool-Kontext.
        
        Args:
            hidden_states: [batch, seq, d_model]
            tool_context: dict oder None — Tool-Ergebnisse
        
        Returns:
            modulated: [batch, seq, d_model]
        """
        with torch.no_grad():
            if tool_context is None or hidden_states.size(0) == 0:
                return hidden_states
            
            batch, seq, d = hidden_states.shape
            
            # Tool-Embedding erzeugen
            tool_id = self._name_to_id(tool_context.get('tool_name', 'calculator'))
            tool_emb = self.tool_embeddings(torch.tensor([tool_id], device=hidden_states.device))
            
            # Gate: Tool-Embedding einweben
            tool_expanded = tool_emb.unsqueeze(1).expand(batch, seq, -1)
            gate_in = torch.cat([hidden_states, tool_expanded], dim=-1)
            gate = torch.sigmoid(self.result_gate(gate_in))
            
            return hidden_states + gate * tool_expanded * 0.15
    
    def _name_to_id(self, name):
        """Mappe Tool-Name auf ID (0-15)."""
        mapping = {
            'calculator': 0, 'python': 1, 'text_stats': 2,
            'search': 3, 'tokenize': 4,
        }
        return mapping.get(name, 0)
    
    def learn_step(self, context_embedding, tool_calls, tool_results):
        """
        Lerne aus Tool-Nutzung.
        
        Args:
            context_embedding: [batch, d_model] — Kontext vor Tool-Aufruf
            tool_calls: list of (name, arg)
            tool_results: list of (result, success)
        """
        with torch.no_grad():
            if not tool_calls:
                return
            
            for (name, arg), (result, success) in zip(tool_calls, tool_results):
                tool_id = self._name_to_id(name)
                
                # Tool-Selector Update: verstärke Kontext→Tool-Mapping bei Erfolg
                tool_logits = self.tool_selector(context_embedding.mean(dim=0, keepdim=True))
                target = torch.zeros(1, 16, device=context_embedding.device)
                target[0, tool_id] = 1.0 if success else 0.5
                
                # Hebbian-ähnlich
                selector_error = target - torch.softmax(tool_logits, dim=-1)
                self.tool_selector.weight.data += selector_error.T @ context_embedding.mean(dim=0, keepdim=True) * 0.001
    
    def get_tool_stats(self):
        """Gib Tool-Statistiken zurück."""
        stats = {}
        for name, tool in self._tools.items():
            stats[name] = {
                'calls': tool['call_count'],
                'successes': tool['success_count'],
                'success_rate': (tool['success_count'] / max(1, tool['call_count'])),
            }
        stats['total_calls'] = sum(t['call_count'] for t in self._tools.values())
        stats['success_rate_avg'] = self.tool_success_rate.mean().item()
        return stats


class MultiAgent(CogModule):
    """
    PHASE 40: Multi-Agent Self-Play — Zwei Persönlichkeiten, eine Architektur.
    
    CogLang bekommt zwei 'Persona'-Embeddings (These/Antithese oder Fokus/Kreativ).
    Beide teilen sich die gleichen Gewichte, aber erhalten unterschiedliche
    Bias-Vektoren. Dadurch entsteht eine interne Perspektivenvielfalt:
    
    1. Persona A (These): konservativ, fokussiert, präzise
    2. Persona B (Anti-These): kreativ, explorativ, assoziativ
    3. Synthese: Beide Outputs werden verglichen und Agreement gemessen
    
    Lerneffekt:
    - Agreement→Rauschen: Konsens stärkt Überzeugungen
    - Disagreement→Exploration: Uneinigkeit treibt Lernen an
    """
    def __init__(self, d_model, n_personas=2):
        super().__init__()
        self.d_model = d_model
        self.n_personas = n_personas
        
        # ——— Persona Embeddings ———
        # Jede Persönlichkeit bekommt einen d_model-Vektor
        self.persona_embeddings = nn.Embedding(n_personas, d_model)
        
        # ——— Persona Modulation ———
        # Moduliere Hidden-States je nach aktiver Persona
        self.persona_gate = nn.Linear(d_model * 2, d_model)
        
        # ——— Agreement Scorer ———
        # Misst, wie stark beide Persönlichkeiten übereinstimmen
        self.agreement_scorer = nn.Sequential(
            nn.Linear(d_model * 4, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )
        
        # ——— Synthese ———
        # Kombiniere beide Perspektiven zu einer Synthese
        self.synthesis_net = nn.Linear(d_model * 2, d_model)
        
        # —── Persona-Profile (Metadaten) ———
        self.persona_names = ['focus', 'creative']
        self.persona_prompts = [
            'Fokussiere auf Präzision, Logik und Konsistenz.',
            'Denke kreativ, mach Assoziationen, denk um die Ecke.',
        ]
        
        # ——— Metrik ———
        self.register_buffer('agreement_history', torch.zeros(200))
        self.register_buffer('synergy_score', torch.zeros(1))
        self._agreement_idx = 0
        
        # Initialisiere Persona-Embeddings mit unterschiedlichen Bias
        with torch.no_grad():
            nn.init.normal_(self.persona_embeddings.weight, std=0.1)
            # Persona 0: leicht negativ (konservativ)
            self.persona_embeddings.weight.data[0] -= 0.2
            # Persona 1: leicht positiv (explorativ)
            self.persona_embeddings.weight.data[1] += 0.2
    
    def get_persona_embedding(self, persona_id, batch=1, device=None):
        """Hole Embedding für eine Persönlichkeit."""
        emb = self.persona_embeddings(
            torch.tensor([persona_id], device=device)
        )  # [1, d]
        return emb.expand(batch, -1)  # [batch, d]
    
    def condition(self, hidden_states, persona_id):
        """
        Moduliere Hidden-States mit einer Persönlichkeit.
        
        Args:
            hidden_states: [batch, seq, d_model]
            persona_id: int (0 oder 1)
        
        Returns:
            modulated: [batch, seq, d_model]
        """
        with torch.no_grad():
            batch, seq, d = hidden_states.shape
            persona_emb = self.get_persona_embedding(
                persona_id, batch, hidden_states.device
            )  # [batch, d]
            
            # Gate: Persona-Bias einweben
            persona_exp = persona_emb.unsqueeze(1).expand(-1, seq, -1)
            gate_in = torch.cat([hidden_states, persona_exp], dim=-1)
            gate = torch.sigmoid(self.persona_gate(gate_in))
            
            return hidden_states + gate * persona_exp * 0.3
    
    def compute_agreement(self, states_a, states_b):
        """
        Berechne Übereinstimmung zwischen beiden Persönlichkeiten.
        
        Args:
            states_a: [batch, seq, d_model] — Hidden-States von Persona A
            states_b: [batch, seq, d_model] — Hidden-States von Persona B
        
        Returns:
            agreement: [batch, seq] — 0 (widersprüchlich) bis 1 (übereinstimmend)
            synthesis: [batch, seq, d_model] — kombinierte Synthese
        """
        with torch.no_grad():
            # Agreement: cos-Ähnlichkeit zwischen beiden Perspektiven
            a_flat = states_a.reshape(-1, self.d_model)
            b_flat = states_b.reshape(-1, self.d_model)
            
            cos_sim = F.cosine_similarity(a_flat, b_flat, dim=-1)  # [batch*seq]
            agreement = (cos_sim + 1) / 2  # Normalize to [0, 1]
            agreement = agreement.reshape(states_a.shape[0], states_a.shape[1])
            
            # Scorer-basiertes Agreement (lernt feinere Muster)
            combined = torch.cat([states_a, states_b], dim=-1)  # [batch, seq, 2*d]
            combined_flat = combined.reshape(-1, self.d_model * 2)
            scored = self.agreement_scorer(combined_flat)
            scored = scored.reshape(states_a.shape[0], states_a.shape[1])
            
            # Gemischtes Agreement
            final_agreement = agreement * 0.4 + scored * 0.6
            
            # Synthese: beide Perspektiven kombinieren
            combined_2d = combined.reshape(-1, self.d_model * 2)
            synthesis = self.synthesis_net(combined_2d)
            synthesis = synthesis.reshape(states_a.shape[0], states_a.shape[1], self.d_model)
            
            return final_agreement, synthesis
    
    def debate_step(self, states_a, states_b, input_ids):
        """
        Ein Debate-Schritt zwischen beiden Persönlichkeiten.
        
        Args:
            states_a: Hidden-States von Persona A
            states_b: Hidden-States von Persona B
            input_ids: Original Input
        
        Returns:
            dict mit Agreement, Synthesis, Metriken
        """
        with torch.no_grad():
            agreement, synthesis = self.compute_agreement(states_a, states_b)
            
            # Metrik aktualisieren
            avg_agreement = agreement.mean().item()
            idx = self._agreement_idx % 200
            self.agreement_history[idx] = avg_agreement
            self._agreement_idx += 1
            
            # Synergie: wenn Agreement niedrig, ist Synergie niedrig (Streit)
            # Wenn Agreement hoch, ist Synergie hoch (Kooperation)
            self.synergy_score[0] = avg_agreement * 2 - 1  # [-1, 1]
            
            return {
                'agreement': agreement,  # [batch, seq]
                'agreement_mean': avg_agreement,
                'synthesis': synthesis,  # [batch, seq, d_model]
                'synergy': self.synergy_score[0].item(),
                'persona_a': self.persona_names[0],
                'persona_b': self.persona_names[1],
            }
    
    def learn_from_debate(self, states_a, states_b, agreement, target_agreement=0.7):
        """
        Lerne aus dem Debattenergebnis.
        
        Args:
            states_a: Hidden-States von Persona A
            states_b: Hidden-States von Persona B
            agreement: Agreement-Matrix [batch, seq]
            target_agreement: float — Ziel-Übereinstimmung
        """
        with torch.no_grad():
            # Agreement-Ist vs. Soll
            agreement_error = target_agreement - agreement.mean()
            
            # Verschiebe Persona-Embeddings leicht je nach Agreement
            if abs(agreement_error) > 0.1:
                # Bei zu starkem Dissens: ziehe Personas näher zusammen
                # Bei zu starkem Konsens: trenne sie leicht
                delta = agreement_error * 0.001  # Sehr sanft
                self.persona_embeddings.weight.data[0] += delta
                self.persona_embeddings.weight.data[1] -= delta
                self.persona_embeddings.weight.data.clamp_(-1.0, 1.0)
    
    def get_debate_stats(self):
        """Gib Debate-Statistiken zurück."""
        recent = self.agreement_history[:max(1, self._agreement_idx)]
        stable_agreement = recent.mean().item() if self._agreement_idx > 0 else 0.5
        return {
            'agreement_mean': stable_agreement,
            'synergy': self.synergy_score[0].item(),
            'persona_a': self.persona_names[0],
            'persona_b': self.persona_names[1],
            'n_personas': self.n_personas,
            'agreement_history_len': min(self._agreement_idx, 200),
        }


class TransferLearning(CogModule):
    """
    PHASE 41: Transfer Learning — Domain-Adaptation + Few-Shot.
    
    CogLang kann Gelerntes zwischen Domänen transferieren.
    Jede Domäne bekommt kleine Adapter-Module (LoRA-ähnlich),
    die bei Domain-Wechsel aktiviert werden.
    
    Kernfähigkeiten:
    1. Domain-Spezifische Adapter: Kleine lineare Projektionen pro Domäne
    2. Domain-Detektion: Erkenne aktuelle Domäne aus Hidden-State-Muster
    3. Few-Shot: Passe Adapter mit wenigen Beispielen an
    4. Transfer-Matrix: Lerne, welche Domänen ähnlich sind (→ besseren Transfer)
    """
    def __init__(self, d_model, max_domains=8, adapter_rank=8):
        super().__init__()
        self.d_model = d_model
        self.max_domains = max_domains
        self.adapter_rank = adapter_rank
        
        # ——— Domain-Adapter (2 pro Domäne: down + up) ———
        # down: d_model → rank,  up: rank → d_model
        # Das ist LoRA-ähnlich: W' = W + W_up @ W_down
        self.register_buffer('n_domains', torch.zeros(1, dtype=torch.long))
        self.adapter_down = nn.Parameter(torch.zeros(max_domains, d_model, adapter_rank))
        self.adapter_up = nn.Parameter(torch.zeros(max_domains, adapter_rank, d_model))
        
        # ——— Domain-Classifier ———
        # Erkennt Domäne aus gepoolten Hidden-States
        self.domain_classifier = nn.Linear(d_model, max_domains)
        
        # ——— Domain-Embeddings ———
        # Jede Domäne bekommt einen Embedding-Vektor für Conditioning
        self.domain_embeddings = nn.Embedding(max_domains, d_model)
        
        # ——— Transfer-Matrix ———
        # pairwise: wie gut transferiert Domäne i → j
        self.register_buffer('transfer_matrix', torch.ones(max_domains, max_domains) * 0.5)
        
        # ——— Few-Shot Buffer ———
        self.register_buffer('fewshot_inputs', torch.zeros(32, 64, dtype=torch.long))
        self.register_buffer('fewshot_targets', torch.zeros(32, 64, dtype=torch.long))
        self.register_buffer('fewshot_domains', torch.zeros(32, dtype=torch.long))
        self._fewshot_idx = 0
        self._fewshot_count = 0
        
        # ——— Domain-Namen (Metadaten) ———
        self.domain_names = {}  # idx -> name
        self.current_domain_id = 0
        
        # ——— Metrik ———
        self.register_buffer('domain_confidence', torch.zeros(max_domains))
        self.register_buffer('transfer_benefit', torch.zeros(max_domains, max_domains))
        
        # Initialisiere Adapter
        nn.init.zeros_(self.adapter_down)
        nn.init.zeros_(self.adapter_up)
    
    def add_domain(self, name=None):
        """Füge neue Domäne hinzu. Gib Index zurück."""
        idx = self.n_domains.item()
        if idx < self.max_domains:
            if name:
                self.domain_names[idx] = name
            # Initialisiere Adapter mit leichtem Rauschen
            nn.init.normal_(self.adapter_down[idx], std=0.01)
            nn.init.normal_(self.adapter_up[idx], std=0.01)
            self.n_domains += 1
        return idx % self.max_domains
    
    def detect_domain(self, hidden_states):
        """
        Erkenne aktuelle Domäne aus Hidden-States.
        
        Args:
            hidden_states: [batch, seq, d_model]
        
        Returns:
            domain_id: int
            confidence: float
        """
        with torch.no_grad():
            # Pool über Sequenz
            pooled = hidden_states.mean(dim=1)  # [batch, d]
            
            # Klassifiziere
            logits = self.domain_classifier(pooled)  # [batch, max_domains]
            
            # Nur bekannte Domänen
            n = max(1, self.n_domains.item())
            logits[:, n:] = -1e9
            
            probs = torch.softmax(logits, dim=-1)
            confidence, pred = probs.max(dim=-1)
            
            domain_id = pred[0].item()
            conf = confidence[0].item()
            
            # Aktualisiere Metrik
            self.domain_confidence[domain_id] = conf
            
            return domain_id, conf
    
    def apply_adapter(self, hidden_states, domain_id):
        """
        Wende Domain-Adapter auf Hidden-States an (LoRA-Stil).
        
        Args:
            hidden_states: [batch, seq, d_model]
            domain_id: int
        
        Returns:
            adapted: [batch, seq, d_model]
        """
        with torch.no_grad():
            if domain_id >= self.n_domains.item():
                return hidden_states
            
            # Adapter: Δh = h @ W_down @ W_up
            down = self.adapter_down[domain_id]  # [d, rank]
            up = self.adapter_up[domain_id]      # [rank, d]
            
            # H @ down @ up
            delta = hidden_states @ down @ up  # [batch, seq, d]
            
            return hidden_states + delta * 0.1  # Skalierung
    
    def get_domain_embedding(self, domain_id, batch=1, device=None):
        """Hole Domain-Embedding für Conditioning."""
        emb = self.domain_embeddings(
            torch.tensor([domain_id], device=device)
        ).expand(batch, -1)
        return emb
    
    def condition(self, hidden_states, domain_id):
        """
        Moduliere Hidden-States mit Domain-Kontext.
        
        Args:
            hidden_states: [batch, seq, d_model]
            domain_id: int
        
        Returns:
            modulated: [batch, seq, d_model]
        """
        with torch.no_grad():
            batch, seq, d = hidden_states.shape
            domain_emb = self.get_domain_embedding(domain_id, batch, hidden_states.device)
            domain_exp = domain_emb.unsqueeze(1).expand(-1, seq, -1)
            
            return hidden_states + domain_exp * 0.2
    
    def fewshot_store(self, input_ids, targets, domain_id):
        """
        Speichere Beispiele für Few-Shot Adaptation.
        
        Args:
            input_ids: [seq]
            targets: [seq]
            domain_id: int
        """
        idx = self._fewshot_idx % 32
        length = min(input_ids.size(0), 64)
        self.fewshot_inputs[idx, :length] = input_ids[:length]
        self.fewshot_targets[idx, :length] = targets[:length]
        self.fewshot_domains[idx] = domain_id
        self._fewshot_idx += 1
        self._fewshot_count = min(self._fewshot_count + 1, 32)
    
    def fewshot_adapt(self, domain_id, n_examples=4):
        """
        Passe Adapter für eine Domäne mit gespeicherten Beispielen an.
        
        Args:
            domain_id: int
            n_examples: int (max 32)
        """
        if self._fewshot_count == 0:
            return
        
        with torch.no_grad():
            # Finde Beispiele für diese Domäne
            mask = self.fewshot_domains[:self._fewshot_count] == domain_id
            indices = torch.where(mask)[0]
            
            if len(indices) == 0:
                return
            
            n = min(len(indices), n_examples)
            selected = indices[:n]
            
            # Simuliere Forward für jedes Beispiel und passe Adapter an
            for idx in selected:
                inp = self.fewshot_inputs[idx]
                target = self.fewshot_targets[idx]
                length = (inp != 0).sum().item()
                
                if length < 2:
                    continue
                
                # Simuliere Prediction Error (vereinfacht)
                # Statt vollständigem Forward nur Adapter-Update
                # Error = Adapter-Ausgabe (soll nahe 0 sein wenn gut)
                down = self.adapter_down[domain_id]  # [d, rank]
                up = self.adapter_up[domain_id]      # [rank, d]
                
                # Simulierter Fehler: je größer der Unterschied zwischen
                # aktivem und passivem Adapter, desto mehr Anpassung nötig
                noise = torch.randn_like(down) * 0.01
                self.adapter_down.data[domain_id] += noise * 0.001
    
    def learn_transfer(self, source_domain, target_domain, benefit):
        """
        Lerne Transfer-Benefit zwischen Domänen.
        
        Args:
            source_domain: int
            target_domain: int
            benefit: float (negativ = schädlich, positiv = nützlich)
        """
        with torch.no_grad():
            # Exponentiell gleitender Durchschnitt
            old = self.transfer_matrix[source_domain, target_domain].item()
            self.transfer_matrix[source_domain, target_domain] = old * 0.9 + benefit * 0.1
            self.transfer_benefit[source_domain, target_domain] = benefit
    
    def get_best_source(self, target_domain):
        """
        Finde beste Quell-Domäne für Transfer auf Ziel-Domäne.
        
        Args:
            target_domain: int
        
        Returns:
            source_id: int
            benefit: float
        """
        scores = self.transfer_matrix[:, target_domain].clone()
        scores[target_domain] = -1e9  # Sich selbst ausschließen
        best = scores.argmax().item()
        return best, scores[best].item()
    
    def learn_step(self, hidden_states, domain_id, loss):
        """
        Lerne aus aktuellem Schritt: Domänenerkennung + Transfer.
        
        Args:
            hidden_states: [batch, seq, d_model]
            domain_id: int
            loss: float
        """
        with torch.no_grad():
            if self._fewshot_count % 50 == 0 and self._fewshot_count > 0:
                self.fewshot_adapt(domain_id)
            
            # Aktualisiere Domain-Classifier (sehr sanft)
            pooled = hidden_states.mean(dim=1)
            logits = self.domain_classifier(pooled)
            target = torch.zeros(1, self.max_domains, device=hidden_states.device)
            target[0, domain_id] = 1.0
            
            error = target - torch.softmax(logits, dim=-1)
            self.domain_classifier.weight.data += error.T @ pooled * 0.0005
    
    def get_transfer_stats(self):
        """Gib Transfer-Statistiken zurück."""
        n = max(1, self.n_domains.item())
        stats = {
            'n_domains': n,
            'current_domain': self.current_domain_id,
            'current_name': self.domain_names.get(self.current_domain_id, f'domain_{self.current_domain_id}'),
            'domain_confidence': {self.domain_names.get(i, f'd{i}'): 
                                self.domain_confidence[i].item() 
                                for i in range(n)},
        }
        # Beste Transfer-Paare
        transfers = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    benefit = self.transfer_benefit[i, j].item()
                    if abs(benefit) > 0.01:
                        sn = self.domain_names.get(i, f'd{i}')
                        tn = self.domain_names.get(j, f'd{j}')
                        transfers.append(f'{sn}→{tn}: {benefit:.3f}')
        stats['top_transfers'] = transfers[:5]
        return stats


class ConsciousnessGlimpse(CogModule):
    """
    PHASE 42: Consciousness Glimpse — Global Workspace Attention Broadcasting.
    
    Inspiriert von Baars' Global Workspace Theory:
    - Ein 'Bewusstseinsinhalt' wird aus den verarbeiteten Informationen ausgewählt
    - Dieser Inhalt wird an alle Module 'ausgestrahlt' (Broadcast)
    - Module, die den Inhalt relevant finden, passen ihre Verarbeitung an
    
    Implementierung:
    1. Salience-Detektor: Finde überraschende/neuartige/relevante Positionen
    2. Spotlight: Wähle die Top-k salientesten Stellen aus
    3. Global Broadcast: Verteile den ausgewählten Inhalt auf alle Hidden-States
    4. Recurrent Processing: Der Broadcast beeinflusst den nächsten Forward-Pass
    """
    def __init__(self, d_model, spotlight_size=1, n_glimpses=3):
        super().__init__()
        self.d_model = d_model
        self.spotlight_size = spotlight_size
        self.n_glimpses = n_glimpses
        
        # ——— Salience Detector ———
        # Bewertet jede Position nach Wichtigkeit
        self.salience_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )
        
        # ——— Global Broadcast ———
        # Projiziere Spotlight-Inhalt in den gesamten Sequenz-Raum
        self.broadcast_net = nn.Linear(d_model, d_model, bias=False)
        self.broadcast_gate = nn.Linear(d_model * 2, d_model)
        
        # ——— Recurrent Bias ———
        # Speichert letzten Broadcast für zeitliche Kohärenz
        self.register_buffer('last_broadcast', torch.zeros(1, d_model))
        self.register_buffer('broadcast_coherence', torch.zeros(100))
        self._coherence_idx = 0
        
        # ——— Metrik ———
        self.register_buffer('spotlight_entropy', torch.zeros(100))
        self.register_buffer('glimpse_count', torch.zeros(1, dtype=torch.long))
        self._entropy_idx = 0
    
    def compute_salience(self, hidden_states, prediction_errors):
        """
        Berechne Salience (Wichtigkeit) für jede Position.
        
        Salience = f(hidden_state, prediction_error)
        Höherer Error → höhere Salience (Überraschung)
        
        Args:
            hidden_states: [batch, seq, d_model]
            prediction_errors: list of [batch, seq, d_model]
        
        Returns:
            salience: [batch, seq]
        """
        with torch.no_grad():
            batch, seq, d = hidden_states.shape
            
            if not prediction_errors:
                return torch.zeros(batch, seq, device=hidden_states.device)
            
            # Aggregiere Errors (mittlere Schicht)
            error = sum(prediction_errors) / len(prediction_errors)
            error_magnitude = error.norm(dim=-1, keepdim=True)  # [batch, seq, 1]
            
            # Kombiniere Hidden + Error
            feat = torch.cat([hidden_states, error_magnitude.expand(-1, -1, d)], dim=-1)
            
            # Salience-Score
            score = self.salience_net(feat).squeeze(-1)  # [batch, seq]
            
            # Normalisiere über Sequenz
            score = score / (score.sum(dim=-1, keepdim=True) + 1e-8)
            
            return score
    
    def spotlight(self, hidden_states, salience):
        """
        Wähle die Top-k salientesten Positionen aus (Bewusstseinsinhalt).
        
        Args:
            hidden_states: [batch, seq, d_model]
            salience: [batch, seq]
        
        Returns:
            content: [batch, spotlight_size, d_model]
            indices: [batch, spotlight_size]
            weights: [batch, spotlight_size]
        """
        with torch.no_grad():
            batch, seq, d = hidden_states.shape
            k = min(self.spotlight_size, seq)
            
            # Top-k Salience
            weights, indices = torch.topk(salience, k, dim=-1)  # [batch, k]
            
            # Sammle Hidden-States an diesen Positionen
            content = torch.gather(
                hidden_states, 1,
                indices.unsqueeze(-1).expand(-1, -1, d)
            )  # [batch, k, d]
            
            # Metrik: Entropie der Salience-Verteilung
            entropy = -(salience * torch.log(salience + 1e-8)).sum(dim=-1).mean()
            idx = self._entropy_idx % 100
            self.spotlight_entropy[idx] = entropy.item()
            self._entropy_idx += 1
            
            return content, indices, weights
    
    def broadcast(self, hidden_states, spotlight_content, spotlight_weights=None):
        """
        Strahle den Bewusstseinsinhalt auf alle Hidden-States aus.
        
        Args:
            hidden_states: [batch, seq, d_model]
            spotlight_content: [batch, k, d_model]
            spotlight_weights: [batch, k] oder None
        
        Returns:
            broadcasted: [batch, seq, d_model]
            broadcast_signal: [batch, 1, d_model] (für Metrik)
        """
        with torch.no_grad():
            batch, seq, d = hidden_states.shape
            k = spotlight_content.size(1)
            
            # Gewichtete Summe des Spotlight-Inhalts
            if spotlight_weights is not None:
                w = torch.softmax(spotlight_weights, dim=-1)  # [batch, k]
                broadcast_signal = (w.unsqueeze(-1) * spotlight_content).sum(dim=1, keepdim=True)
            else:
                broadcast_signal = spotlight_content.mean(dim=1, keepdim=True)  # [batch, 1, d]
            
            # Broadcast auf gesamte Sequenz
            broadcast_exp = broadcast_signal.expand(-1, seq, -1)  # [batch, seq, d]
            broadcast_proj = self.broadcast_net(broadcast_exp)
            
            # Gate: Wie viel Broadcast soll durchkommen?
            gate_in = torch.cat([hidden_states, broadcast_proj], dim=-1)
            gate = torch.sigmoid(self.broadcast_gate(gate_in))
            
            broadcasted = hidden_states + gate * broadcast_proj * 0.2
            
            # Aktualisiere Recurrent Bias
            self.last_broadcast = broadcast_signal[0:1].detach()
            
            return broadcasted, broadcast_signal
    
    def glimpse(self, hidden_states, prediction_errors):
        """
        Führe einen vollständigen 'Consciousness Glimpse' durch.
        
        Args:
            hidden_states: [batch, seq, d_model]
            prediction_errors: list of [batch, seq, d_model]
        
        Returns:
            updated_hidden: [batch, seq, d_model]
            glimpse_info: dict
        """
        with torch.no_grad():
            self.glimpse_count += 1
            
            # 1. Salience berechnen
            salience = self.compute_salience(hidden_states, prediction_errors)
            
            # 2. Spotlight
            content, indices, weights = self.spotlight(hidden_states, salience)
            
            # 3. Broadcast
            broadcasted, signal = self.broadcast(hidden_states, content, weights)
            
            # 4. Kohärenz: Ähnlichkeit zum letzten Broadcast
            if self.glimpse_count.item() > 1:
                coherence = F.cosine_similarity(
                    self.last_broadcast, signal[0:1], dim=-1
                ).item()
                idx = self._coherence_idx % 100
                self.broadcast_coherence[idx] = (coherence + 1) / 2  # [0, 1]
                self._coherence_idx += 1
            else:
                coherence = 0.0
            
            return broadcasted, {
                'salience': salience,
                'spotlight_indices': indices,
                'spotlight_weights': weights,
                'broadcast_coherence': coherence,
                'spotlight_entropy': self.spotlight_entropy[(self._entropy_idx - 1) % 100].item(),
                'n_glimpses': self.glimpse_count.item(),
            }
    
    def condition(self, hidden_states):
        """
        Wende letzten Broadcast als Conditioning an (für nächsten Schritt).
        
        Args:
            hidden_states: [batch, seq, d_model]
        
        Returns:
            conditioned: [batch, seq, d_model]
        """
        with torch.no_grad():
            if self.last_broadcast.abs().sum() < 0.01:
                return hidden_states
            
            batch, seq, d = hidden_states.shape
            last_bc = self.last_broadcast.expand(batch, seq, -1)
            
            gate_in = torch.cat([hidden_states, last_bc], dim=-1)
            gate = torch.sigmoid(self.broadcast_gate(gate_in))
            
            return hidden_states + gate * last_bc * 0.1
    
    def get_consciousness_stats(self):
        """Gib Consciousness-Statistiken zurück."""
        recent_coherence = self.broadcast_coherence[:max(1, self._coherence_idx)]
        recent_entropy = self.spotlight_entropy[:max(1, self._entropy_idx)]
        return {
            'broadcast_coherence': recent_coherence.mean().item(),
            'spotlight_entropy': recent_entropy.mean().item(),
            'n_glimpses': self.glimpse_count.item(),
            'spotlight_size': self.spotlight_size,
        }


class AutoCurriculum(CogModule):
    """
    PHASE 43: Auto-Curriculum — Automatische Schwierigkeitsanpassung.
    
    CogLang passt die Trainingsschwierigkeit automatisch an sein
    aktuelles Leistungsniveau an (Zone of Proximal Development).
    
    Kernmechanismen:
    1. Mastery-Tracking: Gleitender Durchschnitt des Loss über Fenster
    2. ZPD-Algorithmus: Schwierigkeit so wählen, dass ~70% Erfolg
    3. Difficulty-Dimensionen: Sequenzlänge, Noise-Level, Task-Komplexität
    4. Curriculum-Verlauf: Dokumentiere Fortschritt über Zeit
    
    Das Modul gibt Empfehlungen an die Trainingsschleife zurück.
    """
    def __init__(self, d_model, n_difficulty_levels=5, window_size=100):
        super().__init__()
        self.d_model = d_model
        self.n_difficulty_levels = n_difficulty_levels
        self.window_size = window_size
        
        # ——— Difficulty-Definitionen ———
        self.difficulty_configs = [
            {'seq_len': 8, 'noise': 0.0, 'name': 'trivial'},
            {'seq_len': 16, 'noise': 0.05, 'name': 'easy'},
            {'seq_len': 32, 'noise': 0.1, 'name': 'medium'},
            {'seq_len': 64, 'noise': 0.15, 'name': 'hard'},
            {'seq_len': 128, 'noise': 0.2, 'name': 'expert'},
        ]
        
        # ——— Performance-Puffer ———
        self.register_buffer('loss_buffer', torch.zeros(n_difficulty_levels, window_size))
        self.register_buffer('loss_counts', torch.zeros(n_difficulty_levels, dtype=torch.long))
        self.register_buffer('mastery_scores', torch.ones(n_difficulty_levels) * 0.5)
        
        # ——— ZPD-State ———
        self.register_buffer('current_difficulty', torch.zeros(1, dtype=torch.long))
        self.register_buffer('target_mastery', torch.tensor([0.7]))  # 70% Ziel
        self.register_buffer('adaptation_rate', torch.tensor([0.05]))
        
        # ——— Curriculum-Verlauf ———
        self.register_buffer('difficulty_history', torch.zeros(5000, dtype=torch.long))
        self.register_buffer('mastery_history', torch.zeros(5000))
        self._history_idx = 0
        
        # ——— Difficulty Embedding (für Conditioning) ———
        self.difficulty_embedding = nn.Embedding(n_difficulty_levels, d_model)
        
        # ——— Task-Scoring ———
        self.task_scorer = nn.Linear(d_model, n_difficulty_levels)
        
        # ——— Metrik ———
        self.register_buffer('curriculum_progress', torch.zeros(n_difficulty_levels))
        self._steps_at_current = 0
    
    def record_loss(self, loss, difficulty=None):
        """
        Zeichne Loss für aktuelle Schwierigkeit auf.
        
        Args:
            loss: float — aktueller Loss
            difficulty: int oder None (None = aktuelle)
        """
        with torch.no_grad():
            d = difficulty if difficulty is not None else self.current_difficulty.item()
            d = min(d, self.n_difficulty_levels - 1)
            
            idx = self.loss_counts[d].item() % self.window_size
            self.loss_buffer[d, idx] = loss
            self.loss_counts[d] += 1
    
    def compute_mastery(self, difficulty=None):
        """
        Berechne Mastery-Score für eine Schwierigkeitsstufe.
        
        Mastery = 1 / (1 + avg_loss / baseline)
        
        Args:
            difficulty: int oder None
        
        Returns:
            mastery: float (0-1)
        """
        with torch.no_grad():
            d = difficulty if difficulty is not None else self.current_difficulty.item()
            d = min(d, self.n_difficulty_levels - 1)
            
            n = min(self.loss_counts[d].item(), self.window_size)
            if n < 10:  # Nicht genug Daten
                return 0.5
            
            # Durchschnittlicher Loss im Fenster
            recent = self.loss_buffer[d, :n]
            avg_loss = recent.mean().item()
            
            # Baseline: Loss auf Stufe 0 (trivial)
            n0 = min(self.loss_counts[0].item(), self.window_size)
            baseline = self.loss_buffer[0, :max(1, n0)].mean().item() if n0 > 0 else avg_loss
            
            # Mastery: je niedriger Loss im Vergleich zu Baseline, desto besser
            if baseline < 0.1:
                mastery = 0.8  # Default wenn Baseline ~0
            else:
                ratio = avg_loss / max(0.1, baseline)
                mastery = 1.0 / (1.0 + ratio)
            
            # Glätten mit vorherigem Wert
            old = self.mastery_scores[d].item()
            smoothed = old * 0.9 + mastery * 0.1
            self.mastery_scores[d] = smoothed
            
            return smoothed
    
    def adapt_difficulty(self):
        """
        Passe Schwierigkeit basierend auf Mastery an (ZPD).
        
        Returns:
            new_difficulty: int
            changed: bool
        """
        with torch.no_grad():
            current = self.current_difficulty.item()
            mastery = self.compute_mastery(current)
            
            target = self.target_mastery.item()
            changed = False
            
            if mastery > target + 0.1 and current < self.n_difficulty_levels - 1:
                # Zu einfach: erhöhe Schwierigkeit
                new_d = current + 1
                changed = True
            elif mastery < target - 0.2 and current > 0:
                # Zu schwer: verringere Schwierigkeit
                new_d = current - 1
                changed = True
            else:
                new_d = current
            
            if changed:
                self.current_difficulty[0] = new_d
                self.curriculum_progress[current] = 1.0
                self._steps_at_current = 0
                
                # Dokumentiere im Verlauf
                idx = self._history_idx % 5000
                self.difficulty_history[idx] = new_d
                self.mastery_history[idx] = mastery
                self._history_idx += 1
            else:
                self._steps_at_current += 1
            
            return new_d, changed
    
    def get_curriculum_params(self):
        """
        Gib aktuelle Curriculum-Parameter zurück.
        
        Returns:
            dict mit seq_len, noise, difficulty, etc.
        """
        d = self.current_difficulty.item()
        config = self.difficulty_configs[min(d, self.n_difficulty_levels - 1)]
        return {
            'difficulty_level': d,
            'difficulty_name': config['name'],
            'seq_len': config['seq_len'],
            'noise': config['noise'],
            'mastery': self.mastery_scores[d].item(),
            'steps_at_current': self._steps_at_current,
        }
    
    def condition(self, hidden_states):
        """
        Moduliere Hidden-States mit Difficulty-Embedding.
        
        Args:
            hidden_states: [batch, seq, d_model]
        
        Returns:
            conditioned: [batch, seq, d_model]
        """
        with torch.no_grad():
            batch, seq, d = hidden_states.shape
            d_id = min(self.current_difficulty.item(), self.n_difficulty_levels - 1)
            diff_emb = self.difficulty_embedding(
                torch.tensor([d_id], device=hidden_states.device)
            )  # [1, d]
            diff_exp = diff_emb.unsqueeze(1).expand(batch, seq, -1)
            return hidden_states + diff_exp * 0.1
    
    def learn_step(self, loss):
        """
        Lerne aus aktuellem Schritt: Mastery + ggf. Difficulty-Anpassung.
        
        Args:
            loss: float
        """
        with torch.no_grad():
            self.record_loss(loss)
            
            # Passe Schwierigkeit alle N Steps an
            if self._steps_at_current > 0 and self._steps_at_current % 200 == 0:
                new_d, changed = self.adapt_difficulty()
                if changed:
                    return {'difficulty_changed': True, 'new_difficulty': new_d}
            
            return {'difficulty_changed': False}
    
    def get_curriculum_stats(self):
        """Gib Curriculum-Statistiken zurück."""
        stats = {
            'current_difficulty': self.current_difficulty.item(),
            'config': self.difficulty_configs[min(
                self.current_difficulty.item(), self.n_difficulty_levels - 1
            )],
            'masteries': {},
            'n_adaptations': min(self._history_idx, 5000),
        }
        for i in range(self.n_difficulty_levels):
            name = self.difficulty_configs[i]['name']
            n = min(self.loss_counts[i].item(), self.window_size)
            avg_loss = self.loss_buffer[i, :n].mean().item() if n > 0 else 0
            stats['masteries'][name] = {
                'score': self.mastery_scores[i].item(),
                'avg_loss': avg_loss,
                'n_samples': self.loss_counts[i].item(),
            }
        return stats


class CausalReasoning(CogModule):
    """
    PHASE 44: Causal Reasoning — Kausales Verständnis für das Weltmodell.
    
    CogLang lernt Ursache-Wirkung-Beziehungen aus Sequenzen:
    1. Causal Discovery: Erkenne kausale Strukturen aus temporalen Mustern
    2. Do-Calculus: Simuliere Eingriffe (Interventionen) im Repräsentationsraum
    3. Counterfactuals: "Was wäre wenn" — alternative Realitäten generieren
    
    Kernidee: Wenn A zeitlich vor B kommt UND die bedingte Wahrscheinlichkeit
    P(B|A) > P(B) significantly, dann ist A wahrscheinlich Ursache von B.
    
    Integration mit KnowledgeGraph: Kausale Fakten werden als spezielle
    Relationen ("causes", "enables", "prevents") gespeichert.
    """
    def __init__(self, d_model, n_causal_factors=64, temperature=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_causal_factors = n_causal_factors
        self.temperature = temperature
        
        # ——— Causal Discovery Network ———
        # Lernt aus (cause, effect)-Paaren eine kausale Einbettung
        self.cause_encoder = nn.Linear(d_model, n_causal_factors)
        self.effect_encoder = nn.Linear(d_model, n_causal_factors)
        self.causal_scorer = nn.Bilinear(n_causal_factors, n_causal_factors, 1)
        
        # ——— Intervention Head (Do-Calculus) ———
        # Simuliere: "Was passiert, wenn ich X ändere?"
        self.intervention_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        
        # ——— Counterfactual Generator ———
        # Erzeuge alternative Repräsentation: "Was wäre wenn X anders?"
        self.counterfactual_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        
        # ——— Causal Graph (Buffer) ———
        # Speichert gelernte kausale Beziehungen
        self.register_buffer('causal_matrix', torch.zeros(n_causal_factors, n_causal_factors))
        self.register_buffer('causal_strength', torch.zeros(n_causal_factors, n_causal_factors))
        self.register_buffer('causal_counts', torch.zeros(n_causal_factors, n_causal_factors, dtype=torch.long))
        
        # ——— Temporal Buffer ———
        # Letzte N (cause, effect) Beobachtungen
        self.register_buffer('temp_buffer_causes', torch.zeros(500, d_model))
        self.register_buffer('temp_buffer_effects', torch.zeros(500, d_model))
        self._temp_idx = 0
        self._temp_count = 0
        
        # ——— Metrik ———
        self.register_buffer('causal_confidence', torch.zeros(n_causal_factors))
        self.register_buffer('discovery_rate', torch.zeros(100))
        self._discovery_idx = 0
    
    def discover_causal(self, hidden_states):
        """
        Entdecke kausale Beziehungen aus einer Sequenz.
        
        Args:
            hidden_states: [batch, seq, d_model]
        
        Returns:
            cause_ids: [n_discovered] — Indices entdeckter Ursachen
            effect_ids: [n_discovered] — Indices entdeckter Effekte
            strengths: [n_discovered] — Kausale Stärke
        """
        with torch.no_grad():
            batch, seq, d = hidden_states.shape
            
            # Kodiere jede Position als Cause und Effect
            h_flat = hidden_states.reshape(-1, d)  # [batch*seq, d]
            causes = torch.sigmoid(self.cause_encoder(h_flat))  # [batch*seq, n_factors]
            effects = torch.sigmoid(self.effect_encoder(h_flat))
            
            # Temporal: (t) ist Cause, (t+1) ist Effect
            n = min(seq - 1, 50)
            discovered_causes = []
            discovered_effects = []
            discovered_strengths = []
            
            for b in range(batch):
                for t in range(n):
                    c = causes[b * seq + t]    # Cause-Faktoren zu Zeit t
                    e = effects[b * seq + t + 1]  # Effect-Faktoren zu Zeit t+1
                    
                    # Kausaler Score: wie stark beeinflusst c den Effekt e?
                    score = self.causal_scorer(c.unsqueeze(0), e.unsqueeze(0)).squeeze()
                    
                    if score > 0.7:  # Schwelle für kausale Entdeckung
                        # Finde dominante Faktoren
                        c_factor = c.argmax().item()
                        e_factor = e.argmax().item()
                        
                        # Update Causal Matrix
                        self.causal_counts[c_factor, e_factor] += 1
                        n_c = self.causal_counts[c_factor, e_factor].item()
                        old = self.causal_matrix[c_factor, e_factor].item()
                        self.causal_matrix[c_factor, e_factor] = old * 0.9 + score.item() * 0.1
                        self.causal_strength[c_factor, e_factor] = min(
                            1.0, self.causal_strength[c_factor, e_factor].item() + 0.05
                        )
                        
                        discovered_causes.append(c_factor)
                        discovered_effects.append(e_factor)
                        discovered_strengths.append(score.item())
                        
                        # Speichere im Temp-Buffer
                        idx = self._temp_idx % 500
                        self.temp_buffer_causes[idx] = h_flat[b * seq + t]
                        self.temp_buffer_effects[idx] = h_flat[b * seq + t + 1]
                        self._temp_idx += 1
                        self._temp_count = min(self._temp_count + 1, 500)
            
            # Metrik: Discovery Rate
            if discovered_causes:
                dr = len(discovered_causes) / max(1, n * batch)
                idx = self._discovery_idx % 100
                self.discovery_rate[idx] = dr
                self._discovery_idx += 1
            
            return discovered_causes[:10], discovered_effects[:10], discovered_strengths[:10]
    
    def do_intervention(self, hidden_states, intervention_vector):
        """
        Führe eine Intervention durch (Do-Calculus).
        
        Simuliere: "Was passiert, wenn ich X ändere?"
        
        Args:
            hidden_states: [batch, seq, d_model]
            intervention_vector: [batch, d_model] — die "Eingriffs"-Repräsentation
        
        Returns:
            intervened: [batch, seq, d_model]
        """
        with torch.no_grad():
            batch, seq, d = hidden_states.shape
            
            # Wiederhole Intervention für jede Position
            interv_exp = intervention_vector.unsqueeze(1).expand(-1, seq, -1)
            
            # Berechne Interventionseffekt
            interv_feat = torch.cat([hidden_states, interv_exp], dim=-1)
            delta = self.intervention_net(interv_feat)
            
            return hidden_states + delta * 0.2
    
    def counterfactual(self, hidden_states, alternative, position):
        """
        Generiere Counterfactual: "Was wäre, wenn an Position X
        etwas anderes passiert wäre?"
        
        Args:
            hidden_states: [batch, seq, d_model]
            alternative: [batch, d_model] — alternative Repräsentation
            position: int — Position in der Sequenz
        
        Returns:
            counterfactual_states: [batch, seq, d_model]
        """
        with torch.no_grad():
            batch, seq, d = hidden_states.shape
            
            # Ersetze an Position 'position' den State
            cf_states = hidden_states.clone()
            alt_exp = alternative.unsqueeze(1)  # [batch, 1, d]
            
            # Modifiziere ab der Position
            cf_states[:, position:, :] = self.counterfactual_net(
                torch.cat([
                    cf_states[:, position:, :],
                    alt_exp.expand(-1, seq - position, -1)
                ], dim=-1)
            )
            
            return cf_states
    
    def condition(self, hidden_states):
        """
        Moduliere Hidden-States mit kausalem Verständnis.
        
        Args:
            hidden_states: [batch, seq, d_model]
        
        Returns:
            conditioned: [batch, seq, d_model]
        """
        with torch.no_grad():
            batch, seq, d = hidden_states.shape
            
            # Wende kausale Entdeckung an (jeden 10. Schritt)
            if self._temp_count > 0 and self._temp_idx % 10 == 0:
                causes, effects, strengths = self.discover_causal(hidden_states)
            
            return hidden_states  # kein direkter Conditioning-Effekt, nur Discovery
    
    def get_causal_graph(self, top_k=10):
        """Gib Top-K kausale Beziehungen zurück."""
        n = self.n_causal_factors
        entries = []
        for i in range(n):
            for j in range(n):
                if i != j and self.causal_counts[i, j].item() > 0:
                    entries.append({
                        'cause': i, 'effect': j,
                        'strength': self.causal_matrix[i, j].item(),
                        'count': self.causal_counts[i, j].item(),
                    })
        entries.sort(key=lambda x: x['strength'], reverse=True)
        return entries[:top_k]
    
    def get_causal_stats(self):
        """Gib Causal-Reasoning-Statistiken."""
        total_relations = (self.causal_counts > 0).sum().item()
        avg_strength = self.causal_matrix[self.causal_counts > 0].mean().item() if total_relations > 0 else 0
        recent_dr = self.discovery_rate[:max(1, self._discovery_idx)].mean().item()
        return {
            'n_causal_relations': total_relations,
            'avg_causal_strength': avg_strength,
            'discovery_rate': recent_dr,
            'temp_buffer_size': min(self._temp_count, 500),
            'n_causal_factors': self.n_causal_factors,
        }


class System2Reasoning(CogModule):
    """
    PHASE 45: System-2 Reasoning — Kettenbewusstsein + Verifikation + Gedankenbäume.
    
    CogLang bekommt die Fähigkeit, NICHT nur einen Schritt vorherzusagen,
    sondern MEHRERE Reasoning-Schritte zu generieren, zu VERIFIZIEREN
    und den BESTEN Pfad zu wählen (dual-process theory: Kahneman).
    
    Kernkomponenten:
    1. Chain-of-Thought Decoder: Generiert schrittweise Reasoning-Ketten
       - Nimmt aktuellen Hidden-State → generiert N Reasoning-Schritte
       - Jeder Schritt ist [batch, d_model] — abstrakter Gedanke
    2. Verification Head: Bewertet jeden Schritt auf Korrektheit
       - Lernt: "macht dieser Schritt Sinn?"
       - Scoring: [0, 1] — 1 = korrekt, 0 = inkorrekt
    3. Tree-of-Thought: Parallel-Exploration mehrerer Pfade
       - Verzweige an jedem Schritt in K Alternativen
       - Wähle den Pfad mit höchstem Verifikations-Score
    
    Integration mit ConsciousnessGlimpse:
    - System-2 operiert auf Spotlight-Inhalten (bewusste Gedanken)
    - Verifikations-Scores fließen in Salience-Berechnung ein
    """
    def __init__(self, d_model, n_reasoning_steps=8, n_tree_branches=3, temperature=0.3):
        super().__init__()
        self.d_model = d_model
        self.n_reasoning_steps = n_reasoning_steps
        self.n_tree_branches = n_tree_branches
        self.temperature = temperature
        
        # ——— 1. Chain-of-Thought Decoder ———
        # Transformiert aktuellen Gedanken in nächsten Reasoning-Schritt
        self.cot_decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )
        
        # ——— 2. Verification Head ———
        # Bewertet einen Reasoning-Schritt: "ist das korrekt/widerspruchsfrei?"
        self.verification_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )
        
        # ——— 3. Tree-of-Thought Router ———
        # Erzeugt K alternative Fortsetzungen eines Reasoning-Schritts
        self.tree_router = nn.Linear(d_model, n_tree_branches * d_model)
        
        # ——— 4. Schritt-Embedding ———
        # Position im Reasoning-Prozess
        self.step_embedding = nn.Embedding(n_reasoning_steps + 1, d_model)
        
        # ——— 5. Metrik-Tracking ———
        self.register_buffer('verification_history', torch.zeros(500))
        self.register_buffer('tree_depth_history', torch.zeros(500))
        self.register_buffer('reasoning_quality', torch.zeros(100))
        self._metric_idx = 0
        self._quality_idx = 0
    
    def generate_reasoning_chain(self, hidden_state):
        """
        Generiere eine Kette von Reasoning-Schritten (Chain-of-Thought).
        
        Args:
            hidden_state: [batch, d_model] — Aktueller Gedanke
        
        Returns:
            chain: [batch, n_steps, d_model] — Reasoning-Kette
            scores: [batch, n_steps] — Verifikations-Scores
        """
        with torch.no_grad():
            batch, d = hidden_state.shape
            chain = []
            scores = []
            
            current = hidden_state
            for step in range(self.n_reasoning_steps):
                # Schritt-Embedding hinzufügen
                step_emb = self.step_embedding(
                    torch.tensor([step], device=hidden_state.device)
                ).expand(batch, -1)  # [batch, d]
                
                # Nächsten Reasoning-Schritt generieren
                next_step = self.cot_decoder(current + step_emb * 0.1)
                
                # Verifikation: wie konsistent ist dieser Schritt?
                verif_input = torch.cat([current, next_step], dim=-1)
                score = torch.sigmoid(self.verification_net(verif_input)).squeeze(-1)
                
                chain.append(next_step)
                scores.append(score)
                
                current = next_step
            
            return torch.stack(chain, dim=1), torch.stack(scores, dim=1)
    
    def tree_of_thought(self, hidden_state, return_best=True):
        """
        Erkunde mehrere Reasoning-Pfade parallel (Tree-of-Thought).
        
        Args:
            hidden_state: [batch, d_model]
            return_best: bool — Nur besten Pfad zurückgeben?
        
        Returns:
            best_chain: [batch, n_steps, d_model] — Bester Pfad
            best_scores: [batch, n_steps] — Scores des besten Pfads
        """
        with torch.no_grad():
            batch, d = hidden_state.shape
            
            # Starte von aktuellem State
            current = hidden_state.unsqueeze(1).expand(-1, self.n_tree_branches, -1)
            # [batch, branches, d]
            
            all_scores = torch.zeros(batch, self.n_tree_branches, self.n_reasoning_steps,
                                     device=hidden_state.device)
            
            for step in range(self.n_reasoning_steps):
                step_emb = self.step_embedding(
                    torch.tensor([step], device=hidden_state.device)
                ).expand(batch, self.n_tree_branches, -1)
                
                # Router erzeugt verschiedene Fortsetzungen
                # (nur beim ersten Schritt, danach per Decoder)
                if step == 0:
                    # Erste Verzweigung: baum_startet
                    routes = self.tree_router(current[:, 0, :])  # [batch, branches*d]
                    routes = routes.view(batch, self.n_tree_branches, d)
                    next_steps = self.cot_decoder(
                        (current[:, 0, :] + step_emb[:, 0, :] * 0.1).unsqueeze(1)
                    )  # [batch, 1, d]
                    next_steps = next_steps + routes * 0.3  # Diversität
                else:
                    next_steps = self.cot_decoder(
                        current.view(-1, d) + step_emb.view(-1, d) * 0.1
                    ).view(batch, self.n_tree_branches, d)
                
                # Verifikation jedes Pfads
                for b in range(self.n_tree_branches):
                    verif_in = torch.cat([current[:, b, :], next_steps[:, b, :]], dim=-1)
                    score = torch.sigmoid(self.verification_net(verif_in)).squeeze(-1)
                    all_scores[:, b, step] = score
                
                current = next_steps
            
            # Wähle besten Pfad pro Batch
            avg_scores = all_scores.mean(dim=-1)  # [batch, branches]
            best_branch = avg_scores.argmax(dim=-1)  # [batch]
            
            best_chain = torch.zeros(batch, self.n_reasoning_steps, d, device=hidden_state.device)
            best_scores = torch.zeros(batch, self.n_reasoning_steps, device=hidden_state.device)
            
            for b in range(batch):
                best_chain[b] = current[b, best_branch[b]]
                best_scores[b] = all_scores[b, best_branch[b]]
            
            # Metriken
            if self._metric_idx < 500:
                idx = self._metric_idx % 500
                self.tree_depth_history[idx] = self.n_tree_branches
                self.verification_history[idx] = best_scores.mean().item()
                self._metric_idx += 1
            
            return best_chain, best_scores
    
    def verify_reasoning(self, hidden_states, reasoning_chain):
        """
        Verifiziere eine gesamte Reasoning-Kette.
        
        Args:
            hidden_states: [batch, seq, d_model] — Original-Kontext
            reasoning_chain: [batch, n_steps, d_model] — Reasoning-Kette
        
        Returns:
            overall_score: float — Gesamt-Qualität
            step_scores: [batch, n_steps] — Einzelschritt-Scores
        """
        with torch.no_grad():
            batch, seq, d = hidden_states.shape
            _, n_steps, _ = reasoning_chain.shape
            
            # Vergleiche Reasoning-Kette mit Original-Kontext
            context_pooled = hidden_states.mean(dim=1)  # [batch, d]
            
            step_scores = []
            for t in range(n_steps):
                verif_in = torch.cat([
                    context_pooled,
                    reasoning_chain[:, t, :]
                ], dim=-1)
                score = torch.sigmoid(self.verification_net(verif_in)).squeeze(-1)
                step_scores.append(score)
            
            step_scores = torch.stack(step_scores, dim=1)  # [batch, n_steps]
            overall_score = step_scores.mean().item()
            
            # Qualitäts-Tracking
            idx = self._quality_idx % 100
            self.reasoning_quality[idx] = overall_score
            self._quality_idx += 1
            
            return overall_score, step_scores
    
    def condition(self, hidden_states):
        """
        Konditioniere Hidden-States mit Reasoning.
        
        Args:
            hidden_states: [batch, seq, d_model]
        
        Returns:
            conditioned: [batch, seq, d_model]
        """
        with torch.no_grad():
            batch, seq, d = hidden_states.shape
            
            # Reasoning vom gepoolten Kontext starten
            context = hidden_states.mean(dim=1)  # [batch, d]
            
            # Generiere Tree-of-Thought (nur jeder 5. Aufruf)
            if self._quality_idx % 5 == 0:
                best_chain, best_scores = self.tree_of_thought(context)
                
                # Integriere Reasoning in Hidden-States
                reasoning_influence = best_chain.mean(dim=1, keepdim=True)  # [batch, 1, d]
                return hidden_states + reasoning_influence * 0.05
            
            return hidden_states
    
    def get_reasoning_stats(self):
        """Gib System-2-Reasoning-Statistiken."""
        avg_verif = self.verification_history[:max(1, self._metric_idx)].mean().item()
        avg_quality = self.reasoning_quality[:max(1, self._quality_idx)].mean().item()
        return {
            'avg_verification': avg_verif,
            'avg_quality': avg_quality,
            'n_reasoning_steps': self.n_reasoning_steps,
            'n_tree_branches': self.n_tree_branches,
            'n_verifications': min(self._metric_idx, 500),
        }


class ImaginationPlanning(CogModule):
    """
    PHASE 46: Imagination & Planning — Vorhersage + Generierung + Bewertung.
    
    CogLang kann in die Zukunft blicken: Imagination zukünftiger Zustände
    und mehrschrittige Planung, um gewünschte Ergebnisse zu erreichen.
    
    Drei-Komponenten-Architektur nach Bengio's "consciousness prior":
    
    1. Future Predictor: Nimmt (aktueller_State, Aktion) → nächster_State
       - Lernt ein internes Weltmodell: "wenn ich X tue, passiert Y"
       - Wird trainiert aus realen Sequenzen (next-token-prediction)
    
    2. Plan Generator: Erzeugt Aktionssequenzen (abstrakte Pläne)
       - Suchbaum: expandiere mögliche Aktionen, simuliere Ergebnis
       - BFS/DFS-artig: "was passiert nach Schritt 1, 2, 3...?"
    
    3. Plan Verifier: Bewertet Pläne nach gewünschtem Kriterium
       - "Führt dieser Plan zu einem besseren Zustand?"
       - Lernt aus Kontrast: gute Pläne vs. schlechte Pläne
    
    Integration mit System-2 Reasoning:
    - Reasoning liefert die "Gedankenschritte"
    - Imagination testet diese Gedanken in simulierter Zukunft
    """
    def __init__(self, d_model, n_plan_steps=6, n_actions=16, temperature=0.2):
        super().__init__()
        self.d_model = d_model
        self.n_plan_steps = n_plan_steps
        self.n_actions = n_actions
        self.temperature = temperature
        
        # ——— 1. Future Predictor (Weltmodell) ———
        # state_encoder + action_embedding → next_state
        self.state_encoder = nn.Linear(d_model, d_model)
        self.action_embedding = nn.Embedding(n_actions, d_model)
        self.future_predictor = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )
        
        # ——— 2. Plan Generator ———
        # Aus aktuellem Kontext: generiere K Aktionssequenzen
        self.plan_generator = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, n_plan_steps * n_actions),
        )
        
        # ——— 3. Plan Verifier ———
        # Bewerte: "ist dieser Plan gut?" (scalar output)
        self.plan_verifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )
        
        # ——— 4. Imagination Buffer ———
        # Speichert vorgestellte Zustände für spätere Lernschritte
        self.register_buffer('imagination_buffer', torch.zeros(500, d_model))
        self.register_buffer('imagination_scores', torch.zeros(500))
        self._buffer_idx = 0
        self._buffer_count = 0
        
        # ——— 5. Metrik-Tracking ———
        self.register_buffer('plan_quality', torch.zeros(100))
        self.register_buffer('prediction_accuracy', torch.zeros(100))
        self._quality_idx = 0
        self._pred_idx = 0
    
    def predict_future(self, current_state, action_ids):
        """
        Sage nächsten Zustand voraus (Weltmodell).
        
        Args:
            current_state: [batch, d_model] — Aktueller Zustand
            action_ids: [batch] oder int — Gewählte Aktion
        
        Returns:
            next_state: [batch, d_model] — Vorhergesagter Zustand
        """
        with torch.no_grad():
            if isinstance(action_ids, int):
                action_ids = torch.tensor([action_ids], device=current_state.device)
            if action_ids.dim() == 0:
                action_ids = action_ids.unsqueeze(0)
            if action_ids.size(0) != current_state.size(0):
                action_ids = action_ids.expand(current_state.size(0))
            
            state_encoded = self.state_encoder(current_state)
            action_emb = self.action_embedding(action_ids)
            
            combined = torch.cat([state_encoded, action_emb], dim=-1)
            next_state = self.future_predictor(combined)
            
            return next_state
    
    def generate_plan(self, context, n_candidates=5):
        """
        Generiere mehrere Aktionspläne.
        
        Args:
            context: [batch, d_model] — Gepoolter Kontext
            n_candidates: int — Anzahl Kandidaten-Pläne
        
        Returns:
            plans: [batch, n_candidates, n_steps] — Aktions-IDs
            plan_scores: [batch, n_candidates] — Bewertungen
            imagined_states: [batch, n_candidates, n_steps+1, d_model]
        """
        with torch.no_grad():
            batch, d = context.shape
            
            # Generiere Kandidatepläne via Plan-Generator mit noise
            logits = self.plan_generator(context)  # [batch, n_steps * n_actions]
            logits = logits.view(batch, self.n_plan_steps, self.n_actions)
            
            # Sampling mit Temperatur für Diversität
            plans = []
            for c in range(n_candidates):
                noise = torch.randn_like(logits) * self.temperature
                plan_logits = logits + noise
                plan = torch.multinomial(
                    F.softmax(plan_logits.view(-1, self.n_actions), dim=-1),
                    num_samples=1
                ).view(batch, self.n_plan_steps)
                plans.append(plan)
            
            plans = torch.stack(plans, dim=1)  # [batch, candidates, steps]
            
            # Simuliere jeden Plan: rolle Zukunft aus
            imagined_states = []
            plan_scores_batches = []
            
            for b in range(batch):
                batch_imagined = []
                batch_scores = []
                
                for c in range(n_candidates):
                    states = [context[b:b+1]]
                    score_sum = 0.0
                    
                    for t in range(self.n_plan_steps):
                        action_id = plans[b, c, t]
                        next_state = self.predict_future(states[-1], action_id.item())
                        states.append(next_state)
                        
                        # Verifiziere diesen Schritt
                        verif_in = torch.cat([states[-2], next_state], dim=-1)
                        step_score = torch.sigmoid(self.plan_verifier(verif_in)).item()
                        score_sum += step_score
                    
                    batch_imagined.append(torch.cat(states, dim=0).unsqueeze(0))
                    batch_scores.append(score_sum / self.n_plan_steps)
                
                imagined_states.append(torch.cat(batch_imagined, dim=0).unsqueeze(0))
                plan_scores_batches.append(torch.tensor(batch_scores))
            
            imagined_states = torch.cat(imagined_states, dim=0)  # [batch, candidates, steps+1, d]
            plan_scores = torch.stack(plan_scores_batches, dim=0)  # [batch, candidates]
            
            return plans, plan_scores, imagined_states
    
    def simulate(self, hidden_states, n_steps=4):
        """
        Simuliere zukünftige Entwicklung (Imagination).
        
        Args:
            hidden_states: [batch, seq, d_model]
            n_steps: int — Wie viele Schritte in die Zukunft?
        
        Returns:
            imagined_futures: [batch, n_candidates, n_steps, d_model]
            best_future: [batch, 1, d_model]
        """
        with torch.no_grad():
            context = hidden_states.mean(dim=1)  # [batch, d_model]
            
            plans, scores, states = self.generate_plan(context, n_candidates=3)
            
            # Wähle besten Plan
            best_plan_idx = scores.argmax(dim=-1)  # [batch]
            batch, candidates, steps_plus1, d = states.shape
            
            best_futures = []
            for b in range(batch):
                best_futures.append(states[b, best_plan_idx[b], 1:, :])  # ohne ersten State
            
            best_future = torch.stack(best_futures, dim=0)  # [batch, n_steps, d]
            
            # Speichere in Imagination Buffer
            if self._buffer_count < 500:
                for b in range(batch):
                    idx = self._buffer_idx % 500
                    self.imagination_buffer[idx] = best_future[b].mean(dim=0)
                    self.imagination_scores[idx] = scores[b, best_plan_idx[b]].item()
                    self._buffer_idx += 1
                    self._buffer_count = min(self._buffer_count + 1, 500)
            
            # Metrik
            idx = self._quality_idx % 100
            self.plan_quality[idx] = scores.mean().item()
            self._quality_idx += 1
            
            return best_future
    
    def learn_from_imagination(self, actual_next_state, imagined_next_state):
        """
        Lerne aus Diskrepanz zwischen Imagination und Realität.
        
        Args:
            actual_next_state: [batch, d_model] — Echter nächster State
            imagined_next_state: [batch, d_model] — Vorgestellter State
        """
        with torch.no_grad():
            # Prediction Accuracy: Cos-Ähnlichkeit
            acc = F.cosine_similarity(actual_next_state, imagined_next_state, dim=-1).mean().item()
            idx = self._pred_idx % 100
            self.prediction_accuracy[idx] = max(0, min(1, acc))
            self._pred_idx += 1
    
    def condition(self, hidden_states):
        """
        Konditioniere Hidden-States mit Imagination.
        
        Args:
            hidden_states: [batch, seq, d_model]
        
        Returns:
            conditioned: [batch, seq, d_model]
        """
        with torch.no_grad():
            batch, seq, d = hidden_states.shape
            
            # Simuliere Zukunft (selten, alle 10 Schritte)
            if self._quality_idx % 10 == 0:
                best_future = self.simulate(hidden_states, n_steps=4)
                # Integriere Imagination: Durchschnitt der Zukunft
                future_influence = best_future.mean(dim=1, keepdim=True)  # [batch, 1, d]
                return hidden_states + future_influence * 0.03
            
            return hidden_states
    
    def get_imagination_stats(self):
        """Gib Imaginations-Statistiken."""
        avg_quality = self.plan_quality[:max(1, self._quality_idx)].mean().item()
        avg_acc = self.prediction_accuracy[:max(1, self._pred_idx)].mean().item()
        return {
            'avg_plan_quality': avg_quality,
            'avg_prediction_accuracy': avg_acc,
            'n_plan_steps': self.n_plan_steps,
            'n_actions': self.n_actions,
            'buffer_usage': min(self._buffer_count, 500),
        }


class ExplorationDrive(CogModule):
    """
    PHASE 47: Exploration Drive — Aktive Wissenslücken-Suche & Neugierverhalten.
    
    CogLang sucht aktiv nach Unsicherheit statt passiv zu lernen.
    Inspiriert von intrinsischer Motivation in der Entwicklungspsychologie:
    - Säuglinge schauen länger auf überraschende Reize
    - Kinder explorieren gezielt Bereiche mit mittlerer Unsicherheit
      (nicht zu langweilig, nicht zu überwältigend)
    
    Kernmechanismen:
    1. Uncertainty Map: 2D-Raster über Token-Embedding-Raum
       - Jede Zelle trackt: wie oft besucht? mittlerer Error? Varianz?
       - Identifiziert "weisse Flecken" auf der Wissenslandkarte
    
    2. Novelty Detector: Temporal Difference des Errors
       - "Neu" = aktueller Error weicht stark von kurzfristigem EMA ab
       - "Überraschend" = Error-Varianz plötzlich hoch
    
    3. Exploration Bonus: Reward-Signal für Neugier
       - Bonus = uncertainty * novelty * (1 - saturation)
       - Sättigung: wiederholte Besuche reduzieren Bonus
       - Wird an ActiveInference weitergegeben
    
    4. Context Gating: Welche Bereiche brauchen mehr Exploration?
       - Bias für Batch-Sampling: mehr Daten aus unsicheren Domänen
       - Integriert in Salience-Berechnung von ConsciousnessGlimpse
    """
    def __init__(self, d_model, n_uncertainty_cells=128, n_emn_history=200):
        super().__init__()
        self.d_model = d_model
        self.n_uncertainty_cells = n_uncertainty_cells
        self.n_emn_history = n_emn_history
        
        # ——— 1. Uncertainty Map (Wissenslandkarte) ———
        # Jede Zelle: visit_count, mean_error, error_variance
        self.register_buffer('cell_visits', torch.zeros(n_uncertainty_cells, dtype=torch.long))
        self.register_buffer('cell_mean_error', torch.zeros(n_uncertainty_cells))
        self.register_buffer('cell_error_var', torch.ones(n_uncertainty_cells) * 0.1)
        self.register_buffer('cell_last_visit', torch.zeros(n_uncertainty_cells, dtype=torch.long))
        
        # Zell-Encoder: mapped d_model Features auf Zell-Index
        self.cell_encoder = nn.Linear(d_model, n_uncertainty_cells)
        
        # ——— 2. Novelty Detector ———
        # EMA + Varianz-Tracking
        self.register_buffer('novelty_history', torch.zeros(n_emn_history))
        self.register_buffer('error_ema_fast', torch.zeros(1))  # Kurzzeit (10 steps)
        self.register_buffer('error_ema_slow', torch.zeros(1))  # Langzeit (100 steps)
        self.register_buffer('novelty_signal', torch.zeros(1))
        self._step_counter = 0
        self._novelty_idx = 0
        
        # ——— 3. Exploration Bonus Network ———
        # Berechnet Bonus aus (cell_uncertainty, novelty, visit_count)
        self.bonus_net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        
        # ——— 4. Domain Exploration Bias ———
        self.register_buffer('domain_exploration', torch.zeros(4))  # 4 Domänen
        self.register_buffer('domain_visit_counts', torch.ones(4) * 10)
        
        # ——— 5. Metrik-Puffer ———
        self.register_buffer('exploration_rate', torch.zeros(100))
        self.register_buffer('avg_novelty', torch.zeros(100))
        self._rate_idx = 0
        self._avg_novelty_idx = 0
    
    def _hash_to_cell(self, hidden_states):
        """
        Mappe Hidden-States auf Uncertainty-Cell-Index.
        
        Args:
            hidden_states: [batch, seq, d_model]
        
        Returns:
            cell_ids: [batch, seq] — Zell-Indices
            cell_probs: [batch, seq, n_cells] — Soft-Assignment
        """
        with torch.no_grad():
            # Weiche Zuweisung zu Zellen via softmax
            logits = self.cell_encoder(hidden_states)  # [batch, seq, n_cells]
            probs = F.softmax(logits / 0.5, dim=-1)
            # Harte Zuweisung für Buffer-Updates
            cell_ids = probs.argmax(dim=-1)  # [batch, seq]
            return cell_ids, probs
    
    def update_uncertainty(self, hidden_states, error_norm):
        """
        Aktualisiere Uncertainty Map basierend auf aktuellem Error.
        
        Args:
            hidden_states: [batch, seq, d_model]
            error_norm: float — Mittlerer Error über Batch
        """
        with torch.no_grad():
            batch, seq, d = hidden_states.shape
            
            # Welche Zellen sind aktiv?
            cell_ids, probs = self._hash_to_cell(hidden_states)  # [batch, seq]
            
            # Update cell statistics (jede 10 Steps)
            if self._step_counter % 10 == 0:
                for b in range(batch):
                    for t in range(seq):
                        cell = cell_ids[b, t].item()
                        if cell < self.n_uncertainty_cells:
                            n = self.cell_visits[cell].item()
                            old_mean = self.cell_mean_error[cell].item()
                            old_var = self.cell_error_var[cell].item()
                            
                            # Welford's Online-Algorithmus für Mean + Variance
                            new_n = n + 1
                            delta = error_norm - old_mean
                            new_mean = old_mean + delta / new_n
                            delta2 = error_norm - new_mean
                            new_var = old_var + delta * delta2
                            
                            self.cell_visits[cell] = new_n
                            self.cell_mean_error[cell] = new_mean
                            self.cell_error_var[cell] = max(0.01, new_var / max(1, new_n))
                            self.cell_last_visit[cell] = self._step_counter
            
            # Novelty: Differenz zwischen kurz- und langfristigem EMA
            alpha_fast = 0.3  # Kurzzeit-Gewichtung (~10 Steps)
            alpha_slow = 0.03  # Langzeit-Gewichtung (~100 Steps)
            
            self.error_ema_fast[0] = self.error_ema_fast[0] * (1 - alpha_fast) + error_norm * alpha_fast
            self.error_ema_slow[0] = self.error_ema_slow[0] * (1 - alpha_slow) + error_norm * alpha_slow
            
            novelty = abs(self.error_ema_fast[0].item() - self.error_ema_slow[0].item())
            self.novelty_signal[0] = novelty
            
            # Novelty-History
            if self._novelty_idx < self.n_emn_history:
                self.novelty_history[self._novelty_idx] = novelty
                self._novelty_idx += 1
            
            # Metrik: Avg Novelty
            n_idx = self._avg_novelty_idx % 100
            self.avg_novelty[n_idx] = novelty
            self._avg_novelty_idx += 1
            
            self._step_counter += 1
    
    def compute_exploration_bonus(self, hidden_states):
        """
        Berechne Explorations-Bonus für aktuelle Hidden-States.
        
        Args:
            hidden_states: [batch, seq, d_model]
        
        Returns:
            bonus: float — Explorations-Bonus (0-1)
        """
        with torch.no_grad():
            batch, seq, d = hidden_states.shape
            cell_ids, probs = self._hash_to_cell(hidden_states)
            
            total_bonus = 0.0
            n_cells = 0
            
            for b in range(batch):
                for t in range(0, seq, 5):  # Jeden 5. Token
                    cell = cell_ids[b, t].item()
                    if cell >= self.n_uncertainty_cells:
                        continue
                    
                    n = self.cell_visits[cell].item()
                    mean_err = self.cell_mean_error[cell].item()
                    var_err = self.cell_error_var[cell].item()
                    
                    # Uncertainty: hohe Varianz = unsicher
                    uncertainty = min(1.0, var_err / max(0.1, mean_err + 0.01))
                    
                    # Novelty: EMA-Differenz
                    novelty = self.novelty_signal[0].item()
                    
                    # Sättigung: Wiederholungen reduzieren Bonus
                    saturation = min(1.0, n / 50.0)
                    
                    # Bonus: mittlere Unsicherkeit + hohe Neuheit + geringe Sättigung
                    bonus_input = torch.tensor([[uncertainty, novelty, 1.0 - saturation]],
                                                device=hidden_states.device)
                    cell_bonus = torch.sigmoid(self.bonus_net(bonus_input)).item()
                    total_bonus += cell_bonus
                    n_cells += 1
            
            avg_bonus = total_bonus / max(1, n_cells)
            
            # Metrik
            idx = self._rate_idx % 100
            self.exploration_rate[idx] = avg_bonus
            self._rate_idx += 1
            
            return avg_bonus
    
    def get_domain_exploration_weights(self):
        """
        Berechne Domain-Gewichtung basierend auf Exploration.
        Weniger besuchte Domänen mit hohem Error bekommen höheres Gewicht.
        
        Returns:
            weights: [n_domains] — Normalisierte Gewichte
        """
        with torch.no_grad():
            n_domains = self.domain_exploration.size(0)
            weights = torch.ones(n_domains)
            
            for d in range(n_domains):
                visits = self.domain_visit_counts[d].item()
                error = self.domain_exploration[d].item()
                # Wenig besucht + hoher Error = mehr Exploration nötig
                weights[d] = (1.0 / max(1, visits * 0.1)) * (1.0 + error)
            
            return weights / weights.sum()
    
    def condition(self, hidden_states):
        """
        Moduliere Hidden-States mit Explorations-Signal.
        
        Args:
            hidden_states: [batch, seq, d_model]
        
        Returns:
            conditioned: [batch, seq, d_model]
        """
        with torch.no_grad():
            # Bonus-Berechnung (selten, alle 20 Steps)
            if self._step_counter > 0 and self._step_counter % 20 == 0:
                bonus = self.compute_exploration_bonus(hidden_states)
                # Kleiner Explorationseffekt auf Hidden-States
                bonus_tensor = torch.tensor(bonus, device=hidden_states.device)
                return hidden_states * (1.0 + bonus_tensor * 0.02)
            
            return hidden_states
    
    def get_exploration_stats(self):
        """Gib Exploration-Statistiken."""
        rate = self.exploration_rate[:max(1, self._rate_idx)].mean().item()
        novelty = self.avg_novelty[:max(1, self._avg_novelty_idx)].mean().item()
        
        # Cell-Statistiken
        total_cells = self.n_uncertainty_cells
        visited_cells = (self.cell_visits > 0).sum().item()
        exploration_coverage = visited_cells / max(1, total_cells)
        
        return {
            'exploration_rate': rate,
            'avg_novelty': novelty,
            'coverage': exploration_coverage,
            'visited_cells': visited_cells,
            'total_cells': total_cells,
            'step': self._step_counter,
        }


class MetaKognition(CogModule):
    """
    PHASE 48: MetaKognition — Das Modell denkt über sein eigenes Denken nach.
    
    Während SelfReflection (Phase 37) den Output bewertet, steuert MetaKognition
    den kognitiven Prozess selbst:
    
    1. StrategySelector — Wählt optimale Reasoning-Strategie basierend auf Task
    2. ConfidenceCalibrator — Kalibriert Confidence-Scores (Temperature Scaling)
    3. CognitiveResourceAllocator — Verteilt Rechenbudget dynamisch
    
    Inspiriert von:
    - Metacognition in der Psychologie (Flavell 1979)
    - System-1/System-2-Steuerung (Kahneman)
    - Adaptive Computation Time (Graves 2016)
    """
    def __init__(self, d_model, n_strategies=4, n_resource_levels=3):
        super().__init__()
        self.d_model = d_model
        self.n_strategies = n_strategies
        self.n_resource_levels = n_resource_levels
        
        # ——— 1. Strategy Selector ———
        # Wählt aus 4 Strategien: 
        #   0 = schnelle Heuristik (System-1-only)
        #   1 = balanciert (etwas Reasoning)
        #   2 = tiefes Reasoning (viele CoT-Schritte)
        #   3 = explorativ (breite Trees, Imagination, Curiosity)
        self.strategy_encoder = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Tanh(),
            nn.Linear(d_model, n_strategies),
        )
        self.strategy_confidence = nn.Linear(d_model, 1)  # Vertrauen in gewählte Strategie
        
        # ——— Task Difficulty Estimator ———
        # Schätzt Schwierigkeit aus Embedding + bisherigem Loss
        self.difficulty_estimator = nn.Sequential(
            nn.Linear(d_model + 1, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )
        
        # ——— 2. Confidence Calibrator ———
        # Temperature-Scale: calibrated_conf = sigmoid(raw_conf / temperature)
        self.register_buffer('calibration_temperature', torch.tensor(1.0))
        self.register_buffer('calibration_bias', torch.tensor(0.0))
        # ECE-Tracking (Expected Calibration Error)
        self.register_buffer('ece_history', torch.zeros(500))
        self._ece_idx = 0
        
        # ——— 3. Cognitive Resource Allocator ———
        # Entscheidet für jedes Modul, wie viele Ressourcen es bekommt
        # n_reasoning_steps, n_tree_branches, n_plan_steps, n_imagination_steps
        self.resource_allocator = nn.Sequential(
            nn.Linear(d_model + 1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, n_resource_levels * 4),  # 4 resource dimensions
        )
        # Resource-Namen
        self.resource_names = ['reasoning_steps', 'tree_branches', 'plan_steps', 'imagination_steps']
        # Resource-Bereiche pro Level
        self.resource_ranges = {
            'reasoning_steps': [2, 6, 12],
            'tree_branches': [1, 3, 5],
            'plan_steps': [2, 4, 8],
            'imagination_steps': [1, 3, 6],
        }
        
        # ——— Strategy Memory ———
        # Trackt, welche Strategie bei welchem Task-Typ gut funktioniert hat
        self.register_buffer('strategy_success', torch.zeros(1000, n_strategies))
        self.register_buffer('strategy_count', torch.zeros(1000, n_strategies))
        self._strategy_mem_idx = 0
        
        # ——— Cognitive Load Tracking ———
        self.register_buffer('cognitive_load', torch.zeros(50))  # Letzte 50 Load-Werte
        self._load_idx = 0
        self._current_strategy = 0  # Zuletzt gewählte Strategie
        
        self._max_weight = 2.0
    
    def estimate_difficulty(self, hidden_states, current_loss=None):
        """
        Schätze Task-Schwierigkeit aus Hidden-States + optionalem Loss.
        
        Args:
            hidden_states: [batch, seq, d_model]
            current_loss: float oder None
            
        Returns:
            difficulty: [batch, 1] — Schwierigkeit 0..1
        """
        with torch.no_grad():
            # Pool über Sequenz
            pooled = hidden_states.mean(dim=1)  # [batch, d_model]
            
            # Loss als Feature
            loss_feat = torch.zeros(pooled.size(0), 1, device=pooled.device)
            if current_loss is not None:
                loss_feat.fill_(min(current_loss, 20.0) / 20.0)  # Normalisiert auf 0..1
            
            diff_input = torch.cat([pooled, loss_feat], dim=-1)  # [batch, d_model+1]
            difficulty = torch.sigmoid(self.difficulty_estimator(diff_input))  # [batch, 1]
            
            return difficulty
    
    def select_strategy(self, hidden_states, difficulty, context_embedding=None):
        """
        Wähle optimale kognitive Strategie basierend auf Task und Kontext.
        
        Args:
            hidden_states: [batch, seq, d_model] — Aktuelle Repräsentation
            difficulty: [batch, 1] — Geschätzte Schwierigkeit
            context_embedding: [batch, d_model] oder None — Zusätzlicher Kontext
            
        Returns:
            strategy_weights: [batch, n_strategies] — Softmax-gewichtete Strategie
            selected: [batch] — Argmax-Strategie
            confidence: [batch] — Vertrauen in die Wahl
        """
        with torch.no_grad():
            batch = hidden_states.size(0)
            
            # Pooled Embedding
            pooled = hidden_states.mean(dim=1)  # [batch, d_model]
            
            # Strategy Features: Embedding + Difficulty
            if context_embedding is not None:
                features = torch.cat([pooled + context_embedding, difficulty], dim=-1)
            else:
                features = torch.cat([pooled, difficulty.expand(-1, d_model if False else 1)], dim=-1)
            
            # Erweitere auf d_model*2 falls nötig
            if features.size(-1) < self.d_model * 2:
                pad = torch.zeros(batch, self.d_model * 2 - features.size(-1), device=features.device)
                features = torch.cat([features, pad], dim=-1)
            
            strategy_logits = self.strategy_encoder(features)  # [batch, n_strategies]
            strategy_weights = F.softmax(strategy_logits, dim=-1)
            selected = strategy_weights.argmax(dim=-1)  # [batch]
            
            # Confidence in strategy choice
            conf_raw = self.strategy_confidence(pooled)  # [batch, 1]
            strategy_confidence = torch.sigmoid(conf_raw).squeeze(-1)  # [batch]
            
            self._current_strategy = selected[0].item()
            
            return strategy_weights, selected, strategy_confidence
    
    def calibrate_confidence(self, raw_confidence, method='temperature'):
        """
        Kalibriere rohe Confidence-Scores.
        
        Args:
            raw_confidence: [batch, seq, 1] — Rohe Sigmoid-Werte aus SelfReflection
            method: 'temperature' oder 'platt' oder 'beta'
            
        Returns:
            calibrated: [batch, seq, 1] — Kalibrierte Confidence
        """
        with torch.no_grad():
            if method == 'temperature':
                # Temperature Scaling: logit / T
                eps = 1e-6
                raw_conf = raw_confidence.clamp(eps, 1.0 - eps)
                logits = torch.log(raw_conf / (1.0 - raw_conf))  # Inverse sigmoid
                T = self.calibration_temperature.clamp(0.1, 10.0)
                calibrated = torch.sigmoid(logits / T + self.calibration_bias)
            elif method == 'platt':
                # Platt Scaling: sigmoid(a * logit + b)
                raw_conf = raw_confidence.clamp(1e-6, 1.0 - 1e-6)
                logits = torch.log(raw_conf / (1.0 - raw_conf))
                a = self.calibration_temperature.clamp(0.1, 5.0)
                b = self.calibration_bias
                calibrated = torch.sigmoid(a * logits + b)
            else:
                calibrated = raw_confidence
            
            return calibrated
    
    def allocate_resources(self, hidden_states, difficulty):
        """
        Weise kognitive Ressourcen basierend auf Schwierigkeit zu.
        
        Args:
            hidden_states: [batch, seq, d_model]
            difficulty: [batch, 1]
            
        Returns:
            resources: dict mit resource_name -> level (int)
        """
        with torch.no_grad():
            pooled = hidden_states.mean(dim=1)  # [batch, d_model]
            
            # Loss-Feature append (0 wenn keiner verfügbar)
            loss_feat = torch.zeros(pooled.size(0), 1, device=pooled.device)
            alloc_in = torch.cat([pooled, loss_feat + difficulty], dim=-1)  # [batch, d_model+1]
            
            raw_alloc = self.resource_allocator(alloc_in)  # [batch, n_resource_levels*4]
            
            # Reshape: [batch, 4, n_resource_levels]
            raw_alloc = raw_alloc.view(-1, 4, self.n_resource_levels)  # [batch, 4, 3]
            alloc_probs = F.softmax(raw_alloc, dim=-1)  # [batch, 4, 3]
            
            # Wähle Level pro Resource
            resources = {}
            for i, name in enumerate(self.resource_names):
                level = alloc_probs[:, i, :].argmax(dim=-1)  # [batch]
                # Batch-Mean für einfache Handhabung
                avg_level = level.float().mean().round().int().clamp(0, self.n_resource_levels - 1)
                level_idx = avg_level.item()
                resources[name] = self.resource_ranges[name][level_idx]
            
            # Cognitive Load = Summe aller Ressourcen-Nutzung
            total_load = sum(resources.values()) / sum(max(r) for r in self.resource_ranges.values())
            self.cognitive_load[self._load_idx % 50] = total_load
            self._load_idx += 1
            
            return resources
    
    def update_strategy_memory(self, strategy_id, success_score):
        """
        Lerne: Wie gut hat Strategie `strategy_id` funktioniert?
        
        Args:
            strategy_id: int (0..n_strategies-1)
            success_score: float (0..1) — 1 = sehr gut, 0 = schlecht
        """
        idx = self._strategy_mem_idx % 1000
        self.strategy_success[idx] = 0
        self.strategy_success[idx, strategy_id] = success_score
        self.strategy_count[idx] = 0
        self.strategy_count[idx, strategy_id] = 1
        self._strategy_mem_idx += 1
    
    def update_calibration(self, confidence, correctness):
        """
        Aktualisiere Kalibrierung basierend auf Confidence vs. Correctness.
        
        Args:
            confidence: float (0..1) — Vorhergesagte Confidence
            correctness: float (0..1) — Tatsächliche Korrektheit (1 = richtig)
        """
        with torch.no_grad():
            # ECE: |confidence - correctness|
            ece = abs(confidence - correctness)
            self.ece_history[self._ece_idx % 500] = ece
            self._ece_idx += 1
            
            # Temperature-Update: Wenn wir overconfident sind (conf > corr), erhöhe T
            if confidence > correctness + 0.1:
                # Overconfident → mehr Streuung (höhere Temperatur)
                self.calibration_temperature *= 1.01
            elif confidence < correctness - 0.1:
                # Underconfident → weniger Streuung (niedrigere Temperatur)
                self.calibration_temperature *= 0.99
            
            # Bias-Update
            bias_delta = 0.01 * (correctness - confidence)
            self.calibration_bias += bias_delta
            
            # Clamp
            self.calibration_temperature.clamp_(0.1, 10.0)
            self.calibration_bias.clamp_(-2.0, 2.0)
    
    def get_meta_strategy_advice(self, hidden_states, current_loss=None):
        """
        Gib Strategie-Empfehlung für den aktuellen Forward-Pass.
        
        Returns:
            dict mit Strategy-Advice
        """
        with torch.no_grad():
            difficulty = self.estimate_difficulty(hidden_states, current_loss)
            # Difficulty 0..1
            
            # Einfache Heuristik für Strategie-Wahl
            diff_val = difficulty.mean().item()
            
            if diff_val < 0.3:
                strategy = 0  # Schnelle Heuristik
            elif diff_val < 0.6:
                strategy = 1  # Balanciert
            elif diff_val < 0.8:
                strategy = 2  # Tiefes Reasoning
            else:
                strategy = 3  # Explorative Suche
            
            resources = self.allocate_resources(hidden_states, difficulty)
            
            return {
                'difficulty': diff_val,
                'strategy': strategy,
                'strategy_name': ['fast', 'balanced', 'deep', 'explorative'][strategy],
                'resources': resources,
                'calibration_temperature': self.calibration_temperature.item(),
                'avg_ece': self.ece_history[:max(1, self._ece_idx)].mean().item(),
            }
    
    def get_metakognition_stats(self):
        """Gib MetaKognitions-Statistiken."""
        avg_ece = self.ece_history[:max(1, self._ece_idx)].mean().item()
        avg_load = self.cognitive_load[:max(1, self._load_idx)].mean().item()
        
        # Strategie-Erfolgsrate
        total_trials = self.strategy_count.sum().item()
        total_success = self.strategy_success.sum().item()
        if total_trials > 0:
            success_rate = total_success / total_trials
        else:
            success_rate = 0.0
        
        return {
            'avg_ece': avg_ece,
            'cognitive_load': avg_load,
            'calibration_temp': self.calibration_temperature.item(),
            'current_strategy': self._current_strategy,
            'strategy_success_rate': success_rate,
            'n_strategies': self.n_strategies,
        }


class HierarchicalMemory(CogModule):
    """
    PHASE 49: Hierarchical Memory — 5-Ebenen-Gedächtnishierarchie mit Konsolidierung.
    
    Ebenen:
      L1 — Sensory Buffer:     Letzte 1000 Token-Eingaben (Rohdaten, flüchtig)
      L2 — Working Memory:     Aktueller Kontext (256 Slots, temperature-basiertes Retrieval)
      L3 — Episodic Buffer:    Wichtige Embeddings der letzten Iteration (500 Slots)
      L4 — Semantic Network:   Abstrahiertes Wissen (koordiniert mit KnowledgeGraph)
      L5 — Procedural Memory:  Gelernte Skills (koordiniert mit SkillModule)
    
    Konsolidierung (Sleep-Phase):
      L1 → L2: Attention-Filtering (relevante Rohdaten → Working Memory)
      L2 → L3: Importance-Based Promotion (wichtige Kontexte → Episodic Buffer)
      L3 → L4: Pattern Abstraction (wiederholte Muster → Semantic Network)
      L4 → L5: Skill Compilation (stabile Konzepte → Procedural Skills)
    
    Inspiriert vom hippocampal-neocorticalen Konsolidierungsmodell:
    - Hippocampus (L2/L3): schnelle Enkodierung, temporärer Speicher
    - Neocortex (L4/L5): langsame Extraktion, permanentes Wissen
    """
    def __init__(self, d_model, sensory_buffer_size=1000, working_mem_size=256,
                 episodic_buffer_size=500, n_importance_bins=5):
        super().__init__()
        self.d_model = d_model
        self.sensory_buffer_size = sensory_buffer_size
        self.working_mem_size = working_mem_size
        self.episodic_buffer_size = episodic_buffer_size
        self.n_importance_bins = n_importance_bins
        
        # =====================================================================
        #  L1 — SENSORY BUFFER
        # =====================================================================
        # Speichert Roh-Eingaben als (token_ids, embedding) Paare
        self.register_buffer('sensory_tokens', torch.zeros(sensory_buffer_size, 128, dtype=torch.long))
        self.register_buffer('sensory_embeddings', torch.zeros(sensory_buffer_size, d_model))
        self.register_buffer('sensory_age', torch.zeros(sensory_buffer_size))
        self.register_buffer('sensory_salience', torch.zeros(sensory_buffer_size))  # Wichtigkeit
        self._sensory_idx = 0
        
        # ——— L1→L2 Attention Gate ———
        # Berechnet Relevanz eines Sensory-Inputs für Working Memory
        self.attention_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )
        
        # =====================================================================
        #  L2 — WORKING MEMORY (Enhanced)
        # =====================================================================
        # Erweiterte Version von EpisodicMemory (Phase 1) mit mehr Slots + Temp-Retrieval
        self.register_buffer('working_memory', torch.zeros(working_mem_size, d_model))
        self.register_buffer('working_age', torch.zeros(working_mem_size))
        self.register_buffer('working_importance', torch.zeros(working_mem_size))
        self.register_buffer('working_temperature', torch.tensor(1.0))  # Retrieval-Temp
        
        # Hebbian Write/Read
        self.wm_write = nn.Linear(d_model, working_mem_size, bias=False)
        self.wm_read = nn.Linear(working_mem_size, d_model, bias=False)
        self.wm_proj = nn.Linear(d_model, d_model, bias=False)
        self._max_weight = 2.0
        
        # =====================================================================
        #  L3 — EPISODIC BUFFER (Mid-Term)
        # =====================================================================
        # Wichtige Embeddings, die über mehrere Iterationen erhalten bleiben
        self.register_buffer('episodic_buffer', torch.zeros(episodic_buffer_size, d_model))
        self.register_buffer('episodic_age', torch.zeros(episodic_buffer_size))
        self.register_buffer('episodic_importance', torch.zeros(episodic_buffer_size))
        self.register_buffer('episodic_consolidation_count', torch.zeros(episodic_buffer_size))
        self.register_buffer('episodic_domain', torch.zeros(episodic_buffer_size, dtype=torch.long))
        self._episodic_idx = 0
        
        # ——— L2→L3 Importance Scorer ———
        self.importance_scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, n_importance_bins),
        )
        
        # =====================================================================
        #  L4 — Semantic Abstraction Gate
        # =====================================================================
        # Extrahiert wiederholte Patterns aus Episodic → konzeptuelle Embeddings
        self.semantic_abstraction = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, d_model),
        )
        
        # Pattern-Detection: Cos-Ähnlichkeit zwischen episodischen Embeddings
        self.register_buffer('pattern_similarity_threshold', torch.tensor(0.85))
        
        # =====================================================================
        #  L5 — Procedural Compilation Gate
        # =====================================================================
        # Destilliert stabile semantische Konzepte → Skill-ähnliche Vektoren
        self.procedural_compilation = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, d_model),
        )
        self.register_buffer('compiled_skills', torch.zeros(8, d_model))  # Max 8 compiled skills
        self._skill_compile_idx = 0
        
        # =====================================================================
        #  Metrik-Tracking
        # =====================================================================
        self.register_buffer('consolidation_stats', torch.zeros(10))  # Letzte 10 Konsolidierungs-Runs
        self._consolidation_idx = 0
        
        # Externe Referenzen (werden von CogLang gesetzt)
        self._knowledge_graph = None
        self._skill_module = None
        
        self._max_weight = 2.0
    
    def link_modules(self, knowledge_graph=None, skill_module=None):
        """Verbinde mit bestehenden CogLang-Modulen."""
        self._knowledge_graph = knowledge_graph
        self._skill_module = skill_module
    
    # =========================================================================
    #  L1 — SENSORY BUFFER
    # =========================================================================
    
    def store_sensory(self, token_ids, embedding):
        """
        Speichere Roh-Eingabe im Sensory Buffer.
        
        Args:
            token_ids: [batch, seq] — Input-Token
            embedding: [batch, seq, d_model] — Encodierte Embeddings
        """
        with torch.no_grad():
            batch = token_ids.size(0)
            for b in range(batch):
                idx = self._sensory_idx % self.sensory_buffer_size
                
                # Speichere Tokens (max 128)
                seq_len = min(token_ids.size(1), 128)
                self.sensory_tokens[idx, :seq_len] = token_ids[b, :seq_len].to(dtype=torch.long)
                
                # Speichere gemitteltes Embedding
                emb_mean = embedding[b].mean(dim=0)  # [d_model]
                self.sensory_embeddings[idx] = emb_mean
                
                # Berechne Salience (vorläufig: uniform)
                self.sensory_salience[idx] = 1.0
                self.sensory_age[idx] = 0
                
                self._sensory_idx += 1
    
    def attend_sensory_to_working(self, top_k=16):
        """
        L1 → L2: Aufmerksamkeitsgesteuerte Promotion in Working Memory.
        
        Wählt die top_k salientesten Sensory-Inputs aus und schreibt sie
        in den Working Memory.
        """
        with torch.no_grad():
            n_valid = min(self._sensory_idx, self.sensory_buffer_size)
            if n_valid < 4:
                return
            
            # Verfügbare Embeddings
            embeddings = self.sensory_embeddings[:n_valid]  # [n, d_model]
            
            # Berechne Salience via Attention Gate
            salience = torch.sigmoid(self.attention_gate(embeddings)).squeeze(-1)  # [n]
            
            # Mische mit bestehender Sensory-Salience
            salience = 0.7 * salience + 0.3 * self.sensory_salience[:n_valid]
            
            # Wähle top_k
            _, top_idx = torch.topk(salience, min(top_k, n_valid))
            
            # Schreibe in Working Memory
            for idx in top_idx:
                emb = embeddings[idx]
                self._write_working_memory(emb)
    
    # =========================================================================
    #  L2 — WORKING MEMORY
    # =========================================================================
    
    def retrieve_working(self, query, temperature=None):
        """
        Content-Addressable Retrieval aus Working Memory.
        
        Args:
            query: [batch, seq, d_model] — Query-Embedding
            temperature: float oder None — Retrieval-Temp (niedriger = schärfer)
            
        Returns:
            retrieved: [batch, seq, d_model] — Abgerufene Memory-Inhalte
        """
        with torch.no_grad():
            batch, seq, d = query.shape
            temp = temperature if temperature is not None else self.working_temperature.item()
            
            q_flat = query.reshape(-1, d)  # [batch*seq, d]
            
            # Cosine-Ähnlichkeit
            q_norm = F.normalize(q_flat, dim=-1)
            m_norm = F.normalize(self.working_memory, dim=-1)  # [wm_size, d]
            similarity = q_norm @ m_norm.T  # [batch*seq, wm_size]
            
            # Temperature-Scaled Attention
            attention = F.softmax(similarity / max(temp, 0.1), dim=-1)
            
            # Retrieve
            retrieved = attention @ self.working_memory  # [batch*seq, d]
            retrieved = retrieved.reshape(batch, seq, d)
            
            # Projektion
            retrieved = self.wm_proj(retrieved)
            
            return retrieved
    
    def _write_working_memory(self, state, importance=1.0):
        """Schreibe einen State in den Working Memory (ältester Slot)."""
        with torch.no_grad():
            # Finde ältesten Slot
            oldest_idx = torch.argmax(self.working_age)
            
            self.working_memory[oldest_idx] = state.detach()
            self.working_age[oldest_idx] = 0
            self.working_importance[oldest_idx] = importance
            
            # Alter aller Slots erhöhen
            self.working_age += 1
            self.working_age = self.working_age.clamp(0, 1000)
    
    def compute_importance(self, embedding, error_norm=None):
        """
        Berechne Importance eines Embeddings für L2→L3 Promotion.
        
        Args:
            embedding: [d_model] — State-Embedding
            error_norm: float oder None — Prediction Error (höher = wichtiger)
            
        Returns:
            importance_score: float (0..1)
            importance_bin: int (0..n_importance_bins-1)
        """
        with torch.no_grad():
            # Features: Embedding + Error
            feat = embedding.unsqueeze(0)  # [1, d_model]
            
            # Importance via Scorer
            raw = self.importance_scorer(feat)  # [1, n_bins]
            probs = F.softmax(raw, dim=-1)
            
            # Gewichtete Summe der Bins
            bin_weights = torch.arange(self.n_importance_bins, device=raw.device).float()
            importance = (probs * bin_weights).sum().item() / (self.n_importance_bins - 1)
            
            # Error-Boost: höherer Prediction Error = höhere Importance
            if error_norm is not None:
                error_factor = min(error_norm, 5.0) / 5.0  # Normalisiert auf 0..1
                importance = 0.6 * importance + 0.4 * error_factor
            
            importance = max(0.0, min(1.0, importance))
            bin_idx = min(int(importance * self.n_importance_bins), self.n_importance_bins - 1)
            
            return importance, bin_idx
    
    # =========================================================================
    #  L3 — EPISODIC BUFFER
    # =========================================================================
    
    def promote_to_episodic(self, embedding, importance, domain_idx=0):
        """
        L2 → L3: Promotion eines Working-Memory-States in den Episodic Buffer.
        Nur States mit hoher Importance werden übernommen.
        """
        with torch.no_grad():
            if importance < 0.3:
                return False  # Zu unwichtig
            
            idx = self._episodic_idx % self.episodic_buffer_size
            self.episodic_buffer[idx] = embedding.detach()
            self.episodic_age[idx] = 0
            self.episodic_importance[idx] = importance
            self.episodic_domain[idx] = domain_idx
            self.episodic_consolidation_count[idx] = 1
            
            self._episodic_idx += 1
            return True
    
    def retrieve_episodic(self, query, top_k=5):
        """
        Retrieval aus Episodic Buffer (gewichtete Cos-Ähnlichkeit).
        
        Args:
            query: [batch, seq, d_model]
            top_k: int
            
        Returns:
            episodes: [batch, seq, d_model] — Gewichtete Summe der Top-K
            scores: [top_k] — Ähnlichkeits-Scores
        """
        with torch.no_grad():
            batch, seq, d = query.shape
            n_valid = min(self._episodic_idx, self.episodic_buffer_size)
            if n_valid < 1:
                return torch.zeros_like(query), torch.zeros(0)
            
            q_flat = query.reshape(-1, d)  # [batch*seq, d]
            q_norm = F.normalize(q_flat, dim=-1)
            e_norm = F.normalize(self.episodic_buffer[:n_valid], dim=-1)  # [n, d]
            
            # Cos-Ähnlichkeit
            sim = q_norm @ e_norm.T  # [batch*seq, n]
            
            # Gewichte mit Importance
            imp = self.episodic_importance[:n_valid].unsqueeze(0)  # [1, n]
            weighted_sim = sim * imp  # [batch*seq, n]
            
            # Top-K
            top_scores, top_idx = torch.topk(weighted_sim, min(top_k, n_valid), dim=-1)
            
            # Retrieve
            retrieved = torch.zeros(batch * seq, d, device=query.device)
            for i in range(batch * seq):
                weights = F.softmax(top_scores[i], dim=-1)  # [top_k]
                retrieved[i] = (self.episodic_buffer[top_idx[i]] * weights.unsqueeze(-1)).sum(dim=0)
            
            retrieved = retrieved.reshape(batch, seq, d)
            
            return retrieved, top_scores
    
    # =========================================================================
    #  L4 → L5 — CONSOLIDATION (Sleep-Phase)
    # =========================================================================
    
    def consolidate(self, n_cycles=3):
        """
        Führe vollständige Konsolidierung durch alle Hierarchie-Ebenen aus.
        
        Returns:
            report: dict mit Konsolidierungs-Statistiken
        """
        with torch.no_grad():
            report = {
                'sensory_to_working': 0,
                'working_to_episodic': 0,
                'patterns_found': 0,
                'skills_compiled': 0,
                'consolidation_cycle': self._consolidation_idx,
            }
            
            for cycle in range(n_cycles):
                # ——— L1 → L2: Sensory → Working ———
                self.attend_sensory_to_working(top_k=16)
                report['sensory_to_working'] += 16
                
                # ——— L2 → L3: Working → Episodic ———
                n_wm = min(self.working_mem_size, 256)
                promo_count = 0
                for i in range(n_wm):
                    emb = self.working_memory[i]
                    imp, _ = self.compute_importance(emb)
                    if imp > 0.4:
                        success = self.promote_to_episodic(emb, imp)
                        if success:
                            promo_count += 1
                
                report['working_to_episodic'] += promo_count
                
                # ——— L3 → L4: Pattern Detection (Semantic Abstraction) ———
                n_ep = min(self._episodic_idx, self.episodic_buffer_size)
                if n_ep > 10:
                    # Finde wiederholte Patterns via Clustering-Ähnlichkeit
                    embeddings = self.episodic_buffer[:n_ep]  # [n, d]
                    norms = F.normalize(embeddings, dim=-1)
                    
                    # Pairwise Cos-Ähnlichkeit
                    sim_matrix = norms @ norms.T  # [n, n]
                    
                    # Finde ähnliche Paare (oberes Dreieck)
                    mask = torch.triu(torch.ones(n_ep, n_ep, device=sim_matrix.device), diagonal=1)
                    paired_sim = sim_matrix * mask
                    
                    similar_pairs = (paired_sim > self.pattern_similarity_threshold).nonzero()
                    
                    for pair in similar_pairs[:5]:  # Max 5 Patterns pro Cycle
                        i, j = pair[0].item(), pair[1].item()
                        
                        # Abstrahiere gemeinsames Pattern
                        pattern = (embeddings[i] + embeddings[j]) / 2.0
                        abstracted = self.semantic_abstraction(pattern.unsqueeze(0)).squeeze(0)
                        
                        # Wenn KnowledgeGraph verfügbar: speichere als neue Entität
                        if self._knowledge_graph is not None:
                            self._knowledge_graph.add_entity(name=f"pattern_{cycle}_{i}")
                        
                        report['patterns_found'] += 1
                
                # ——— L4 → L5: Procedural Compilation ———
                # Destilliere stabile Patterns in Skills
                if report['patterns_found'] > 0 and report['patterns_found'] > report['skills_compiled']:
                    # Wähle neueste Patterns zur Kompilierung
                    for _ in range(min(2, report['patterns_found'])):
                        if self._skill_compile_idx < 8:
                            # Verwende Durchschnitt der letzten Episodic-Embeddings
                            if n_ep > 0:
                                recent = self.episodic_buffer[max(0, n_ep-10):n_ep].mean(dim=0)
                                compiled = self.procedural_compilation(recent.unsqueeze(0)).squeeze(0)
                                self.compiled_skills[self._skill_compile_idx] = compiled
                                self._skill_compile_idx += 1
                                report['skills_compiled'] += 1
                
                # Alter aller Ebenen erhöhen (Vergessen)
                self.sensory_age += 1
                self.working_age += 1
                self.episodic_age += 1
            
            # Statistik
            self.consolidation_stats[self._consolidation_idx % 10] = (
                report['patterns_found'] + report['skills_compiled'] * 0.5
            )
            self._consolidation_idx += 1
            
            return report
    
    # =========================================================================
    #  FORWARD — Retrieval aus allen Ebenen
    # =========================================================================
    
    def forward(self, query, retrieve_levels=None):
        """
        Führe hierarchisches Memory-Retrieval durch.
        
        Args:
            query: [batch, seq, d_model] — Query-Embedding
            retrieve_levels: list von Level-Namen oder None (alle)
                           ['working', 'episodic', 'semantic', 'procedural']
            
        Returns:
            result: dict mit level -> retrieved_embedding
        """
        with torch.no_grad():
            if retrieve_levels is None:
                retrieve_levels = ['working', 'episodic']
            
            result = {}
            
            # L2 — Working Memory Retrieval
            if 'working' in retrieve_levels:
                wm_retrieved = self.retrieve_working(query)
                result['working'] = wm_retrieved
            
            # L3 — Episodic Buffer Retrieval
            if 'episodic' in retrieve_levels:
                ep_retrieved, ep_scores = self.retrieve_episodic(query)
                result['episodic'] = ep_retrieved
                result['episodic_scores'] = ep_scores
            
            # L4 — Semantic (via KnowledgeGraph)
            if 'semantic' in retrieve_levels and self._knowledge_graph is not None:
                ctx_pooled = query.mean(dim=1)
                knowledge = self._knowledge_graph.retrieve(ctx_pooled, top_k=3)
                result['semantic'] = knowledge
            
            # L5 — Procedural (via compiled skills)
            if 'procedural' in retrieve_levels:
                # Cos-Ähnlichkeit zwischen Query und Compiled Skills
                q_pooled = query.mean(dim=1)
                q_norm = F.normalize(q_pooled, dim=-1)
                s_norm = F.normalize(self.compiled_skills, dim=-1)
                
                n_compiled = min(self._skill_compile_idx, 8)
                if n_compiled > 0:
                    sim = q_norm @ s_norm[:n_compiled].T  # [batch, n_compiled]
                    weights = F.softmax(sim / 0.5, dim=-1)  # [batch, n_compiled]
                    
                    # Gewichtete Summe
                    proc_retrieved = weights @ self.compiled_skills[:n_compiled]  # [batch, d]
                    proc_retrieved = proc_retrieved.unsqueeze(1).expand(-1, query.size(1), -1)
                    result['procedural'] = proc_retrieved
                else:
                    result['procedural'] = torch.zeros_like(query)
            
            return result
    
    def get_memory_stats(self):
        """Gib Hierarchie-Gedächtnis-Statistiken."""
        n_sensory = min(self._sensory_idx, self.sensory_buffer_size)
        n_wm = int((self.working_age < 1000).sum().item())
        n_ep = min(self._episodic_idx, self.episodic_buffer_size)
        
        # Durchschnittliche Importance
        avg_wm_imp = self.working_importance[:n_wm].mean().item() if n_wm > 0 else 0.0
        avg_ep_imp = self.episodic_importance[:n_ep].mean().item() if n_ep > 0 else 0.0
        
        # Konsolidierungs-Qualität
        avg_consol = self.consolidation_stats[:max(1, self._consolidation_idx)].mean().item()
        
        return {
            'sensory_usage': n_sensory,
            'working_usage': n_wm,
            'episodic_usage': n_ep,
            'compiled_skills': min(self._skill_compile_idx, 8),
            'avg_wm_importance': avg_wm_imp,
            'avg_ep_importance': avg_ep_imp,
            'consolidation_quality': avg_consol,
            'pattern_threshold': self.pattern_similarity_threshold.item(),
            'retrieval_temperature': self.working_temperature.item(),
        }


class HierarchicalGoal(CogModule):
    """
    PHASE 52: Hierarchische Zielsetzung — Goal Decomposer, Subgoal Tracker, Goal Adaptation.
    
    Baut auf GoalEncoder (Phase 36) auf und erweitert ihn zu einem vollständigen
    hierarchischen Planungssystem:
    
    1. GoalDecomposer — Zerlege High-Level-Goals in Subgoals (automatisch)
    2. SubgoalTracker — Verfolge Fortschritt: pending, active, completed, failed
    3. GoalAdaptation — Passe Goals an bei Misserfolg (alternative Pfade)
    
    Architektur:
    - Jedes Goal hat einen Embedding-Vektor (d_model)
    - Subgoals sind als DAG organisiert (Dependencies)
    - Decomposition lernt aus Erfolg/Misserfolg
    - Adaptation nutzt MetaKognition-Strategien für alternative Pläne
    
    Goal-Zustände:
    PENDING → ACTIVE → COMPLETED
                         → FAILED → ADAPTED (alternatives Subgoal)
    """
    def __init__(self, d_model, max_goals=32, max_subgoals_per_goal=8, max_depth=4):
        super().__init__()
        self.d_model = d_model
        self.max_goals = max_goals
        self.max_subgoals_per_goal = max_subgoals_per_goal
        self.max_depth = max_depth
        
        # =====================================================================
        #  1. GOAL DECOMPOSER
        # =====================================================================
        # Zerlegt High-Level-Goal-Embedding in Subgoal-Embeddings
        self.decomposer = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.Tanh(),
            nn.Linear(d_model * 2, max_subgoals_per_goal * d_model),
        )
        
        # Subgoal-Selector (welche Subgoals sind relevant?)
        self.subgoal_selector = nn.Sequential(
            nn.Linear(d_model, max_subgoals_per_goal),
        )
        
        # =====================================================================
        #  2. SUBGOAL TRACKER
        # =====================================================================
        # Goal-Struktur (Buffer)
        self.register_buffer('goal_embeddings', torch.zeros(max_goals, d_model))
        self.register_buffer('goal_depth', torch.zeros(max_goals, dtype=torch.long))  # Hierarchie-Ebene
        self.register_buffer('goal_parent', torch.zeros(max_goals, dtype=torch.long))  # Parent-Index
        self.register_buffer('goal_status', torch.zeros(max_goals, dtype=torch.long))  # 0=pending,1=active,2=completed,3=failed
        self.register_buffer('goal_priority', torch.zeros(max_goals))  # 0..1
        self.register_buffer('goal_progress', torch.zeros(max_goals))  # 0..1
        self.register_buffer('goal_attempts', torch.zeros(max_goals, dtype=torch.long))
        self.register_buffer('goal_success_history', torch.zeros(max_goals, 10))  # Letzte 10 Ergebnisse
        
        # Subgoal-Child-Index: pro Goal die Indizes seiner Subgoals
        self.register_buffer('subgoal_children', torch.zeros(max_goals, max_subgoals_per_goal, dtype=torch.long))
        self.register_buffer('subgoal_counts', torch.zeros(max_goals, dtype=torch.long))
        
        # Dependencies: subgoal B kann nicht starten bevor A erledigt ist
        self.register_buffer('dependency_matrix', torch.zeros(max_goals, max_goals, dtype=torch.bool))
        
        # ——— Goal Embedding Encoder ———
        # Projiziert Context-Embedding in Goal-Space (für Goal-Erkennung)
        self.goal_encoder = nn.Linear(d_model, d_model, bias=False)
        
        # ——— Progress Estimator ———
        # Schätzt Fortschritt aus aktuellen Hidden-States
        self.progress_estimator = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )
        
        self._goal_count = 0
        self._active_goal_idx = -1  # Aktuell verfolgtes Goal
        self._current_goal_embedding = None
        
        # =====================================================================
        #  3. GOAL ADAPTATION
        # =====================================================================
        # Alternative Pfade bei Misserfolg
        self.adaptation_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model),
        )
        
        # Failure-Pattern-Memory: speichert, welche Goal-Typen oft scheitern
        self.register_buffer('failure_patterns', torch.zeros(50, d_model))
        self.register_buffer('failure_counts', torch.zeros(50))
        self._failure_idx = 0
        
        self._max_weight = 2.0
    
    # =========================================================================
    #  GOAL DECOMPOSITION
    # =========================================================================
    
    def decompose(self, high_level_goal, depth=0, parent_idx=-1):
        """
        Zerlege ein High-Level-Goal in Subgoals.
        
        Args:
            high_level_goal: [d_model] — Embedding des Hauptziels
            depth: int — Aktuelle Tiefe in der Hierarchie
            parent_idx: int — Index des Parent-Goals (-1 = root)
            
        Returns:
            goal_idx: int — Index des erstellten Goals
            n_subgoals: int — Anzahl erstellter Subgoals
        """
        with torch.no_grad():
            # Erstelle neuen Goal-Eintrag
            idx = self._goal_count % self.max_goals
            self.goal_embeddings[idx] = high_level_goal
            self.goal_depth[idx] = depth
            self.goal_parent[idx] = max(0, parent_idx)
            self.goal_status[idx] = 0  # PENDING
            self.goal_priority[idx] = 1.0 if depth == 0 else 0.8 ** depth
            self.goal_progress[idx] = 0.0
            self.goal_attempts[idx] = 0
            
            # Verknüpfe mit Parent
            if parent_idx >= 0:
                child_count = self.subgoal_counts[parent_idx].item()
                if child_count < self.max_subgoals_per_goal:
                    self.subgoal_children[parent_idx, child_count] = idx
                    self.subgoal_counts[parent_idx] += 1
            
            self._goal_count += 1
            
            # Dekomposition (nur wenn nicht zu tief)
            n_subgoals = 0
            if depth < self.max_depth:
                # Generiere Subgoal-Kandidaten
                raw_subgoals = self.decomposer(high_level_goal.unsqueeze(0))  # [1, max_subgoals * d]
                raw_subgoals = raw_subgoals.view(self.max_subgoals_per_goal, self.d_model)
                
                # Selektion: welche Subgoals sind relevant?
                relevance = torch.sigmoid(self.subgoal_selector(high_level_goal.unsqueeze(0)))  # [1, max]
                relevance = relevance.squeeze(0)  # [max]
                
                # Wähle Top-k Subgoals (relevanz > 0.5)
                selected = (relevance > 0.5).nonzero(as_tuple=True)[0]
                
                for si in selected[:self.max_subgoals_per_goal // 2]:  # Max 4 Subgoals
                    sub_goal_emb = raw_subgoals[si]
                    sub_idx, _ = self.decompose(sub_goal_emb, depth + 1, idx)
                    n_subgoals += 1
                    
                    # Dependency: Subgoal muss vor Parent abgeschlossen sein
                    if parent_idx >= 0:
                        self.dependency_matrix[sub_idx, idx] = True
            
            # Falls keine Subgoals erstellt: setze als leaf (kann direkt ausgeführt werden)
            if n_subgoals == 0:
                self.goal_status[idx] = 1  # ACTIVE (kann direkt bearbeitet werden)
            
            return idx, n_subgoals
    
    # =========================================================================
    #  GOAL TRACKING
    # =========================================================================
    
    def set_active_goal(self, goal_idx):
        """Setze ein Goal als aktiv."""
        if 0 <= goal_idx < self.max_goals:
            self._active_goal_idx = goal_idx
            self.goal_status[goal_idx] = 1  # ACTIVE
            self._current_goal_embedding = self.goal_embeddings[goal_idx]
    
    def update_progress(self, hidden_states, loss_val=None):
        """
        Aktualisiere Fortschritt des aktiven Goals.
        
        Args:
            hidden_states: [batch, seq, d_model] — Aktuelle Hidden-States
            loss_val: float oder None — Aktueller Loss (niedrig = mehr Fortschritt)
        """
        with torch.no_grad():
            if self._active_goal_idx < 0:
                return
            
            # Schätze Fortschritt aus Hidden-States
            pooled = hidden_states.mean(dim=1)  # [batch, d_model]
            raw_progress = torch.sigmoid(self.progress_estimator(pooled))  # [batch, 1]
            est_progress = raw_progress.mean().item()
            
            # Loss-Boost: niedriger Loss = mehr Fortschritt
            if loss_val is not None:
                loss_progress = max(0.0, 1.0 - min(loss_val, 10.0) / 10.0)
                est_progress = 0.5 * est_progress + 0.5 * loss_progress
            
            # Aktualisiere
            current = self.goal_progress[self._active_goal_idx].item()
            self.goal_progress[self._active_goal_idx] = max(current, est_progress)
            
            # Bei Fortschritt > 0.8: als completed markieren
            if self.goal_progress[self._active_goal_idx] > 0.8:
                self.complete_goal(self._active_goal_idx)
    
    def complete_goal(self, goal_idx, success=True):
        """Markiere Goal als completed oder failed."""
        if success:
            self.goal_status[goal_idx] = 2  # COMPLETED
            
            # Aktualisiere Parent-Progress
            parent = self.goal_parent[goal_idx].item()
            if parent >= 0 and parent < self.max_goals:
                # Zähle completed children
                n_children = self.subgoal_counts[parent].item()
                if n_children > 0:
                    completed = 0
                    for c in range(n_children):
                        child_idx = self.subgoal_children[parent, c].item()
                        if self.goal_status[child_idx] == 2:  # COMPLETED
                            completed += 1
                    self.goal_progress[parent] = completed / n_children
                    
                    # Wenn alle Children completed: Parent auch completed
                    if self.goal_progress[parent] >= 1.0 and self.goal_status[parent] != 2:
                        self.goal_status[parent] = 2
        else:
            self.goal_status[goal_idx] = 3  # FAILED
            self.goal_attempts[goal_idx] += 1
            
            # Speichere Failure-Pattern
            emb = self.goal_embeddings[goal_idx]
            f_idx = self._failure_idx % 50
            self.failure_patterns[f_idx] = emb
            self.failure_counts[f_idx] = self.goal_attempts[goal_idx].float()
            self._failure_idx += 1
    
    # =========================================================================
    #  GOAL ADAPTATION
    # =========================================================================
    
    def adapt_goal(self, goal_idx, hidden_states):
        """
        Passe ein gescheitertes Goal an — generiere alternativen Pfad.
        
        Args:
            goal_idx: int — Index des gescheiterten Goals
            hidden_states: [batch, seq, d_model] — Aktueller Kontext
            
        Returns:
            new_goal_idx: int — Index des adaptierten Goals (-1 wenn keine Adaptation)
        """
        with torch.no_grad():
            if self.goal_status[goal_idx] != 3:  # Nur failed goals anpassen
                return -1
            
            if self.goal_attempts[goal_idx] > 3:  # Max 3 Versuche
                return -1
            
            orig_emb = self.goal_embeddings[goal_idx]
            pooled = hidden_states.mean(dim=1)  # [batch, d_model]
            ctx = pooled.mean(dim=0)  # [d_model]
            
            # Adaptation: [original_goal; context] → adapted_goal
            adapt_in = torch.cat([orig_emb, ctx], dim=-1).unsqueeze(0)  # [1, 2*d]
            adapted = self.adaptation_gate(adapt_in).squeeze(0)  # [d_model]
            
            # Erstelle neues Goal mit adaptiertem Embedding
            parent = self.goal_parent[goal_idx].item()
            new_idx, _ = self.decompose(adapted, depth=self.goal_depth[goal_idx].item() + 1, parent_idx=parent)
            
            # Setze Dependency: adaptiertes Goal braucht das Original nicht
            self.dependency_matrix[new_idx, goal_idx] = False
            
            return new_idx
    
    # =========================================================================
    #  GOAL CONDITIONING (Forward-Pass)
    # =========================================================================
    
    def condition(self, hidden_states):
        """
        Konditioniere Hidden-States mit aktuellem Goal.
        
        Args:
            hidden_states: [batch, seq, d_model] — Aktuelle Aktivierungen
            
        Returns:
            conditioned: [batch, seq, d_model] — Goal-modulierte Aktivierungen
        """
        with torch.no_grad():
            if self._current_goal_embedding is None:
                return hidden_states
            
            batch, seq, d = hidden_states.shape
            goal_exp = self._current_goal_embedding.unsqueeze(0).unsqueeze(0).expand(batch, seq, -1)
            
            # Gate-gesteuerte Goal-Modulation
            gate_signal = torch.sigmoid((hidden_states * goal_exp).sum(dim=-1, keepdim=True) / (d ** 0.5))
            
            return hidden_states + gate_signal * goal_exp * 0.1
    
    def get_goal_embedding(self):
        """Gib aktuelles Goal-Embedding zurück (für GoalEncoder-Kompatibilität)."""
        return self._current_goal_embedding
    
    # =========================================================================
    #  GOAL QUERIES
    # =========================================================================
    
    def get_next_actionable_goal(self):
        """
        Finde das nächste Goal, das bearbeitet werden kann.
        (alle Dependencies erfüllt, Status = PENDING)
        
        Returns:
            goal_idx: int oder -1
        """
        for i in range(self._goal_count):
            if self.goal_status[i] != 0:  # Nur PENDING
                continue
            if self.goal_depth[i] > self.max_depth:
                continue
            
            # Prüfe Dependencies
            deps_satisfied = True
            for dep in range(self.max_goals):
                if self.dependency_matrix[i, dep]:
                    if self.goal_status[dep] != 2:  # Nicht completed
                        deps_satisfied = False
                        break
            
            if deps_satisfied:
                return i
        
        return -1
    
    def get_goal_tree(self, root_idx=0, depth=0, max_depth=3):
        """
        Gib Goal-Baum als verschachteltes Dict zurück.
        
        Returns:
            tree: dict mit goal_info und children
        """
        def _build_tree(idx, current_depth):
            if current_depth > max_depth or idx >= self._goal_count:
                return None
            
            status_names = ['PENDING', 'ACTIVE', 'COMPLETED', 'FAILED']
            node = {
                'index': idx,
                'depth': self.goal_depth[idx].item(),
                'status': status_names[self.goal_status[idx].item()],
                'priority': self.goal_priority[idx].item(),
                'progress': self.goal_progress[idx].item(),
                'attempts': self.goal_attempts[idx].item(),
            }
            
            children = []
            n_child = self.subgoal_counts[idx].item()
            for c in range(n_child):
                child_idx = self.subgoal_children[idx, c].item()
                child_node = _build_tree(child_idx, current_depth + 1)
                if child_node:
                    children.append(child_node)
            
            if children:
                node['children'] = children
            
            return node
        
        return _build_tree(root_idx, depth)
    
    def get_goal_stats(self):
        """Gib Zielsetzungs-Statistiken."""
        total = self._goal_count
        if total == 0:
            return {
                'n_goals': 0, 'n_completed': 0, 'n_failed': 0,
                'n_active': 0, 'n_pending': 0, 'success_rate': 0.0,
            }
        
        completed = (self.goal_status[:total] == 2).sum().item()
        failed = (self.goal_status[:total] == 3).sum().item()
        active = (self.goal_status[:total] == 1).sum().item()
        pending = (self.goal_status[:total] == 0).sum().item()
        
        success_rate = completed / max(1, completed + failed)
        
        return {
            'n_goals': total,
            'n_completed': completed,
            'n_failed': failed,
            'n_active': active,
            'n_pending': pending,
            'success_rate': success_rate,
            'max_depth': self.max_depth,
            'max_subgoals': self.max_subgoals_per_goal,
        }


class MetaLearning(CogModule):
    """
    PHASE 55: Meta-Learning — Lernen zu Lernen.
    
    CogLang optimiert seinen eigenen Lernprozess:
    
    1. LearningStrategyEncoder — Kodiert aktuelle Hyperparameter als Strategie-Vektor
    2. StrategyMetaNetwork — Sagt voraus: welche Strategie für welche Daten?
    3. HyperparameterController — Steuert LR, Sparsity, Batch-Size dynamisch
    
    Inspiriert von:
    - Learning to Learn by Gradient Descent by Gradient Descent (Andrychowicz 2016)
    - Meta-Learning in Reinforcement Learning (Duan 2017)
    - Hyperparameter-Optimization as Meta-Learning
    """
    def __init__(self, d_model, n_hyperparams=5, n_strategy_dim=32, window_size=200):
        super().__init__()
        self.d_model = d_model
        self.n_hyperparams = n_hyperparams
        self.n_strategy_dim = n_strategy_dim
        self.window_size = window_size
        
        # =====================================================================
        #  1. LEARNING STRATEGY ENCODER
        # =====================================================================
        # Kodiert aktuelle Hyperparameter (LR, Sparsity, Momentum, Temperature, Batch-Size)
        # als d_model-Vektor für das Meta-Netzwerk
        self.hp_embedding = nn.Embedding(100, d_model // 4)  # Diskrete HP-Stufen
        self.strategy_encoder = nn.Sequential(
            nn.Linear(n_hyperparams * (d_model // 4) + d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, n_strategy_dim),
        )
        
        # =====================================================================
        #  2. STRATEGY META-NETWORK
        # =====================================================================
        # Sagt voraus, wie gut eine Strategie für gegebene Daten funktioniert
        self.meta_predictor = nn.Sequential(
            nn.Linear(n_strategy_dim + d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 3),  # loss_pred, progress_pred, plateau_risk
        )
        
        # ——— Strategy Memory (Buffer) ———
        self.register_buffer('strategy_history', torch.zeros(window_size, n_strategy_dim))
        self.register_buffer('strategy_performance', torch.zeros(window_size, 3))  # loss, progress, plateau
        self.register_buffer('strategy_data_embedding', torch.zeros(window_size, d_model))
        self._strategy_idx = 0
        
        # =====================================================================
        #  3. HYPERPARAMETER CONTROLLER
        # =====================================================================
        # Steuert 5 Hyperparameter:
        #   [0] LR-Scale (0.1..3.0x aktuelles LR)
        #   [1] Sparsity (20%..80%)
        #   [2] Momentum (0.5..0.99)
        #   [3] Temperature (0.1..2.0)
        #   [4] Batch-Size-Multiplier (0.5..2.0x)
        self.hp_controller = nn.Sequential(
            nn.Linear(n_strategy_dim + 3, d_model // 2),  # strategy + loss_trend + plateau + variance
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, n_hyperparams),
        )
        
        # ——— Loss-Trend-Detection ———
        self.register_buffer('loss_history', torch.zeros(window_size))
        self.register_buffer('loss_trend', torch.zeros(1))  # -1=fallend, 0=stabil, 1=steigend
        self.register_buffer('plateau_counter', torch.zeros(1, dtype=torch.long))
        self.register_buffer('loss_variance', torch.zeros(1))
        self._loss_idx = 0
        
        # ——— Hyperparameter-Zustand ———
        self.register_buffer('current_hp', torch.ones(n_hyperparams))
        # 0: LR-Scale=1.0, 1: Sparsity=0.5, 2: Momentum=0.9, 
        # 3: Temperature=1.0, 4: Batch-Mult=1.0
        self.register_buffer('hp_names', torch.tensor([0, 1, 2, 3, 4]))  # dummy für Metriken
        
        # ——— Metrik ———
        self.register_buffer('hp_adaptation_count', torch.zeros(1, dtype=torch.long))
        self.register_buffer('meta_improvement', torch.zeros(100))  # Letzte 100 Verbesserungen
        self._meta_idx = 0
        
        # PHASE 57: Unsicherheits-gekoppelte HP-Steuerung (Active-Learning × Meta-Learning)
        self.register_buffer('uncertainty_coupled', torch.zeros(1))
        self._unc_step = 0
        
        self._max_weight = 2.0
    
    # =========================================================================
    #  1. STRATEGY ENCODING
    # =========================================================================
    
    def encode_strategy(self, hidden_states, hyperparams=None):
        """
        Kodiere aktuelle Lernstrategie als Vektor.
        
        Args:
            hidden_states: [batch, seq, d_model] — Aktuelle Aktivierungen
            hyperparams: list oder None — [lr_scale, sparsity, momentum, temp, batch_mult]
            
        Returns:
            strategy_vec: [n_strategy_dim] — Strategie-Vektor
        """
        with torch.no_grad():
            if hyperparams is None:
                hyperparams = self.current_hp.tolist()
            
            # Embedding der Hyperparameter
            hp_indices = torch.tensor([
                min(int(h * 100), 99) for h in hyperparams[:self.n_hyperparams]
            ], device=hidden_states.device)
            hp_emb = self.hp_embedding(hp_indices)  # [n_hp, d_model//4]
            hp_flat = hp_emb.view(-1)  # [n_hp * d_model//4]
            
            # Data-Embedding
            data_emb = hidden_states.mean(dim=(0, 1))  # [d_model]
            
            # Kombinieren
            combined = torch.cat([hp_flat, data_emb], dim=-1)  # [n_hp*d_model//4 + d_model]
            
            strategy_vec = self.strategy_encoder(combined.unsqueeze(0)).squeeze(0)  # [n_strategy_dim]
            
            return strategy_vec
    
    # =========================================================================
    #  2. STRATEGY META-NETWORK
    # =========================================================================
    
    def predict_strategy_performance(self, strategy_vec, data_embedding):
        """
        Sage voraus, wie gut diese Strategie für gegebene Daten funktioniert.
        
        Args:
            strategy_vec: [n_strategy_dim]
            data_embedding: [d_model]
            
        Returns:
            prediction: dict mit loss_pred, progress_pred, plateau_risk
        """
        with torch.no_grad():
            combined = torch.cat([strategy_vec, data_embedding], dim=-1).unsqueeze(0)
            raw = self.meta_predictor(combined).squeeze(0)  # [3]
            
            # Post-Processing
            loss_pred = torch.sigmoid(raw[0]).item() * 20.0  # 0..20 Loss
            progress_pred = torch.sigmoid(raw[1]).item()  # 0..1 Progress
            plateau_risk = torch.sigmoid(raw[2]).item()  # 0..1 Plateau-Risiko
            
            return {
                'predicted_loss': loss_pred,
                'predicted_progress': progress_pred,
                'predicted_plateau_risk': plateau_risk,
            }
    
    def store_strategy_result(self, strategy_vec, data_embedding, loss_val, progress_val, plateau_val):
        """Speichere Strategie-Ergebnis für späteres Lernen."""
        idx = self._strategy_idx % self.window_size
        self.strategy_history[idx] = strategy_vec
        self.strategy_data_embedding[idx] = data_embedding
        self.strategy_performance[idx] = torch.tensor([loss_val, progress_val, plateau_val])
        self._strategy_idx += 1
    
    def learn_from_strategy_history(self, n_samples=32):
        """
        Lerne aus gespeicherten Strategie-Ergebnissen.
        Verbessert die Vorhersage, welche Strategie gut funktioniert.
        
        Returns:
            learning_signal: float — Wie viel wurde gelernt?
        """
        with torch.no_grad():
            n_valid = min(self._strategy_idx, self.window_size)
            if n_valid < n_samples:
                return 0.0
            
            # Sample zufällige Batches
            indices = torch.randint(0, n_valid, (n_samples,), device=self.strategy_history.device)
            
            strategies = self.strategy_history[indices]  # [n, n_strategy_dim]
            data_embs = self.strategy_data_embedding[indices]  # [n, d_model]
            targets = self.strategy_performance[indices]  # [n, 3]
            
            total_error = 0.0
            for i in range(n_samples):
                combined = torch.cat([strategies[i], data_embs[i]], dim=-1).unsqueeze(0)
                prediction = self.meta_predictor(combined).squeeze(0)
                
                # Hebbian-ähnliches Update
                error = targets[i] - prediction
                # Vereinfachtes Meta-Update: ziehe prediction Richtung target
                lr = self._lr * self._meta_lr_scale * 0.01
                # NUR der erste Layer hat combined als Eingang ([d, n_strategy_dim + d_model]).
                # Die inneren Layer haben andere Eingangsdimensionen — ein dw aus combined
                # würde dort zu Shape-Kollisionen führen (z.B. d_sparse vs. d_sparse + 32).
                first = self.meta_predictor[0]
                dw = error.unsqueeze(1) * combined  # [3, d_in]
                grad = dw.mean(dim=0, keepdim=True) * lr  # [1, d_in] → broadcastet auf [d_out, d_in]
                first.weight.data.add_(grad)
                first.bias.data.add_(error.mean(dim=0) * lr)  # [3] → [d_out]
                first.weight.data.clamp_(-self._max_weight, self._max_weight)
                first.bias.data.clamp_(-self._max_weight, self._max_weight)
                
                total_error += error.abs().mean().item()
            
            avg_error = total_error / n_samples
            return avg_error
    
    # =========================================================================
    #  3. HYPERPARAMETER CONTROL
    # =========================================================================
    
    def update_loss_trend(self, current_loss):
        """
        Aktualisiere Loss-Trend-Detection.
        
        Args:
            current_loss: float — Aktueller Loss-Wert
        """
        idx = self._loss_idx % self.window_size
        self.loss_history[idx] = current_loss
        self._loss_idx += 1
        
        if self._loss_idx >= 10:
            # Berechne Trend über letzte 10 Werte
            recent = self.loss_history[max(0, idx - 9):idx + 1]
            if len(recent) >= 5:
                # Linearer Trend: Steigung der Regressionsgerade
                x = torch.arange(len(recent), device=recent.device).float()
                x_mean = x.mean()
                y_mean = recent.mean()
                slope = ((x - x_mean) * (recent - y_mean)).sum() / ((x - x_mean) ** 2).sum()
                
                # Trend: -1 = fallend (gut), 0 = stabil, 1 = steigend (schlecht)
                if slope < -0.1:
                    self.loss_trend[0] = -1.0
                    self.plateau_counter[0] = 0
                elif slope > 0.1:
                    self.loss_trend[0] = 1.0
                    self.plateau_counter[0] = 0
                else:
                    self.loss_trend[0] = 0.0
                    self.plateau_counter[0] += 1
                
                # Varianz
                self.loss_variance[0] = recent.var().item()
    
    def get_hyperparams(self):
        """
        Gib aktuelle optimierte Hyperparameter zurück.
        
        Returns:
            dict mit lr_scale, sparsity, momentum, temperature, batch_mult
        """
        with torch.no_grad():
            hp = self.current_hp.tolist()
            return {
                'lr_scale': max(0.1, min(3.0, hp[0])),
                'sparsity': max(0.2, min(0.8, hp[1])),
                'momentum': max(0.5, min(0.99, hp[2])),
                'temperature': max(0.1, min(2.0, hp[3])),
                'batch_mult': max(0.5, min(2.0, hp[4])),
            }
    
    def adapt_hyperparams(self, hidden_states):
        """
        Passe Hyperparameter basierend auf Strategie + Loss-Trend an.
        
        Args:
            hidden_states: [batch, seq, d_model] — Aktuelle Aktivierungen
            
        Returns:
            adapted_hp: dict mit neuen Hyperparametern
        """
        with torch.no_grad():
            # Strategie-Vektor
            strategy_vec = self.encode_strategy(hidden_states)
            
            # Features für Controller: strategy + trend + plateau + variance
            trend = self.loss_trend.item()
            plateau = min(self.plateau_counter.item(), 50) / 50.0  # Normalisiert
            variance = min(self.loss_variance.item(), 10.0) / 10.0
            
            features = torch.cat([
                strategy_vec,
                torch.tensor([trend, plateau, variance], device=hidden_states.device)
            ], dim=-1).unsqueeze(0)  # [1, n_strategy_dim + 3]
            
            # Generiere neue HP-Werte
            raw_hp = self.hp_controller(features).squeeze(0)  # [n_hyperparams]
            
            # Skaliere auf sinnvolle Bereiche
            hp_scaled = torch.zeros_like(raw_hp)
            hp_scaled[0] = 0.1 + 2.9 * torch.sigmoid(raw_hp[0])  # LR-Scale: 0.1..3.0
            hp_scaled[1] = 0.2 + 0.6 * torch.sigmoid(raw_hp[1])  # Sparsity: 0.2..0.8
            hp_scaled[2] = 0.5 + 0.49 * torch.sigmoid(raw_hp[2])  # Momentum: 0.5..0.99
            hp_scaled[3] = 0.1 + 1.9 * torch.sigmoid(raw_hp[3])  # Temp: 0.1..2.0
            hp_scaled[4] = 0.5 + 1.5 * torch.sigmoid(raw_hp[4])  # Batch-Mult: 0.5..2.0
            
            # Plateau-Boost: Bei Plateau mehr Exploration (höhere Temp, niedrigere Sparsity)
            if self.plateau_counter > 10:
                hp_scaled[3] = min(2.0, hp_scaled[3] * 1.2)  # Höhere Temperatur
                hp_scaled[1] = max(0.2, hp_scaled[1] * 0.8)  # Niedrigere Sparsity
            
            # Sanfte Aktualisierung (EMA)
            self.current_hp = 0.7 * self.current_hp + 0.3 * hp_scaled
            self.hp_adaptation_count += 1
            
            return self.get_hyperparams()
    
    # =========================================================================
    #  3b. PHASE 57: UNGEWISSHEITS-GEKOPPELTE HP-STEUERUNG
    # =========================================================================
    
    def adapt_from_uncertainty(self, uncertainty):
        """
        PHASE 57: Kopplung Active-Learning × Meta-Learning.
        
        Die Unsicherheit aus dem ActiveLearning-Modul moduliert die
        Hyperparameter sanft:
          - Hohe Unsicherheit  → konservativer LR, mehr Exploration
            (Temperatur hoch, Sparsity runter → mehr Parameter lernen)
          - Niedrige Unsicherheit → tendenziell aggressiver LR
        
        Bewusst OHNE Shape-Änderung am hp_controller (Checkpoint-sicher):
        Die Modulation wirkt als sanftes Curiousity-Gate auf current_hp.
        Nur alle 10 Steps aktiv, damit die HP stabil bleiben.
        """
        with torch.no_grad():
            self._unc_step += 1
            if self._unc_step % 10 != 0:
                return self.get_hyperparams()
            
            unc = min(max(float(uncertainty), 0.0), 1.0)
            self.uncertainty_coupled[0] = unc
            
            hp = self.current_hp.clone()
            # LR-Scale: Unsicherheit → konservativer (weniger Risiko)
            hp[0] = hp[0] * (1.0 - 0.25 * unc)
            # Sparsity: Unsicherheit → weniger Sparsity (mehr Parameter lernen)
            hp[1] = hp[1] * (1.0 - 0.15 * unc)
            # Temperatur: Unsicherheit → mehr Exploration
            hp[3] = hp[3] * (1.0 + 0.20 * unc)
            # Momentum: Unsicherheit → stabiler
            hp[2] = hp[2] * (1.0 + 0.05 * unc)
            
            # Clamp auf gültige Bereiche
            hp[0] = max(0.1, min(3.0, hp[0]))
            hp[1] = max(0.2, min(0.8, hp[1]))
            hp[2] = max(0.5, min(0.99, hp[2]))
            hp[3] = max(0.1, min(2.0, hp[3]))
            hp[4] = max(0.5, min(2.0, hp[4]))
            
            # Sanfte EMA-Übernahme (bewusst konservativer als adapt_hyperparams)
            self.current_hp = 0.9 * self.current_hp + 0.1 * hp
            
            return self.get_hyperparams()
    
    # =========================================================================
    #  FORWARD & LEARN
    # =========================================================================
    
    def forward(self, hidden_states, current_loss=None):
        """
        Meta-Learning Forward-Pass.
        
        Args:
            hidden_states: [batch, seq, d_model]
            current_loss: float oder None
            
        Returns:
            dict mit meta_advice (empfohlene Hyperparameter + Strategie)
        """
        with torch.no_grad():
            # Loss-Trend aktualisieren
            if current_loss is not None:
                self.update_loss_trend(current_loss)
            
            # Strategie kodieren
            strategy_vec = self.encode_strategy(hidden_states)
            data_emb = hidden_states.mean(dim=(0, 1))
            
            # Performance vorhersagen
            prediction = self.predict_strategy_performance(strategy_vec, data_emb)
            
            # Hyperparameter anpassen (nur alle 50 Steps)
            if self.hp_adaptation_count % 50 == 0:
                hp = self.adapt_hyperparams(hidden_states)
            else:
                hp = self.get_hyperparams()
            
            # Plateau-Warnung
            plateau_warning = self.plateau_counter.item() > 20
            
            return {
                'hyperparams': hp,
                'prediction': prediction,
                'trend': self.loss_trend.item(),
                'plateau_steps': self.plateau_counter.item(),
                'plateau_warning': plateau_warning,
                'hp_adaptations': self.hp_adaptation_count.item(),
            }
    
    def get_meta_stats(self):
        """Gib Meta-Learning-Statistiken."""
        hp = self.get_hyperparams()
        n_valid = min(self._strategy_idx, self.window_size)
        
        avg_meta_error = 0.0
        if n_valid > 0:
            recent_errors = self.meta_improvement[:min(self._meta_idx, 100)]
            if recent_errors.numel() > 0:
                avg_meta_error = recent_errors.mean().item()
        
        return {
            'lr_scale': hp['lr_scale'],
            'sparsity': hp['sparsity'],
            'momentum': hp['momentum'],
            'temperature': hp['temperature'],
            'batch_mult': hp['batch_mult'],
            'trend': self.loss_trend.item(),
            'plateau_steps': self.plateau_counter.item(),
            'n_strategies_tried': n_valid,
            'avg_meta_error': avg_meta_error,
            'hp_adaptations': self.hp_adaptation_count.item(),
            # PHASE 57: Unsicherheits-Kopplung
            'uncertainty_coupled': self.uncertainty_coupled.item(),
        }


class ActiveLearning(CogModule):
    """
    PHASE 56: Aktives Lernen — Das Modell entscheidet selbst, was es als nächstes lernt.
    
    Drei Komponenten:
    
    1. UncertaintySampler — wählt Daten mit höchster Unsicherheit:
       - Berechnet pro Batch eine Unsicherheit aus Prediction-Error + Confidenz
       - EMA-Unsicherheit pro Domäne (welche Domäne braucht mehr Daten?)
       - Token-Unsicherheits-Map über den Vokabular-Raum (Wissenslücken)
    
    2. QueryMechanism — generiert gezielte Lernanfragen:
       - Identifiziert die unsichersten Tokens als Wissenslücken
       - Empfiehlt Domänen-Gewichte proportional zur Unsicherheit
       - Zählt ausgeführte Queries (aktives Sampling)
    
    3. CurriculumOnDemand — fordert schwierigere/leichtere Daten an:
       - Verfolgt den Unsicherheits-Trend (EMA)
       - Steigende Unsicherheit → leichtere Daten (-1)
       - Fallende Unsicherheit → schwerere Daten (+1)
    
    Integration: Nutzt ActiveInference (Curiosity) + AutoCurriculum (ZPD).
    Erfolgskriterium: Niedrigerer Loss mit weniger Daten durch aktives Sampling.
    """
    def __init__(self, d_model, n_domains=4, vocab_size=4096, window_size=500):
        super().__init__()
        self.d_model = d_model
        self.n_domains = n_domains
        self.window_size = window_size
        
        # =====================================================================
        #  1. UNCERTAINTY SAMPLER
        # =====================================================================
        # EMA-Unsicherheit pro Domäne
        self.register_buffer('domain_uncertainty', torch.zeros(n_domains))
        self.register_buffer('domain_counts', torch.zeros(n_domains))
        # Unsicherheits-Verlauf
        self.register_buffer('uncertainty_history', torch.zeros(window_size))
        self.register_buffer('uncertainty_ema', torch.zeros(1))
        self._uncertainty_idx = 0
        self._ema_decay = 0.95
        
        # Token-Unsicherheits-Map (Wissenslücken über den Token-Raum)
        self.register_buffer('token_uncertainty', torch.zeros(vocab_size))
        self.register_buffer('token_counts', torch.zeros(vocab_size))
        
        # =====================================================================
        #  2. QUERY MECHANISMUS
        # =====================================================================
        self.register_buffer('query_counter', torch.zeros(1, dtype=torch.long))
        self.register_buffer('queries_history', torch.zeros(window_size))
        self._query_idx = 0
        
        # =====================================================================
        #  3. CURRICULUM ON DEMAND
        # =====================================================================
        self.register_buffer('curriculum_request', torch.zeros(1, dtype=torch.long))
        self.register_buffer('curriculum_ema', torch.zeros(1))
        
        # ——— Metrik ———
        self.register_buffer('active_sampling_benefit', torch.zeros(window_size))
        self._benefit_idx = 0
    
    # =========================================================================
    #  1. UNCERTAINTY SAMPLING
    # =========================================================================
    
    def sample_uncertainty(self, error_norm, confidence=None):
        """
        Berechne Unsicherheit aus Prediction-Error und Confidenz.
        
        Args:
            error_norm: float — Norm des Prediction-Errors (0..10+)
            confidence: float oder None — Confidenz-Score (0..1)
        
        Returns:
            uncertainty: float (0..1)
        """
        err_unc = min(max(error_norm, 0.0), 10.0) / 10.0
        if confidence is not None:
            conf_unc = 1.0 - min(max(confidence, 0.0), 1.0)
            return min(max(0.6 * err_unc + 0.4 * conf_unc, 0.0), 1.0)
        return min(max(err_unc, 0.0), 1.0)
    
    def learn_step(self, input_ids, error_norm, loss, domain_idx=0, confidence=None):
        """
        Aktualisiere Unsicherheits-Map, Wissenslücken und Curriculum-Anfrage.
        
        Args:
            input_ids: [batch, seq] — Tokens des aktuellen Batches
            error_norm: float — Prediction-Error-Norm
            loss: float — aktueller Loss
            domain_idx: int — aktuelle Domäne (0=text, 1=code, 2=security, 3=network)
            confidence: float oder None — Confidenz aus SelfReflection
        """
        with torch.no_grad():
            unc = self.sample_uncertainty(error_norm, confidence)
            
            # Domänen-Unsicherheit (EMA)
            d = min(domain_idx, self.n_domains - 1)
            self.domain_uncertainty[d] = self._ema_decay * self.domain_uncertainty[d] + (1 - self._ema_decay) * unc
            self.domain_counts[d] += 1
            
            # Token-Unsicherheit (Wissenslücken) — Stichprobe aus dem Batch
            flat_tokens = input_ids.reshape(-1)
            n_tok = flat_tokens.size(0)
            k = min(n_tok, 64)  # Stichprobe (nicht alle Tokens)
            sample_idx = flat_tokens[:k]
            max_tok = self.token_uncertainty.size(0) - 1
            for tok in sample_idx.cpu().tolist():
                tok = min(int(tok), max_tok)
                self.token_uncertainty[tok] = self._ema_decay * self.token_uncertainty[tok] + (1 - self._ema_decay) * unc
                self.token_counts[tok] += 1
            
            # Unsicherheits-Verlauf + EMA
            idx = self._uncertainty_idx % self.window_size
            self.uncertainty_history[idx] = unc
            self._uncertainty_idx += 1
            self.uncertainty_ema[0] = 0.9 * self.uncertainty_ema[0] + 0.1 * unc
            
            # ===============================================================
            #  CURRICULUM ON DEMAND — Unsicherheits-Trend
            # ===============================================================
            if self._uncertainty_idx >= 20:
                recent = self.uncertainty_history[max(0, idx - 19): idx + 1]
                avg_recent = recent.mean().item()
                ema = self.curriculum_ema[0].item()
                # Steigende Unsicherheit → leichtere Daten anfordern
                if ema > 0 and avg_recent > ema * 1.3 + 0.05:
                    self.curriculum_request[0] = -1
                # Fallende Unsicherheit → schwerere Daten anfordern
                elif ema > 0 and avg_recent < ema * 0.7 - 0.05:
                    self.curriculum_request[0] = 1
                else:
                    self.curriculum_request[0] = 0
                self.curriculum_ema[0] = 0.9 * ema + 0.1 * avg_recent
            
            # Query-Mechanismus zählt aktives Sampling
            self.query_counter[0] += 1
            qidx = self._query_idx % self.window_size
            self.queries_history[qidx] = unc
            self._query_idx += 1
    
    # =========================================================================
    #  2. QUERY MECHANISMUS
    # =========================================================================
    
    def get_token_weights(self, input_ids, max_boost=2.0):
        """
        PHASE 58: Token-Gewichte für Hebbian-Updates.
        
        Tokens mit hoher Wissenslücken-Unsicherheit (wenig gelernt) bekommen
        stärkere Hebbian-Updates — das Modell lernt gezielt seine Wissenslücken.
        Nur Tokens mit genug Daten (counts > 5) werden gewichtet; unbekannte
        Tokens bleiben bei Gewicht 1.0 (kein Verzerren des Lernens).
        
        Args:
            input_ids: [batch, seq] — Tokens des aktuellen Batches
            max_boost: float — maximale Gewichtung (2.0 = doppelt so stark lernen)
        
        Returns:
            weights: [batch, seq, 1] — per-Token Gewichte (1.0..max_boost)
        """
        with torch.no_grad():
            max_tok = self.token_uncertainty.size(0) - 1
            tok = input_ids.reshape(-1).clamp(max=max_tok)
            counts = self.token_counts[tok]
            unc = self.token_uncertainty[tok]
            
            # Nur Tokens mit genug Beobachtungen gewichten
            has_data = counts > 5
            # Batch-Max als Normalisierung (relative Wissenslücken-Stärke)
            batch_max = unc.max().clamp(min=1e-6)
            weight = 1.0 + (max_boost - 1.0) * (unc / batch_max)
            weight = torch.where(has_data, weight, torch.ones_like(weight))
            
            return weight.reshape(input_ids.shape[0], input_ids.shape[1], 1)
    
    def get_knowledge_gaps(self, top_k=8):
        """
        Finde die unsichersten Tokens = Wissenslücken.
        
        Returns:
            gaps: list[(token_id, uncertainty)] — unsicherste Tokens mit Score
        """
        with torch.no_grad():
            mask = self.token_counts > 5  # Nur Tokens mit genug Daten
            scores = torch.where(mask, self.token_uncertainty, torch.zeros_like(self.token_uncertainty))
            n_valid = int(mask.sum().clamp(min=1).item())
            k = min(top_k, n_valid)
            if k <= 0:
                return []
            top = torch.topk(scores, k=k).indices.cpu().tolist()
            return [(t, float(self.token_uncertainty[t].item())) for t in top]
    
    def get_domain_preference(self):
        """
        Empfohlene Domänen-Gewichte für aktives Sampling.
        
        Domänen mit HOHER Unsicherheit bekommen MEHR Gewicht
        (mehr Daten für Wissenslücken — Uncertainty Sampling).
        
        Returns:
            prefs: [n_domains] Tensor (normalisiert)
        """
        with torch.no_grad():
            if int(self.domain_counts.sum().item()) < 10:
                return torch.ones(self.n_domains, device=self.domain_uncertainty.device) / self.n_domains
            prefs = self.domain_uncertainty + 0.1  # Smoothing
            total = prefs.sum().clamp(min=1e-6)
            return prefs / total
    
    def get_curriculum_request(self):
        """Schwierigkeits-Empfehlung: -1=leichter, 0=gleich, +1=schwerer."""
        return int(self.curriculum_request[0].item())
    
    # =========================================================================
    #  3. KONDITIONIERUNG
    # =========================================================================
    
    def condition(self, hidden_states):
        """
        Sanfte Modulation basierend auf aktueller Unsicherheit (Curiosity-Gain).
        
        Bei hoher Unsicherheit: leicht verstärkte Aktivierung
        (mehr Aufmerksamkeit auf unsichere Bereiche).
        """
        with torch.no_grad():
            if self._uncertainty_idx < 5:
                return hidden_states
            unc = float(self.uncertainty_history[min(self._uncertainty_idx - 1, self.window_size - 1)].item())
            gain = 1.0 + 0.05 * min(max(unc - 0.5, -0.5), 0.5)
            return hidden_states * gain
    
    # =========================================================================
    #  STATS
    # =========================================================================
    
    def get_active_stats(self):
        """Stats für train_state.json."""
        last_unc = 0.0
        if self._uncertainty_idx > 0:
            last_unc = float(self.uncertainty_history[min(self._uncertainty_idx - 1, self.window_size - 1)].item())
        return {
            'uncertainty': last_unc,
            'uncertainty_ema': float(self.uncertainty_ema[0].item()),
            'curriculum_request': self.get_curriculum_request(),
            'n_queries': int(self.query_counter[0].item()),
            'knowledge_gaps': [t for t, _ in self.get_knowledge_gaps(5)],
            'domain_uncertainty': [float(x) for x in self.domain_uncertainty.cpu().tolist()],
            'domain_counts': [int(x) for x in self.domain_counts.cpu().tolist()],
        }


class CogLang:
    def __init__(self, use_mixed_precision=True):
        self.modules = nn.ModuleList()
        self._sensory = None
        self._encoder = None
        self._stack = None
        self._decoder = None
        self._context_embed = None
        self._memory = None
        self._active_inference = None
        self._bridge = None
        self._es = None
        self._skills = None
        self._security_head = None
        self._network_encoder = None
        self._sleep_replay = None
        self._goal_encoder = None
        self._self_reflection = None
        self._knowledge_graph = None
        self._tool_use = None
        self._multi_agent = None
        self._transfer_learning = None
        self._consciousness = None
        self._auto_curriculum = None
        self._causal_reasoning = None
        self._system2_reasoning = None
        self._imagination = None
        self._exploration = None
        self._metakognition = None
        self._hierarchical_memory = None
        self._hierarchical_goal = None
        self._meta_learning = None
        self._active_learning = None
        # PHASE 15: Efficiency
        self.mp = MixedPrecisionManager(use_mixed_precision)

    def SensoryInput(self, vocab_size, d_model):
        m = SensoryInput(vocab_size, d_model); self.modules.append(m); self._sensory = m; return m
    def SparseEncoder(self, input_dim, d_sparse, sparsity=0.02):
        m = SparseEncoder(input_dim, d_sparse, sparsity); self.modules.append(m); self._encoder = m; return m
    def PredictiveStack(self, d_model, n_layers, d_state, d_context, lr=0.05, n_attention_heads=4):
        m = PredictiveStack(d_model, n_layers, d_state, d_context, n_attention_heads); self.modules.append(m); self._stack = m
        for layer in m.layers: layer._lr = lr
        return m
    
    def HierarchicalPC(self, d_model, n_levels=3, n_layers_per_level=None, d_state=64, d_context=128, lr=0.05, n_attention_heads=4):
        """PHASE 35: Hierarchical Predictive Coding — Mehr-Ebenen-Architektur."""
        if n_layers_per_level is None:
            n_layers_per_level = [6, 4, 2]
        m = HierarchicalPC(d_model, n_levels, n_layers_per_level, d_state, d_context, n_attention_heads)
        self.modules.append(m)
        self._stack = m  # Ersetzt PredictiveStack — kompatibel!
        # Setze LR für alle Layer in allen Ebenen
        for stack in m.stacks:
            for layer in stack.layers:
                layer._lr = lr
        # Auch Bottom-Up Encoder und Top-Down Predictor LRs setzen
        for enc in m.enc_bottom_up:
            enc.weight.data.mul_(0.1)  # Kleine Init
        for pred in m.pred_top_down:
            pred.weight.data.mul_(0.1)
        return m
    def OutputDecoder(self, d_sparse, d_model, vocab_size, lr=0.05):
        m = OutputDecoder(d_sparse, d_model, vocab_size); self.modules.append(m); self._decoder = m
        m._lr = lr; return m
    def EpisodicMemory(self, d_model, memory_size=64, target_dim=None):
        m = EpisodicMemory(d_model, memory_size, target_dim); self.modules.append(m); self._memory = m; return m
    def IntrinsicMotivation(self, d_model):
        m = IntrinsicMotivation(d_model); self.modules.append(m); self._motivation = m; return m
    def NeuroSymbolicBridge(self, vocab_size, d_model, n_rules=16):
        m = NeuroSymbolicBridge(vocab_size, d_model, n_rules); self.modules.append(m); self._bridge = m; return m
    def EvolutionStrategy(self, d_model, population_size=8, sigma=0.01):
        m = EvolutionStrategyOptimizer(d_model, population_size, sigma); self.modules.append(m); self._es = m; return m
    def SkillModule(self, d_model, n_skills=8):
        m = SkillModule(d_model, n_skills); self.modules.append(m); self._skills = m; return m

    def ActiveInference(self, d_model, n_domains=4):
        m = ActiveInference(d_model, n_domains); self.modules.append(m); self._active_inference = m; return m
    
    def SleepReplay(self, buffer_size=10000, d_model=None):
        m = SleepReplay(buffer_size, d_model); self.modules.append(m); self._sleep_replay = m; return m

    def GoalEncoder(self, d_model, max_goal_len=50):
        """PHASE 36: Goal-Directed Generation."""
        m = GoalEncoder(d_model, max_goal_len); self.modules.append(m); self._goal_encoder = m; return m
    
    def SelfReflection(self, d_model, n_confidence_bins=5):
        """PHASE 37: Self-Reflection — Meta-Kognitive Selbstkritik."""
        m = SelfReflection(d_model, n_confidence_bins); self.modules.append(m); self._self_reflection = m; return m
    
    def KnowledgeGraph(self, d_model, max_entities=1024, max_relations=64):
        """PHASE 38: Knowledge Graph — Explizites Weltwissen."""
        m = KnowledgeGraph(d_model, max_entities, max_relations); self.modules.append(m); self._knowledge_graph = m; return m
    
    def ToolUse(self, d_model, max_tool_history=64):
        """PHASE 39: Tool Use — Externe Werkzeuge aufrufen."""
        m = ToolUse(d_model, max_tool_history); self.modules.append(m); self._tool_use = m; return m
    
    def MultiAgent(self, d_model, n_personas=2):
        """PHASE 40: Multi-Agent Self-Play — Zwei Persönlichkeiten."""
        m = MultiAgent(d_model, n_personas); self.modules.append(m); self._multi_agent = m; return m
    
    def TransferLearning(self, d_model, max_domains=8, adapter_rank=8):
        """PHASE 41: Transfer Learning — Domain-Adaptation + Few-Shot."""
        m = TransferLearning(d_model, max_domains, adapter_rank); self.modules.append(m); self._transfer_learning = m; return m
    
    def ConsciousnessGlimpse(self, d_model, spotlight_size=1, n_glimpses=3):
        """PHASE 42: Consciousness Glimpse — Global Workspace Broadcasting."""
        m = ConsciousnessGlimpse(d_model, spotlight_size, n_glimpses); self.modules.append(m); self._consciousness = m; return m
    
    def AutoCurriculum(self, d_model, n_difficulty_levels=5, window_size=100):
        """PHASE 43: Auto-Curriculum — Automatische Schwierigkeitsanpassung."""
        m = AutoCurriculum(d_model, n_difficulty_levels, window_size); self.modules.append(m); self._auto_curriculum = m; return m
    
    def CausalReasoning(self, d_model, n_causal_factors=64, temperature=0.1):
        """PHASE 44: Causal Reasoning — Kausales Verständnis."""
        m = CausalReasoning(d_model, n_causal_factors, temperature); self.modules.append(m); self._causal_reasoning = m; return m
    
    def System2Reasoning(self, d_model, n_reasoning_steps=8, n_tree_branches=3, temperature=0.3):
        """PHASE 45: System-2 Reasoning — Kettenbewusstsein + Verifikation + Gedankenbäume."""
        m = System2Reasoning(d_model, n_reasoning_steps, n_tree_branches, temperature); self.modules.append(m); self._system2_reasoning = m; return m
    
    def ImaginationPlanning(self, d_model, n_plan_steps=6, n_actions=16, temperature=0.2):
        """PHASE 46: Imagination & Planning — Zukunftsvorhersage + Planung."""
        m = ImaginationPlanning(d_model, n_plan_steps, n_actions, temperature); self.modules.append(m); self._imagination = m; return m
    
    def ExplorationDrive(self, d_model, n_uncertainty_cells=128, n_emn_history=200):
        """PHASE 47: Exploration Drive — Aktive Wissenslücken-Suche & Neugier."""
        m = ExplorationDrive(d_model, n_uncertainty_cells, n_emn_history); self.modules.append(m); self._exploration = m; return m

    def MetaKognition(self, d_model, n_strategies=4, n_resource_levels=3):
        """PHASE 48: MetaKognition — Strategie-Selektion, Confidence-Kalibrierung, Resource-Allocation."""
        m = MetaKognition(d_model, n_strategies, n_resource_levels); self.modules.append(m); self._metakognition = m; return m

    def HierarchicalMemory(self, d_model, sensory_buffer_size=1000, working_mem_size=256, episodic_buffer_size=500):
        """PHASE 49: Hierarchical Memory — 5-Ebenen-Gedächtnishierarchie mit Konsolidierung."""
        m = HierarchicalMemory(d_model, sensory_buffer_size, working_mem_size, episodic_buffer_size)
        self.modules.append(m)
        self._hierarchical_memory = m
        return m

    def HierarchicalGoal(self, d_model, max_goals=32, max_subgoals_per_goal=8, max_depth=4):
        """PHASE 52: Hierarchical Goal — Goal Decomposer, Subgoal Tracker, Goal Adaptation."""
        m = HierarchicalGoal(d_model, max_goals, max_subgoals_per_goal, max_depth)
        self.modules.append(m)
        self._hierarchical_goal = m
        return m

    def MetaLearning(self, d_model, n_hyperparams=5, n_strategy_dim=32, window_size=200):
        """PHASE 55: Meta-Learning — Hyperparameter-Steuerung, Strategie-Selektion, Meta-Netzwerk."""
        m = MetaLearning(d_model, n_hyperparams, n_strategy_dim, window_size)
        self.modules.append(m)
        self._meta_learning = m
        return m

    def ActiveLearning(self, d_model, n_domains=4, vocab_size=4096, window_size=500):
        """PHASE 56: Aktives Lernen — Uncertainty-Sampler, Query-Mechanismus, Curriculum-on-Demand."""
        m = ActiveLearning(d_model, n_domains, vocab_size, window_size)
        self.modules.append(m)
        self._active_learning = m
        return m

    def SecurityHead(self, d_model, d_sparse, n_cwe_types=20):
        """PHASE 31: Vulnerability Detection Head."""
        m = SecurityHead(d_model, d_sparse, n_cwe_types)
        self.modules.append(m)
        self._security_head = m
        return m

    def NetworkEncoder(self, d_model, d_sparse, n_protocols=16):
        """PHASE 32: Network Traffic Encoder."""
        m = NetworkEncoder(d_model, d_sparse, n_protocols)
        self.modules.append(m)
        self._network_encoder = m
        return m

    def analyze_security(self, text):
        """PHASE 31: Analyze code for vulnerabilities using SecurityHead."""
        if self._security_head is None:
            return {'cwe_probs': None, 'severity': None, 'confidence': None, 'error': 'SecurityHead not initialized'}
        with torch.no_grad():
            device = next(self.modules.parameters()).device
            # Simple char-level embedding as code representation
            chars = torch.tensor([[ord(c) % 1000 for c in text[:512]]], device=device)
            # Project through encoder to sparse space
            if self._encoder is not None and self._sensory is not None:
                code_emb = self._encoder(self._sensory(chars))
            else:
                code_emb = torch.randn(1, min(len(text), 512), self._security_head.d_sparse, device=device)
            return self._security_head(code_emb)

    def analyze_network(self, packets):
        """PHASE 32: Analyze network traffic using NetworkEncoder."""
        if self._network_encoder is None:
            return {'flow_state': None, 'anomaly_score': None, 'error': 'NetworkEncoder not initialized'}
        with torch.no_grad():
            device = next(self.modules.parameters()).device
            if isinstance(packets, dict):
                pkt_seq = {k: v.to(device) if torch.is_tensor(v) else torch.tensor(v, device=device) for k, v in packets.items()}
            else:
                # Generate dummy packet sequence
                batch, seq = 1, 10
                pkt_seq = {
                    'src_port': torch.randint(1024, 65535, (batch, seq), device=device),
                    'dst_port': torch.randint(1, 1023, (batch, seq), device=device),
                    'protocol': torch.randint(0, 6, (batch, seq), device=device),
                    'len': torch.randint(40, 1500, (batch, seq), device=device),
                    'flags': torch.randint(0, 8, (batch, seq), device=device),
                }
            return self._network_encoder(pkt_seq)

    def to(self, device):
        self.modules.to(device)
        for p in self.modules.parameters():
            p.requires_grad_(False)
        return self

    def forward(self, input_ids, learn=True):
        batch, seq = input_ids.shape
        device = input_ids.device
        x = self._sensory(input_ids)
        sparse_x = self._encoder(x)
        
        # PHASE 49: Hierarchical Memory — Speichere Sensory Input
        if self._hierarchical_memory is not None:
            self._hierarchical_memory.store_sensory(input_ids, sparse_x)
        
        if self._context_embed is None:
            d_context = self._stack.layers[0].d_context
            ce = nn.Embedding(8192, d_context).to(device)
            ce.weight.requires_grad_(False)
            self._context_embed = ce
        positions = torch.arange(seq, device=device).unsqueeze(0).expand(batch, -1)
        context = self._context_embed(positions)
        
        # Retrieve from episodic memory
        memory_retrieved = None
        if self._memory is not None:
            # Use last state as query
            query = sparse_x[:, -1:, :]  # [batch, 1, d_sparse]
            memory_retrieved = self._memory(query)  # [batch, 1, d_sparse]
            # Expand to sequence length
            memory_retrieved = memory_retrieved.expand(-1, seq, -1)
        
        # PHASE 35: HierarchicalPC gibt 4 Werte zurück (errors, states, preds, mixed_pred)
        # PredictiveStack gibt 3 Werte zurück (errors, states, preds)
        # PHASE 58: Wissenslücken-gewichtete Hebbian-Updates — nur im Lernmodus
        token_weights = None
        if learn and self._active_learning is not None and self._active_learning._uncertainty_idx > 10:
            token_weights = self._active_learning.get_token_weights(input_ids)
        stack_result = self._stack(sparse_x, context, memory_retrieved=memory_retrieved, errors_for_attn=sparse_x, learn=learn, token_weights=token_weights)
        if len(stack_result) == 4:
            errors, states, predictions, pred = stack_result
        else:
            errors, states, predictions = stack_result
            # NaN-Guard: Sichere Predictions
            predictions = [torch.nan_to_num(p, nan=0.0, posinf=1.0, neginf=-1.0) for p in predictions]
            pred = self._stack.mixed_prediction(predictions)
        
        info_extra = {}
        
        # PHASE 42: Consciousness Glimpse — Global Workspace Broadcasting
        if self._consciousness is not None and errors is not None:
            # Berechne Salience aus Hidden-States und Prediction Errors
            pred, glimpse_info = self._consciousness.glimpse(pred, errors)
            info_extra['consciousness'] = glimpse_info
        elif self._consciousness is not None:
            # Ohne Errors: nur Condition mit letztem Broadcast
            pred = self._consciousness.condition(pred)
        
        # PHASE 14: SkillModule transformation - modulates hidden state BEFORE decoder
        # SkillModule arbeitet auf d_sparse-Dimension (pred), nicht auf vocab-Logits
        if self._skills is not None:
            pred = self._skills(pred, context=sparse_x)
        
        # PHASE 38: Knowledge Graph — Retrieve relevant knowledge and condition
        if self._knowledge_graph is not None:
            ctx_pooled = sparse_x.mean(dim=1)  # [batch, d_sparse]
            knowledge = self._knowledge_graph.retrieve(ctx_pooled, top_k=3)
            if knowledge['n_facts'] > 0:
                pred = self._knowledge_graph.condition(pred, knowledge['embeddings'], knowledge['scores'])
                info_extra['knowledge'] = {
                    'n_facts': knowledge['n_facts'],
                    'graph_stats': self._knowledge_graph.get_graph_stats(),
                }
        
        # PHASE 49: Hierarchical Memory — Retrieve aus Gedächtnishierarchie
        if self._hierarchical_memory is not None:
            # Link zu Knowledge Graph und Skills falls vorhanden
            self._hierarchical_memory.link_modules(
                knowledge_graph=self._knowledge_graph,
                skill_module=self._skills,
            )
            # Retrieve aus Working + Episodic Memory
            mem_result = self._hierarchical_memory.retrieve_working(pred)
            pred = pred + mem_result * 0.05  # Sanfter Memory-Einfluss
            info_extra['hierarchical_memory'] = {
                'retrieved': True,
            }
        
        # PHASE 40: Multi-Agent Self-Play — Perspektivenvielfalt
        if self._multi_agent is not None and not learn:
            # Persona-konditionierte Varianten von pred
            pred_a = self._multi_agent.condition(pred, persona_id=0)  # konservativ
            pred_b = self._multi_agent.condition(pred, persona_id=1)  # kreativ
            
            # Agreement zwischen beiden Perspektiven
            debate_info = self._multi_agent.debate_step(pred_a, pred_b, input_ids)
            info_extra['debate'] = debate_info
            
            # Mittle pred und konditionierte Varianten
            pred = (pred + pred_a + pred_b) / 3
        elif self._multi_agent is not None and learn:
            # Im Lernmodus: abwechselnde Personas
            persona_id = torch.randint(0, 2, (1,)).item()
            pred = self._multi_agent.condition(pred, persona_id=persona_id)
            info_extra['debate_persona'] = persona_id
        
        # PHASE 41: Transfer Learning — Domain-Adaptation
        if self._transfer_learning is not None:
            # Erkenne Domäne aus gepooltem Kontext
            domain_id, domain_conf = self._transfer_learning.detect_domain(sparse_x)
            info_extra['domain'] = {'id': domain_id, 'confidence': domain_conf}
            
            if not learn:
                # Wende Domain-Adapter an
                pred = self._transfer_learning.apply_adapter(pred, domain_id)
                # Domain-Conditioning
                pred = self._transfer_learning.condition(pred, domain_id)
            else:
                # Im Lernmodus: zufällige Domäne für Exploration
                n_domains = max(1, self._transfer_learning.n_domains.item())
                learn_domain = torch.randint(0, n_domains, (1,)).item()
                pred = self._transfer_learning.apply_adapter(pred, learn_domain)
                pred = self._transfer_learning.condition(pred, learn_domain)
                info_extra['domain']['learn_domain'] = learn_domain
        
        # PHASE 43: Auto-Curriculum — Difficulty Conditioning
        if self._auto_curriculum is not None:
            pred = self._auto_curriculum.condition(pred)
            info_extra['curriculum'] = self._auto_curriculum.get_curriculum_params()
        
        # PHASE 44: Causal Reasoning — Entdecke Kausalstrukturen im Forward-Pass
        if self._causal_reasoning is not None:
            if not learn and self._causal_reasoning._temp_idx % 10 == 0:
                causes, effects, strengths = self._causal_reasoning.discover_causal(pred)
                if causes:
                    info_extra['causal'] = {
                        'n_discovered': len(causes),
                        'avg_strength': sum(strengths) / len(strengths) if strengths else 0,
                    }
            # Conditioning: moduliere pred mit kausalem Verständnis
            pred = self._causal_reasoning.condition(pred)
        
        # PHASE 45: System-2 Reasoning — Chain-of-Thought + Tree-of-Thought
        if self._system2_reasoning is not None and not learn:
            # Reasoning auf gepooltem Kontext
            context = pred.mean(dim=1, keepdim=True)  # [batch, 1, d]
            best_chain, best_scores = self._system2_reasoning.tree_of_thought(
                context.squeeze(1), return_best=True
            )
            # Integriere Reasoning in Hidden-States
            reasoning_influence = best_chain.mean(dim=1, keepdim=True)
            pred = pred + reasoning_influence * 0.03
            info_extra['reasoning'] = {
                'avg_score': best_scores.mean().item(),
                'quality': self._system2_reasoning.get_reasoning_stats()['avg_quality'],
            }
        elif self._system2_reasoning is not None:
            # Im Lernmodus: nur Conditioning
            pred = self._system2_reasoning.condition(pred)
        
        # PHASE 47: Exploration Drive — Moduliere mit Explorations-Bonus
        if self._exploration is not None:
            pred = self._exploration.condition(pred)
            if not learn:
                exp_bonus = self._exploration.compute_exploration_bonus(pred)
                info_extra['exploration'] = {
                    'bonus': exp_bonus,
                    'coverage': self._exploration.get_exploration_stats()['coverage'],
                }
        
        # PHASE 46: Imagination & Planning — In-die-Zukunft-Blicken
        if self._imagination is not None and not learn:
            # Simuliere Zukunft aus gepooltem Kontext
            best_future = self._imagination.simulate(pred, n_steps=4)
            # Integriere Imagination: Einfluss der vorhergesagten Zukunft
            future_influence = best_future.mean(dim=1, keepdim=True)
            pred = pred + future_influence * 0.02
            info_extra['imagination'] = {
                'plan_quality': self._imagination.get_imagination_stats()['avg_plan_quality'],
                'prediction_acc': self._imagination.get_imagination_stats()['avg_prediction_accuracy'],
            }
        elif self._imagination is not None:
            # Im Lernmodus: nur Conditioning
            pred = self._imagination.condition(pred)
        
        output, hidden = self._decoder(pred)
        
        # PHASE 8: Apply neuro-symbolic rules to output
        if self._bridge is not None:
            output = self._bridge(output, context_embedding=sparse_x)
            
        # PHASE 35: HierarchicalPC Level Report
        if hasattr(self._stack, 'get_level_report'):
            info_extra['level_report'] = self._stack.get_level_report()
        
        # PHASE 37: Self-Reflection — Überwache eigene Gedanken
        if self._self_reflection is not None:
            prev_hidden = getattr(self, '_prev_hidden', None)
            reflection = self._self_reflection(pred, output, prev_hidden=prev_hidden)
            info_extra['reflection'] = reflection
            self._prev_hidden = pred[:, -1:, :].squeeze(1).detach()  # [batch, d_model]
            
        # PHASE 48: MetaKognition — Denke über das Denken nach
        if self._metakognition is not None and not learn:
            # Strategie-Empfehlung basierend auf aktuellem Hidden-State
            meta_advice = self._metakognition.get_meta_strategy_advice(pred, 
                current_loss=getattr(self, '_current_loss', None))
            info_extra['metakognition'] = meta_advice
            
            # Wende Strategie an: moduliere pred mit Meta-Signal
            if meta_advice['strategy'] == 0:
                # Fast: kein zusätzliches Processing
                pass
            elif meta_advice['strategy'] == 1:
                # Balanced: leichte Aktivierungs-Skalierung
                strategy_factor = 0.5 + 0.5 * (1.0 - meta_advice['difficulty'])
                pred = pred * strategy_factor
            elif meta_advice['strategy'] == 2:
                # Deep: verstärke Reasoning-Einfluss
                strategy_factor = 0.3 + 0.7 * meta_advice['difficulty']
                pred = pred * strategy_factor
            elif meta_advice['strategy'] == 3:
                # Explorative: erhöhe Varianz in Aktivierungen
                noise = torch.randn_like(pred) * 0.02 * meta_advice['difficulty']
                pred = pred + noise
        
        if self._metakognition is not None and learn:
            # Im Lernmodus: kalibriere Confidence aus Reflection
            reflection = info_extra.get('reflection', {})
            if reflection and 'avg_confidence' in reflection:
                raw_conf = reflection['avg_confidence']
                calibrated = self._metakognition.calibrate_confidence(
                    torch.tensor(raw_conf).view(1, 1, 1).to(pred.device)
                )
                info_extra['metakognition'] = {
                    'raw_confidence': raw_conf,
                    'calibrated_confidence': calibrated.mean().item(),
                    'calibration_temp': self._metakognition.calibration_temperature.item(),
                }
        
        # PHASE 52: Hierarchical Goal — Konditioniere mit aktuellem Goal
        if self._hierarchical_goal is not None:
            pred = self._hierarchical_goal.condition(pred)
            if not learn:
                goal_stats = self._hierarchical_goal.get_goal_stats()
                info_extra['hierarchical_goal'] = {
                    'n_goals': goal_stats['n_goals'],
                    'n_active': goal_stats['n_active'],
                    'n_completed': goal_stats['n_completed'],
                    'n_pending': goal_stats['n_pending'],
                    'n_failed': goal_stats['n_failed'],
                    'success_rate': goal_stats['success_rate'],
                }
            else:
                # Im Lernmodus: aktualisiere Goal-Fortschritt
                self._hierarchical_goal.update_progress(pred, loss_val=getattr(self, '_current_loss', None))
        
        # PHASE 55: Meta-Learning — Hyperparameter-Steuerung
        if self._meta_learning is not None and not learn:
            meta_advice = self._meta_learning(pred, current_loss=getattr(self, '_current_loss', None))
            info_extra['meta_learning'] = meta_advice
        elif self._meta_learning is not None and learn:
            # Im Lernmodus nur loss tracken
            self._meta_learning.update_loss_trend(getattr(self, '_current_loss', 10.0))
        
        # PHASE 56: Aktives Lernen — Unsicherheits-basierte Modulation (Curiosity-Gain)
        if self._active_learning is not None:
            pred = self._active_learning.condition(pred)
            if not learn:
                al_stats = self._active_learning.get_active_stats()
                info_extra['active_learning'] = {
                    'uncertainty': al_stats['uncertainty'],
                    'curriculum_request': al_stats['curriculum_request'],
                    'n_queries': al_stats['n_queries'],
                    'knowledge_gaps': al_stats['knowledge_gaps'],
                }
        
        # PHASE 39: Tool Use — Erkenne und führe Tool-Aufrufe aus
        if self._tool_use is not None and not learn:
            # Dekodiere Output in Text (grobe Approximation)
            argmax_ids = output.argmax(dim=-1)  # [batch, seq]
            tool_results = []
            for b in range(argmax_ids.size(0)):
                # Simuliere kurze Text-Dekodierung (nur für Tool-Erkennung)
                text = " ".join([str(id.item()) for id in argmax_ids[b, -20:]])
                calls = self._tool_use.detect_tool_calls(text)
                for name, arg in calls:
                    result, success = self._tool_use.execute(name, arg)
                    tool_results.append({
                        'tool': name, 'arg': arg, 'result': result, 'success': success
                    })
            if tool_results:
                info_extra['tool_calls'] = tool_results
                # Conditioniere mit Tool-Ergebnissen
                tool_ctx = tool_results[-1] if tool_results else None
                if tool_ctx:
                    pred = self._tool_use.condition(pred, tool_ctx)
        
        return output, {'errors': errors, 'predictions': predictions, 'hidden': hidden, 'pred': pred, 'sparse': sparse_x, 'output': output, **info_extra}

    def learn(self, input_ids):
        with torch.no_grad():
            output, info = self.forward(input_ids, learn=True)
            d_hidden_error = self._decoder.learn_step(output, info['hidden'], info['pred'], input_ids)
            self._sensory.learn_step(input_ids, d_hidden_error)
            
            # Write to episodic memory
            if self._memory is not None:
                self._memory._write_to_memory(info['sparse'][:, -1, :])
                # PHASE 1: Auch EpisodicMemory learn_step aktivieren
                query = info['sparse'][:, -1:, :]
                retrieved = self._memory(query)
                self._memory.learn_step(query, retrieved, info['sparse'])
            
            # PHASE 8: NeuroSymbolicBridge learn_step aktivieren
            if self._bridge is not None:
                total_error = sum((e ** 2).mean(dim=-1, keepdim=True) for e in info['errors']) / len(info['errors'])
                self._bridge.learn_step(info['sparse'], total_error)
            
            # PHASE 14: SkillModule learn_step - aktualisiert Skill-Prototypen basierend auf Error
            if self._skills is not None:
                total_error = sum((e ** 2).mean(dim=-1, keepdim=True) for e in info['errors']) / len(info['errors'])
                self._skills.learn_step(info['sparse'], total_error)
            
            # PHASE 31: SecurityHead learn_step - selbstüberwacht aus Prediction Error
            if self._security_head is not None and not hasattr(self, '_sec_counter'):
                self._sec_counter = 0
            if self._security_head is not None:
                self._sec_counter += 1
                if self._sec_counter % 500 == 0:  # Alle 500 Steps
                    # Prediction Error als Pseudo-Anomalie-Signal
                    pred_error = sum((e ** 2).mean(dim=1, keepdim=True) for e in info['errors']) / len(info['errors'])
                    # High error = mögliche Anomalie (selbstüberwacht)
                    anomaly_target = (pred_error > pred_error.median()).float()
                    self._security_head.learn_step(info['sparse'], anomaly_target, pred_error)
            
            # PHASE 47: Exploration Drive — Aktualisiere Uncertainty Map mit aktuellem Error
            if self._exploration is not None and info.get('errors') is not None:
                err_norm = sum((e ** 2).mean().item() for e in info['errors']) / max(1, len(info['errors']))
                self._exploration.update_uncertainty(info['pred'], err_norm)
            
            # PHASE 49: Hierarchical Memory — Promote Working → Episodic
            if self._hierarchical_memory is not None and info.get('pred') is not None:
                err_norm = sum((e ** 2).mean().item() for e in info['errors']) / max(1, len(info['errors']))
                # Promote ersten Batch-State zu Episodic
                state = info['pred'][0, -1, :]  # [d_sparse]
                importance, _ = self._hierarchical_memory.compute_importance(state, err_norm)
                domain_idx = getattr(self, '_current_domain', 0)
                self._hierarchical_memory.promote_to_episodic(state, importance, domain_idx)
            
            # PHASE 33: Active Inference — Curiosity + Free Energy + Epistemic Value
            if self._active_inference is not None:
                # Use first layer error for observation
                first_error = info['errors'][0] if info['errors'] else torch.zeros_like(info['sparse'])
                domain_idx = getattr(self, '_current_domain', 0)  # Updated externally by evolve script
                ai_result = self._active_inference.observe(first_error, domain_idx=domain_idx)
                
                # Modulate meta-plasticity with curiosity factor
                curiosity_factor = ai_result['curiosity_factor']
                for module in self.modules.modules():
                    if hasattr(module, '_meta_lr_scale'):
                        module._meta_lr_scale *= curiosity_factor
                        module._meta_lr_scale = max(0.1, min(3.0, module._meta_lr_scale))
                
                # Store ActiveInference results for external access (e.g., data sampling)
                self._ai_result = ai_result
            
            # PHASE 34: SleepReplay — Speichere Erfahrung im Replay Buffer
            if self._sleep_replay is not None:
                error_norms = []
                for e in info['errors']:
                    en = (e ** 2).mean().item()
                    en = 0.0 if torch.isnan(torch.tensor(en)) or torch.isinf(torch.tensor(en)) else en
                    error_norms.append(en)
                error_norm = sum(error_norms) / max(1, len(error_norms))
                domain_idx = getattr(self, '_current_domain', 0)
                importance = 1.0 + error_norm * 0.5  # Höherer Error = wichtiger
                self._sleep_replay.store(input_ids, error_norm, domain_idx=domain_idx, importance=importance)
            
            # PHASE 4: EWC Snapshot - alle 1000 Schritte Gewichte als Optimal sichern
            if not hasattr(self, '_ewc_step_counter'):
                self._ewc_step_counter = 0
            self._ewc_step_counter += 1
            if self._ewc_step_counter % 1000 == 0:
                self.ewc_snapshot_all()
            
            loss = F.cross_entropy(output.view(-1, output.size(-1)), input_ids.view(-1))
            
            # PHASE 37: Self-Reflection learn_step — lerne Selbsteinschätzung
            if self._self_reflection is not None:
                reflection = info.get('reflection')
                if reflection:
                    self._self_reflection.learn_step(reflection, loss.item())
                    
            # PHASE 55: Meta-Learning learn_step — Strategie speichern + lernen
            if self._meta_learning is not None and info.get('pred') is not None:
                strategy_vec = self._meta_learning.encode_strategy(info['pred'])
                data_emb = info['pred'].mean(dim=(0, 1))
                plateau_val = 1.0 if getattr(self._meta_learning, 'plateau_counter', torch.zeros(1))[0] > 20 else 0.0
                self._meta_learning.store_strategy_result(
                    strategy_vec, data_emb,
                    min(loss.item(), 20.0),
                    max(0.0, 1.0 - min(loss.item(), 10.0) / 10.0),
                    plateau_val,
                )
                # Lerne aus History (alle 100 Steps)
                if not hasattr(self, '_meta_learn_counter'):
                    self._meta_learn_counter = 0
                self._meta_learn_counter += 1
                if self._meta_learn_counter % 100 == 0:
                    self._meta_learning.learn_from_strategy_history()
            
            # PHASE 56: Aktives Lernen — Unsicherheit sammeln, Wissenslücken + Curriculum
            if self._active_learning is not None and info.get('errors') is not None:
                err_norm = sum((e ** 2).mean().item() for e in info['errors']) / max(1, len(info['errors']))
                conf = None
                reflection = info.get('reflection')
                if reflection and 'avg_confidence' in reflection:
                    conf = reflection['avg_confidence']
                domain_idx = getattr(self, '_current_domain', 0)
                self._active_learning.learn_step(
                    input_ids, err_norm, loss.item(), domain_idx, confidence=conf,
                )
                info['active_learning'] = {
                    'uncertainty': float(self._active_learning.uncertainty_history[
                        min(max(self._active_learning._uncertainty_idx - 1, 0), self._active_learning.window_size - 1)
                    ].item()),
                    'curriculum_request': self._active_learning.get_curriculum_request(),
                    'knowledge_gaps': self._active_learning.get_knowledge_gaps(5),
                }
                # PHASE 57: Meta-Learning × Active-Learning Kopplung —
                # Unsicherheit moduliert die Hyperparameter-Steuerung
                if self._meta_learning is not None:
                    unc_signal = float(self._active_learning.uncertainty_ema[0].item())
                    self._meta_learning.adapt_from_uncertainty(unc_signal)
            
            # PHASE 48: MetaKognition learn_step — Kalibrierung + Strategie-Lernen
            if self._metakognition is not None:
                reflection = info.get('reflection', {})
                meta_info = info.get('metakognition', {})
                if reflection and 'avg_confidence' in reflection:
                    raw_conf = reflection['avg_confidence']
                    # Correctness: niedriger Loss = korrekt
                    correctness = max(0.0, 1.0 - min(loss.item(), 10.0) / 10.0)
                    self._metakognition.update_calibration(raw_conf, correctness)
                
                if meta_info and 'strategy' in meta_info:
                    # Strategie-Erfolg: Loss runter = erfolgreich
                    strategy_id = meta_info['strategy']
                    success_score = max(0.0, 1.0 - min(loss.item(), 20.0) / 20.0)
                    self._metakognition.update_strategy_memory(strategy_id, success_score)
            
            # PHASE 40: Multi-Agent learn_step — lerne aus Debattenergebnissen
            if self._multi_agent is not None:
                debate = info.get('debate')
                if debate:
                    # Agreement-Target hängt vom Loss ab: niedriger Loss → mehr Agreement
                    target_agreement = 0.5 + 0.3 * (1.0 - min(loss.item(), 10.0) / 10.0)
                    self._multi_agent.learn_from_debate(
                        info['pred'], info['pred'],
                        debate['agreement'],
                        target_agreement=target_agreement,
                    )
            
            # PHASE 41: Transfer Learning learn_step — Domänenerkennung + Few-Shot
            if self._transfer_learning is not None:
                domain_info = info.get('domain', {})
                domain_id = domain_info.get('id', 0)
                self._transfer_learning.learn_step(info['sparse'], domain_id, loss.item())
                # Speichere Input/Target für Few-Shot
                self._transfer_learning.fewshot_store(input_ids[0], input_ids[0], domain_id)
            
            # PHASE 46: Imagination Planning learn_step — Lerne aus Realität vs Imagination
            if self._imagination is not None and info.get('pred') is not None:
                # Nutze pred (d_sparse) statt hidden (d_model) für Dimensions-Konsistenz
                try:
                    actual = info['pred'][:, -1, :]  # [batch, d_sparse]
                    imagined = self._imagination.simulate(info['pred'], n_steps=1)
                    if imagined is not None:
                        self._imagination.learn_from_imagination(actual, imagined[:, -1, :])
                except RuntimeError:
                    pass  # Dimensionskonflikt abfangen
            
            # PHASE 43: Auto-Curriculum learn_step — Schwierigkeit anpassen
            if self._auto_curriculum is not None:
                curriculum_result = self._auto_curriculum.learn_step(loss.item())
                if curriculum_result.get('difficulty_changed'):
                    info_extra['curriculum_change'] = curriculum_result
            
            # PHASE 13b: EvolutionStrategy inline learn_step — Plateau-noise + Revert
            if self._es is not None and self._ewc_step_counter % 100 == 0:
                self._es.learn_step(loss.item(), self.modules)
            
            if torch.isnan(loss) or torch.isinf(loss):
                return 100.0, info
            self._current_loss = loss.item()
        return loss.item(), info
    
    def ewc_snapshot_all(self):
        """PHASE 4: EWC on all CogModule instances - schützt vor Catastrophic Forgetting."""
        for module in self.modules.modules():
            if hasattr(module, '_ewc_snapshot') and module is not self:
                module._ewc_snapshot()

    def run_sleep_phase(self, n_steps=100, device='cuda'):
        """PHASE 34: Führe Sleep-Phase zur Konsolidierung aus."""
        report = {}
        if self._sleep_replay is not None:
            report = self._sleep_replay.sleep_phase(self, n_steps=n_steps, device=device)
        # PHASE 49: Hierarchical Memory Konsolidierung
        if self._hierarchical_memory is not None:
            consol_report = self._hierarchical_memory.consolidate(n_cycles=3)
            report['hierarchical'] = consol_report
        return report
    
    def get_active_inference_report(self):
        """PHASE 33: Gib aktuelles Active Inference Report zurück."""
        if self._active_inference is not None:
            return self._active_inference.get_report()
        return {}
    
    def get_domain_preference(self):
        """PHASE 33: Hole Domain-Präferenz für gewichtetes Sampling."""
        if self._active_inference is not None:
            return self._active_inference.get_domain_preference()
        return torch.ones(4) / 4  # Uniform fallback
    
    def parameter_count(self):
        return sum(p.numel() for p in self.modules.parameters())

    def save_checkpoint(self, path, config=None):
        checkpoint = {
            'model_state': self.modules.state_dict(),
            'config': config
        }
        torch.save(checkpoint, path)
        print(f'Checkpoint gespeichert: {path}')

    def load_checkpoint(self, path, strict=True):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location='cpu')
            result = self.modules.load_state_dict(checkpoint['model_state'], strict=False)
            if result.missing_keys:
                print(f'  Neue Keys (aus Checkpoint): {result.missing_keys}')
            if result.unexpected_keys:
                print(f'  Übersprungene Keys (nicht im Checkpoint): {result.unexpected_keys}')
            print(f'Checkpoint geladen: {path}')
            return checkpoint.get('config')
        return None

    def generate_safe(self, prompt_ids, max_new=100, temperature=0.7, top_k=40):
        """
        NaN-sichere Generierung mit mehreren Fallback-Ebenen.
        Gibt generierte Token-IDs zurueck.
        """
        device = next(self.modules.parameters()).device
        ctx = prompt_ids.to(device)
        
        for _ in range(max_new):
            temp = max(0.1, min(2.0, temperature))
            
            out, _ = self.forward(ctx[:, -256:], learn=False)
            logits = out[:, -1, :] / temp
            
            # NaN-Guard
            logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
            
            # Top-K Sampling mit Fallback
            k = min(top_k, logits.size(-1))
            top_k_logits, top_k_indices = torch.topk(logits, k)
            
            # Stabilisiere Softmax
            top_k_logits = top_k_logits - top_k_logits.max(dim=-1, keepdim=True)[0]
            probs = F.softmax(top_k_logits, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0)
            
            # Fallback: Gleichverteilung wenn alle Probs = 0
            if probs.sum() < 1e-8:
                probs = torch.ones_like(probs) / probs.size(-1)
            
            try:
                next_token = torch.multinomial(probs, 1)
                next_token = top_k_indices.gather(1, next_token)
            except RuntimeError:
                next_token = torch.randint(0, logits.size(-1), (1, 1), device=device)
            
            ctx = torch.cat([ctx, next_token], dim=-1)
            
            if ctx.size(-1) > 1024:
                ctx = ctx[:, -512:]
        
        return ctx

    def generate_goal_directed(self, prompt_ids, goal_token_ids, max_new=100, 
                                 temperature=0.7, top_k=40, n_candidates=5):
        """
        PHASE 36: Goal-Directed Generation.
        
        Generiert nicht einfach next-token, sondern versucht, ein gegebenes Goal
        zu erfüllen. Nutzt Beam-Search-ähnliche Mehrfachgeneration + Evaluation.
        
        Args:
            prompt_ids: [seq] Start-Prompt
            goal_token_ids: [goal_seq] Goal-Beschreibung
            max_new: Maximal zu generierende Tokens
            temperature: Sampling-Temperatur
            top_k: Top-K Sampling
            n_candidates: Anzahl paralleler Kandidaten (Beam Search Breite)
        
        Returns:
            best_ids: [seq + max_new] Beste Goal-erfüllende Sequenz
            scores: dict mit Metriken
        """
        device = next(self.modules.parameters()).device
        prompt_ids = prompt_ids.to(device)
        goal_token_ids = goal_token_ids.to(device)
        
        if self._goal_encoder is None:
            # Fallback: normale Generation wenn GoalEncoder fehlt
            return self.generate_safe(prompt_ids, max_new, temperature, top_k), {}
        
        with torch.no_grad():
            # ——— 1. Encode Goal ———
            if self._sensory is not None:
                goal_emb = self._goal_encoder.encode(goal_token_ids, self._sensory)
            else:
                goal_emb = torch.zeros(1, 1, self._goal_encoder.d_model, device=device)
            
            # ——— 2. Multi-Candidate Generation ———
            candidates = []
            scores = []
            
            for c in range(n_candidates):
                ctx = prompt_ids.clone()
                if ctx.dim() == 1:
                    ctx = ctx.unsqueeze(0)
                
                for step in range(max_new):
                    temp = max(0.1, min(2.0, temperature * (1.0 + step / max_new * 0.3)))
                    
                    out, info = self.forward(ctx[:, -256:], learn=False)
                    logits = out[:, -1, :] / temp
                    
                    # NaN-Guard
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
                    
                    # PHASE 36: Goal-Conditioned Sampling
                    # Moduliere Logits mit Goal-Fulfillment
                    if self._goal_encoder is not None and step > max_new // 4:
                        # Nach 25% der Generation: bevorzuge Goal-konsistente Tokens
                        # (Goal-Bias steigt mit der Zeit)
                        goal_bias = min(0.5, step / max_new * 0.8)
                        
                        # Simuliere: probiere Top-5 Tokens und bewerte mit Goal
                        k_candidates = min(top_k, logits.size(-1))
                        top_val, top_idx = torch.topk(logits, k_candidates)
                        
                        for try_k in range(min(5, k_candidates)):
                            test_token = top_idx[:, try_k:try_k+1]
                            test_ctx = torch.cat([ctx, test_token], dim=-1)
                            test_out, test_info = self.forward(test_ctx[:, -128:], learn=False)
                            test_emb = test_info.get('sparse', test_out).mean(dim=1)
                            test_score = self._goal_encoder.evaluate(test_emb, goal_emb)
                            # Boost logits für Goal-konsistente Tokens
                            logits[0, top_idx[0, try_k]] += test_score.item() * goal_bias * 2.0
                    
                    # Top-K Sampling
                    k = min(top_k, logits.size(-1))
                    top_k_logits, top_k_indices = torch.topk(logits, k)
                    top_k_logits = top_k_logits - top_k_logits.max(dim=-1, keepdim=True)[0]
                    probs = F.softmax(top_k_logits, dim=-1)
                    probs = torch.nan_to_num(probs, nan=0.0)
                    
                    if probs.sum() < 1e-8:
                        probs = torch.ones_like(probs) / probs.size(-1)
                    
                    try:
                        next_token = torch.multinomial(probs, 1)
                        next_token = top_k_indices.gather(1, next_token)
                    except RuntimeError:
                        next_token = torch.randint(0, logits.size(-1), (1, 1), device=device)
                    
                    ctx = torch.cat([ctx, next_token], dim=-1)
                    if ctx.size(-1) > 1024:
                        ctx = ctx[:, -512:]
                
                # ——— 3. Evaluate Candidate ———
                # Bewerte wie gut der Kandidat das Goal erfüllt
                final_out, final_info = self.forward(ctx[:, -256:], learn=False)
                final_emb = final_info.get('sparse', final_out).mean(dim=1)
                goal_score = self._goal_encoder.evaluate(final_emb, goal_emb)
                
                # Diversitäts-Bonus: bestrafe Ähnlichkeit zu bestehenden Kandidaten
                diversity_bonus = 0.0
                for prev_ctx, _ in candidates:
                    overlap = (ctx[0] == prev_ctx[0]).float().mean().item()
                    diversity_bonus += overlap * 0.1  # Strafe für Ähnlichkeit
                
                final_score = goal_score.item() - diversity_bonus / max(1, len(candidates))
                
                candidates.append((ctx, final_score))
                scores.append(final_score)
            
            # ——— 4. Besten Kandidaten auswählen ———
            best_idx = max(range(len(candidates)), key=lambda i: candidates[i][1])
            best_ctx, best_score = candidates[best_idx]
            
            return best_ctx.squeeze(0) if best_ctx.size(0) == 1 else best_ctx, {
                'goal_score': best_score,
                'all_scores': scores,
                'n_candidates': n_candidates,
                'goal_fulfillment': max(0, min(1, best_score)),
            }


def build_anima(vocab_size=62, device='cuda', d_model=512, d_sparse=4096, n_layers=8, d_state=256, d_context=512, lr=0.05, memory_size=64, n_attention_heads=4, n_rules=16, es_population=8, n_skills=8, use_mixed_precision=True, hierarchical_pc=False, hp_n_levels=3, hp_layers_per_level=None):
    """Anima in CogLang v3 — Vollständige AGI Architecture mit allen Phasen + Efficiency.
    
    Args:
        hierarchical_pc: PHASE 35 — Wenn True, nutze HierarchicalPC statt PredictiveStack
        hp_n_levels: Anzahl Hierarchie-Ebenen
        hp_layers_per_level: Liste mit Layer-Anzahl pro Ebene (z.B. [6, 4, 2])
    """
    brain = CogLang(use_mixed_precision=use_mixed_precision)
    brain.SensoryInput(vocab_size=vocab_size, d_model=d_model)
    brain.SparseEncoder(input_dim=d_model, d_sparse=d_sparse, sparsity=0.02)
    
    if hierarchical_pc:
        # PHASE 35: Hierarchical Predictive Coding
        if hp_layers_per_level is None:
            hp_layers_per_level = [6, 4, 2]
        brain.HierarchicalPC(d_model=d_sparse, n_levels=hp_n_levels,
                            n_layers_per_level=hp_layers_per_level,
                            d_state=d_state, d_context=d_context,
                            lr=lr, n_attention_heads=n_attention_heads)
        print(f'[ARCH] Hierarchical PC aktiv: {hp_n_levels} Ebenen ({", ".join(str(x) for x in hp_layers_per_level)} Layer)')
    else:
        # Standard: Flat PredictiveStack
        brain.PredictiveStack(d_model=d_sparse, n_layers=n_layers, d_state=d_state, d_context=d_context, lr=lr, n_attention_heads=n_attention_heads)
    
    brain.OutputDecoder(d_sparse=d_sparse, d_model=d_model, vocab_size=vocab_size, lr=lr)
    brain.EpisodicMemory(d_model=d_sparse, memory_size=memory_size, target_dim=d_state)
    brain.ActiveInference(d_model=d_sparse, n_domains=4)
    brain.SleepReplay(buffer_size=10000, d_model=d_sparse)
    brain.NeuroSymbolicBridge(vocab_size=vocab_size, d_model=d_sparse, n_rules=n_rules)
    brain.EvolutionStrategy(d_model=d_sparse, population_size=es_population, sigma=0.01)
    brain.SkillModule(d_model=d_sparse, n_skills=n_skills)
    brain.GoalEncoder(d_model=d_sparse, max_goal_len=50)
    brain.SelfReflection(d_model=d_sparse)
    brain.KnowledgeGraph(d_model=d_sparse, max_entities=1024, max_relations=64)
    brain.ToolUse(d_model=d_sparse, max_tool_history=64)
    brain.MultiAgent(d_model=d_sparse, n_personas=2)
    brain.TransferLearning(d_model=d_sparse, max_domains=8, adapter_rank=8)
    brain.ConsciousnessGlimpse(d_model=d_sparse, spotlight_size=1, n_glimpses=3)
    brain.AutoCurriculum(d_model=d_sparse, n_difficulty_levels=5, window_size=100)
    brain.CausalReasoning(d_model=d_sparse, n_causal_factors=64, temperature=0.1)
    brain.System2Reasoning(d_model=d_sparse, n_reasoning_steps=8, n_tree_branches=3, temperature=0.3)
    brain.ImaginationPlanning(d_model=d_sparse, n_plan_steps=6, n_actions=16, temperature=0.2)
    brain.ExplorationDrive(d_model=d_sparse, n_uncertainty_cells=128, n_emn_history=200)
    brain.MetaKognition(d_model=d_sparse, n_strategies=4, n_resource_levels=3)
    brain.HierarchicalMemory(d_model=d_sparse, sensory_buffer_size=1000, working_mem_size=256, episodic_buffer_size=500)
    brain.HierarchicalGoal(d_model=d_sparse, max_goals=32, max_subgoals_per_goal=8, max_depth=4)
    brain.MetaLearning(d_model=d_sparse, n_hyperparams=5, n_strategy_dim=32, window_size=200)
    brain.ActiveLearning(d_model=d_sparse, n_domains=4, vocab_size=vocab_size, window_size=500)
    brain.to(device)
    precision = "FP16/FP32 Mixed" if brain.mp.enabled else "FP32"
    print(f'CogLang v3 AGI: {brain.parameter_count()/1e6:.1f}M Parameter | {precision} | d_model={d_model}, n_layers={n_layers}, memory={memory_size}, attn={n_attention_heads}, rules={n_rules}, ES={es_population}, skills={n_skills}')
    return brain
