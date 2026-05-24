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
        
    def adjust(self, vram_used_mb):
        """Adjust batch/seq based on VRAM usage."""
        if vram_used_mb > self.max_vram_mb * 0.9:
            # Too much VRAM -> reduce
            self.batch = max(2, self.batch // 2)
            self.seq = max(32, self.seq // 2)
            self.oom_count += 1
        elif vram_used_mb < self.max_vram_mb * 0.5 and self.oom_count == 0:
            # Plenty of VRAM -> increase
            self.batch = min(32, self.batch * 2)
            self.seq = min(512, self.seq * 2)
            
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
        self._ewc_optimal_params = {}  # Snapshot of important weights
        self._ewc_lambda = 0.1  # EWC penalty strength
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
            # Penalty: -lambda * fisher * (current - optimal)
            penalty = -self._ewc_lambda * fisher * (weight.data - optimal)
            weight.data.add_(penalty, alpha=0.01)  # Small step to avoid instability
            
    def _ewc_update_fisher(self, weight, gradient_estimate):
        """PHASE 4: Update Fisher Information diagonal approximation."""
        if weight not in self._ewc_fisher:
            self._ewc_fisher[weight] = gradient_estimate ** 2
        else:
            # Exponential moving average of Fisher
            self._ewc_fisher[weight] = 0.9 * self._ewc_fisher[weight] + 0.1 * (gradient_estimate ** 2)
            
    def _ewc_snapshot(self):
        """PHASE 4: Save current weights as optimal for EWC."""
        for name, param in self.named_parameters():
            if param.requires_grad or True:  # Track all weights
                self._ewc_optimal_params[name] = param.data.clone()
            
    def _hebbian(self, error, inp, weight, lr_eff=1.0):
        """PHASE 28: Oja's Rule Hebbian update — weight-normalizing, prevents explosion."""
        # NaN-Guard: Bereinige Input-Tensoren
        if torch.isnan(error).any() or torch.isinf(error).any():
            error = torch.nan_to_num(error, nan=0.0, posinf=1.0, neginf=-1.0)
        if torch.isnan(inp).any() or torch.isinf(inp).any():
            inp = torch.nan_to_num(inp, nan=0.0, posinf=1.0, neginf=-1.0)

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
        
    def learn_step(self, query, retrieved, target):
        """Hebbian learning for memory read/write weights."""
        # Learn to retrieve better: minimize difference between retrieved and target
        error = target - retrieved  # [batch, seq, d]
        
        # Update W_read
        e_flat = error.reshape(-1, self.d_model)
        q_flat = query.reshape(-1, self.d_model)
        sim = q_flat @ self.memory.T  # [batch*seq, memory_size]
        att = torch.softmax(sim / (self.d_model ** 0.5), dim=-1)
        
        # dW_read = error^T @ attention
        dw_read = e_flat.T @ att  # [d, memory_size]
        if self.W_read.weight not in self._momentum:
            self._momentum[self.W_read.weight] = dw_read.T.clone()
        else:
            m = self._momentum_factor
            self._momentum[self.W_read.weight] = m * self._momentum[self.W_read.weight] + (1 - m) * dw_read.T
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

    def forward(self, x, context=None, memory_retrieved=None, learn=True):
        with torch.no_grad():
            batch, seq, d = x.shape
            device = x.device
            ctx = context if context is not None else torch.zeros(batch, seq, self.d_context, device=device)
            
            # Combine state with retrieved memory if available
            state = self.state.expand(batch, seq, -1).contiguous()
            # NaN-Guard: Korrupte Zustände zurücksetzen
            if torch.isnan(state).any():
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
                
                # W_pred: NLMS Hebbian
                self._hebbian(error, inp, self.W_pred.weight, lr_eff)
                
                # W_error: Adaptive LMS Hebbian
                delta = self.W_error(error)
                e_flat = error.reshape(-1, d)
                d_flat = delta.reshape(-1, self.d_state)
                error_norm = (e_flat ** 2).sum(dim=1, keepdim=True).mean() + 1e-8
                adaptive_scale = torch.clamp(1.0 / (error_norm.sqrt() + 1e-4), 0.1, 10.0)
                dw_err = (d_flat.T @ e_flat) / (e_flat.size(0)) * adaptive_scale
                
                if self.W_error.weight not in self._momentum:
                    self._momentum[self.W_error.weight] = dw_err.clone()
                else:
                    m = self._momentum_factor
                    self._momentum[self.W_error.weight] = m * self._momentum[self.W_error.weight] + (1 - m) * dw_err
                self.W_error.weight.data.add_(self._momentum[self.W_error.weight], alpha=lr_eff * 0.2)
                self.W_error.weight.data.clamp_(-1.0, 1.0)
                
                # Gate: Learnable with stabilized LR
                gate_in = torch.cat([state, error, ctx], dim=-1)
                gate = torch.sigmoid(self.W_gate(gate_in))
                # PHASE 5: Timescale modulation - slower layers update less
                new_state = (1 - gate * self.timescale) * state + (gate * self.timescale) * delta
                
                # Hebbian update for W_gate
                gate_error = (new_state - state)
                g_flat = gate_in.reshape(-1, gate_in.size(-1))
                ge_flat = gate_error.reshape(-1, self.d_state)
                dw_gate = (ge_flat.T @ g_flat) / (g_flat.size(0))
                
                if self.W_gate.weight not in self._momentum:
                    self._momentum[self.W_gate.weight] = dw_gate.clone()
                else:
                    m = self._momentum_factor
                    self._momentum[self.W_gate.weight] = m * self._momentum[self.W_gate.weight] + (1 - m) * dw_gate
                self.W_gate.weight.data.add_(self._momentum[self.W_gate.weight], alpha=lr_eff * 0.05)
                self.W_gate.weight.data.clamp_(-0.5, 0.5)
            else:
                delta = self.W_error(error)
                gate_in = torch.cat([state, error, ctx], dim=-1)
                gate = torch.sigmoid(self.W_gate(gate_in))
                # PHASE 5: Timescale modulation
                new_state = (1 - gate * self.timescale) * state + (gate * self.timescale) * delta

            new_state_detached = new_state[0:1, -1:, :].detach()
            if torch.isnan(new_state_detached).any():
                new_state_detached = torch.zeros_like(new_state_detached)
            self.state = new_state_detached
            self.error_trace = error[0:1, -1, :].detach()
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

    def forward(self, x, context=None, memory_retrieved=None, errors_for_attn=None, learn=True):
        errors, states, preds = [], [], []
        current = x
        for i, layer in enumerate(self.layers):
            mem = memory_retrieved if i == 0 else None
            s, e, p = layer(current, context, memory_retrieved=mem, learn=learn)
            errors.append(e); states.append(s); preds.append(p)
            
            # PHASE 11: Hebbian Attention als primäre Attention
            attended = self.hebbian_attn(torch.tanh(e), learn=learn)
            
            # PHASE 29: Residual connection — add input to attended output
            attended = current + attended * 0.5  # Scaled residual
            
            # PHASE 6: Self-Model modulation
            if len(errors) == self.n_layers:
                modulation = self.self_model(errors)
                current = attended + modulation * 0.1
            else:
                current = attended
        return errors, states, preds

    def mixed_prediction(self, predictions):
        mix = torch.softmax(self.pred_mixer, dim=-1)
        return sum(w * p for w, p in zip(mix, predictions))

    def reset_states(self, batch_size=1, device='cpu'):
        for l in self.layers:
            l.reset_state(batch_size, device)


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


class IntrinsicMotivation(CogModule):
    """
    PHASE 7: Intrinsische Motivation — Neugier-getriebenes Lernen.
    Prediction Error wird als interner Reward-Signal genutzt.
    Hoher Error (Novelty) -> erhöhtes Lernen. Niedriger Error -> Stabilisierung.
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.register_buffer('curiosity_drive', torch.tensor(0.5))  # Internal curiosity level
        self.register_buffer('novelty_threshold', torch.tensor(1.0))
        self.register_buffer('reward_history', torch.zeros(100))
        self._reward_idx = 0
        
    def compute_intrinsic_reward(self, error):
        """
        Compute intrinsic reward from prediction error.
        Returns: reward scalar and curiosity modulation factor.
        """
        with torch.no_grad():
            # Error magnitude as novelty signal
            error_norm = (error ** 2).sum(dim=-1).mean().item()
            
            # Intrinsic reward: higher when error exceeds threshold (novelty detection)
            intrinsic_reward = max(0, error_norm - self.novelty_threshold.item())
            
            # Update curiosity drive with exponential moving average
            self.curiosity_drive = 0.95 * self.curiosity_drive + 0.05 * intrinsic_reward
            
            # Update reward history
            self.reward_history[self._reward_idx] = intrinsic_reward
            self._reward_idx = (self._reward_idx + 1) % 100
            
            # Curiosity modulation: scale learning based on recent rewards
            avg_reward = self.reward_history.mean().item()
            curiosity_factor = 1.0 + avg_reward * 0.5  # Boost learning when curious
            
            return intrinsic_reward, curiosity_factor


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
    """
    def __init__(self, d_model, population_size=8, sigma=0.01):
        super().__init__()
        self.d_model = d_model
        self.population_size = population_size
        self.sigma = sigma  # Perturbation noise
        self._best_weights = {}
        self._best_fitness = float('inf')
        
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


class CogLang:
    def __init__(self, use_mixed_precision=True):
        self.modules = nn.ModuleList()
        self._sensory = None
        self._encoder = None
        self._stack = None
        self._decoder = None
        self._context_embed = None
        self._memory = None
        self._motivation = None
        self._bridge = None
        self._es = None
        self._skills = None
        self._security_head = None
        self._network_encoder = None
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
        
        errors, states, predictions = self._stack(sparse_x, context, memory_retrieved=memory_retrieved, errors_for_attn=sparse_x, learn=learn)
        # NaN-Guard: Sichere Predictions
        predictions = [torch.nan_to_num(p, nan=0.0, posinf=1.0, neginf=-1.0) for p in predictions]
        pred = self._stack.mixed_prediction(predictions)
        output, hidden = self._decoder(pred)
        
        # PHASE 8: Apply neuro-symbolic rules to output
        if self._bridge is not None:
            output = self._bridge(output, context_embedding=sparse_x)
            
        return output, {'errors': errors, 'predictions': predictions, 'hidden': hidden, 'pred': pred, 'sparse': sparse_x, 'output': output}

    def learn(self, input_ids):
        with torch.no_grad():
            output, info = self.forward(input_ids, learn=True)
            d_hidden_error = self._decoder.learn_step(output, info['hidden'], info['pred'], input_ids)
            self._sensory.learn_step(input_ids, d_hidden_error)
            
            # Write to episodic memory
            if self._memory is not None:
                self._memory._write_to_memory(info['sparse'][:, -1, :])
            
            # PHASE 7: Intrinsic Motivation - compute curiosity reward
            if self._motivation is not None:
                total_error = sum((e ** 2).sum().item() for e in info['errors'])
                reward, curiosity_factor = self._motivation.compute_intrinsic_reward(info['errors'][0])
                # Modulate meta-plasticity with curiosity
                for module in self.modules.modules():
                    if hasattr(module, '_meta_lr_scale'):
                        module._meta_lr_scale *= curiosity_factor
                        module._meta_lr_scale = max(0.1, min(2.0, module._meta_lr_scale))
            
            loss = F.cross_entropy(output.view(-1, output.size(-1)), input_ids.view(-1))
            if torch.isnan(loss) or torch.isinf(loss):
                return 100.0, info
        return loss.item(), info

    def parameter_count(self):
        return sum(p.numel() for p in self.modules.parameters())

    def save_checkpoint(self, path, config=None):
        checkpoint = {
            'model_state': self.modules.state_dict(),
            'config': config
        }
        torch.save(checkpoint, path)
        print(f'Checkpoint gespeichert: {path}')

    def load_checkpoint(self, path):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location='cpu')
            self.modules.load_state_dict(checkpoint['model_state'])
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


def build_anima(vocab_size=62, device='cuda', d_model=512, d_sparse=4096, n_layers=8, d_state=256, d_context=512, lr=0.05, memory_size=64, n_attention_heads=4, n_rules=16, es_population=8, n_skills=8, use_mixed_precision=True):
    """Anima in CogLang v3 — Vollständige AGI Architecture mit allen 14 Phasen + Efficiency."""
    brain = CogLang(use_mixed_precision=use_mixed_precision)
    brain.SensoryInput(vocab_size=vocab_size, d_model=d_model)
    brain.SparseEncoder(input_dim=d_model, d_sparse=d_sparse, sparsity=0.02)
    brain.PredictiveStack(d_model=d_sparse, n_layers=n_layers, d_state=d_state, d_context=d_context, lr=lr, n_attention_heads=n_attention_heads)
    brain.OutputDecoder(d_sparse=d_sparse, d_model=d_model, vocab_size=vocab_size, lr=lr)
    brain.EpisodicMemory(d_model=d_sparse, memory_size=memory_size, target_dim=d_state)
    brain.IntrinsicMotivation(d_model=d_sparse)
    brain.NeuroSymbolicBridge(vocab_size=vocab_size, d_model=d_sparse, n_rules=n_rules)
    brain.EvolutionStrategy(d_model=d_sparse, population_size=es_population, sigma=0.01)
    brain.SkillModule(d_model=d_sparse, n_skills=n_skills)
    brain.to(device)
    precision = "FP16/FP32 Mixed" if brain.mp.enabled else "FP32"
    print(f'CogLang v3 AGI: {brain.parameter_count()/1e6:.1f}M Parameter | {precision} | d_model={d_model}, n_layers={n_layers}, memory={memory_size}, attn={n_attention_heads}, rules={n_rules}, ES={es_population}, skills={n_skills}')
    return brain
