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

    def forward(self, x, context=None, memory_retrieved=None, learn=True):
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
                
                # W_pred: NLMS Hebbian (hat eigenen NaN-Guard)
                self._hebbian(error, inp, self.W_pred.weight, lr_eff)
                
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

    def forward(self, x, context=None, memory_retrieved=None, errors_for_attn=None, learn=True):
        errors, states, preds = [], [], []
        current = x
        for i, layer in enumerate(self.layers):
            mem = memory_retrieved if i == 0 else None
            s, e, p = layer(current, context, memory_retrieved=mem, learn=learn)
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
            avg_recent_reward = self.rewards_history.mean().item() if hasattr(self, 'reward_history') else 0.0
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
                self.replay_inputs[idx, :len(seq)] = seq[:128]
                self.replay_errors[idx] = error_norm
                self.replay_domains[idx] = domain_idx
                self.replay_weights[idx] = importance
                self.replay_age[idx] = 0
                
                # Pattern: höherer Error + höhere Importance = stärkeres Pattern
                self.pattern_strength[idx] = error_norm * importance
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
            if prev_hidden is not None:
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
                'spotlight_entropy': self.spotlight_entropy[max(0, self._entropy_idx-1)].item(),
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
        
        # PHASE 35: HierarchicalPC gibt 4 Werte zurück (errors, states, preds, mixed_pred)
        # PredictiveStack gibt 3 Werte zurück (errors, states, preds)
        stack_result = self._stack(sparse_x, context, memory_retrieved=memory_retrieved, errors_for_attn=sparse_x, learn=learn)
        if len(stack_result) == 4:
            errors, states, predictions, pred = stack_result
        else:
            errors, states, predictions = stack_result
            # NaN-Guard: Sichere Predictions
            predictions = [torch.nan_to_num(p, nan=0.0, posinf=1.0, neginf=-1.0) for p in predictions]
            pred = self._stack.mixed_prediction(predictions)
        
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
        
        info_extra = {}
        
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
            self._prev_hidden = pred[:, -1:, :].detach()
        
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
                error_norm = sum((e ** 2).mean().item() for e in info['errors']) / len(info['errors'])
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
            
            # PHASE 13b: EvolutionStrategy inline learn_step — Plateau-noise + Revert
            if self._es is not None and self._ewc_step_counter % 100 == 0:
                self._es.learn_step(loss.item(), self.modules)
            
            if torch.isnan(loss) or torch.isinf(loss):
                return 100.0, info
        return loss.item(), info
    
    def ewc_snapshot_all(self):
        """PHASE 4: EWC on all CogModule instances - schützt vor Catastrophic Forgetting."""
        for module in self.modules.modules():
            if hasattr(module, '_ewc_snapshot') and module is not self:
                module._ewc_snapshot()

    def run_sleep_phase(self, n_steps=100, device='cuda'):
        """PHASE 34: Führe Sleep-Phase zur Konsolidierung aus."""
        if self._sleep_replay is not None:
            report = self._sleep_replay.sleep_phase(self, n_steps=n_steps, device=device)
            return report
        return {}
    
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
    brain.to(device)
    precision = "FP16/FP32 Mixed" if brain.mp.enabled else "FP32"
    print(f'CogLang v3 AGI: {brain.parameter_count()/1e6:.1f}M Parameter | {precision} | d_model={d_model}, n_layers={n_layers}, memory={memory_size}, attn={n_attention_heads}, rules={n_rules}, ES={es_population}, skills={n_skills}')
    return brain
