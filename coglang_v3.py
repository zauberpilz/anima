"""
CogLang v3 — AGI Architecture
Predictive Coding + Hebbian Learning + Working Memory + Meta-Plasticity
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


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
        """NLMS Hebbian update mit Meta-Plastizität + EWC."""
        e_2d = error.reshape(-1, error.size(-1))
        i_2d = inp.reshape(-1, inp.size(-1))
        inp_pow = (i_2d ** 2).sum(dim=1, keepdim=True) + 1e-8
        dW = (e_2d / inp_pow).T @ i_2d
        lr_eff *= self._meta_lr_scale
        
        if weight not in self._momentum:
            self._momentum[weight] = dW.clone()
        else:
            m = self._momentum_factor
            self._momentum[weight] = m * self._momentum[weight] + (1 - m) * dW
            
        # PHASE 4: Apply EWC penalty before update
        self._ewc_consolidate(weight)
        
        weight.data.add_(lr_eff * self._momentum[weight])
        weight.data.clamp_(-self._max_weight, self._max_weight)
        
        # PHASE 4: Update Fisher with current gradient magnitude
        self._ewc_update_fisher(weight, self._momentum[weight])


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
    def __init__(self, d_model, d_state=64, d_context=128):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_context = d_context
        self.W_pred = nn.Linear(d_state + d_context, d_model, bias=False)
        self.W_error = nn.Linear(d_model, d_state, bias=False)
        self.W_gate = nn.Linear(d_state + d_model + d_context, d_state)
        self.register_buffer('state', torch.zeros(1, 1, d_state))
        self.register_buffer('error_trace', torch.zeros(1, d_model))

    def forward(self, x, context=None, memory_retrieved=None, learn=True):
        with torch.no_grad():
            batch, seq, d = x.shape
            device = x.device
            ctx = context if context is not None else torch.zeros(batch, seq, self.d_context, device=device)
            
            # Combine state with retrieved memory if available
            state = self.state.expand(batch, seq, -1).contiguous()
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
                new_state = (1 - gate) * state + gate * delta
                
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
                new_state = (1 - gate) * state + gate * delta

            self.state = new_state[0:1, -1:, :].detach()
            self.error_trace = error[0:1, -1, :].detach()
            return new_state, error, prediction

    def reset_state(self, batch_size=1, device='cpu'):
        self.state = torch.zeros(1, 1, self.d_state, device=device)
        self.error_trace = torch.zeros(1, self.d_model, device=device)


class PredictiveStack(CogModule):
    def __init__(self, d_model, n_layers=4, d_state=64, d_context=128, n_attention_heads=4):
        super().__init__()
        self.layers = nn.ModuleList([PredictiveLayer(d_model, d_state, d_context) for _ in range(n_layers)])
        self.pred_mixer = nn.Parameter(torch.ones(n_layers) / n_layers)
        # PHASE 3: Predictive Attention
        self.attention = PredictiveAttention(d_model, n_heads=n_attention_heads)

    def forward(self, x, context=None, memory_retrieved=None, errors_for_attn=None, learn=True):
        errors, states, preds = [], [], []
        current = x
        for i, layer in enumerate(self.layers):
            mem = memory_retrieved if i == 0 else None
            s, e, p = layer(current, context, memory_retrieved=mem, learn=learn)
            errors.append(e); states.append(s); preds.append(p)
            
            # PHASE 3: Apply attention after each layer
            if errors_for_attn is not None:
                current = self.attention(torch.tanh(e), error=errors_for_attn, learn=learn)
            else:
                current = torch.tanh(e)
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


class CogLang:
    def __init__(self):
        self.modules = nn.ModuleList()
        self._sensory = None
        self._encoder = None
        self._stack = None
        self._decoder = None
        self._context_embed = None
        self._memory = None

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
        pred = self._stack.mixed_prediction(predictions)
        output, hidden = self._decoder(pred)
        return output, {'errors': errors, 'predictions': predictions, 'hidden': hidden, 'pred': pred, 'sparse': sparse_x}

    def learn(self, input_ids):
        with torch.no_grad():
            output, info = self.forward(input_ids, learn=True)
            d_hidden_error = self._decoder.learn_step(output, info['hidden'], info['pred'], input_ids)
            self._sensory.learn_step(input_ids, d_hidden_error)
            
            # Write to episodic memory
            if self._memory is not None:
                self._memory._write_to_memory(info['sparse'][:, -1, :])
            
            loss = F.cross_entropy(output.view(-1, output.size(-1)), input_ids.view(-1))
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


def build_anima(vocab_size=62, device='cuda', d_model=512, d_sparse=4096, n_layers=8, d_state=256, d_context=512, lr=0.05, memory_size=64, n_attention_heads=4):
    """Anima in CogLang v3 — mit Working Memory + Predictive Attention."""
    brain = CogLang()
    brain.SensoryInput(vocab_size=vocab_size, d_model=d_model)
    brain.SparseEncoder(input_dim=d_model, d_sparse=d_sparse, sparsity=0.02)
    brain.PredictiveStack(d_model=d_sparse, n_layers=n_layers, d_state=d_state, d_context=d_context, lr=lr, n_attention_heads=n_attention_heads)
    brain.OutputDecoder(d_sparse=d_sparse, d_model=d_model, vocab_size=vocab_size, lr=lr)
    brain.EpisodicMemory(d_model=d_sparse, memory_size=memory_size, target_dim=d_state)
    brain.to(device)
    print(f'CogLang v3: {brain.parameter_count()/1e6:.1f}M Parameter | d_model={d_model}, n_layers={n_layers}, memory={memory_size}, attn_heads={n_attention_heads}')
    return brain
