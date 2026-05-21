"""
CogLang — Declarative Cognitive Architecture Language.
Kein Backprop. Kein Autograd. Pure Hebbian Intelligence.
v2: W_error learning mit NLMS, W_gate frozen, weight clamping.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CogModule(nn.Module):
    """Basis für alle CogLang-Module — ist nn.Module."""
    def __init__(self, name=None):
        super().__init__()
        self.name = name or self.__class__.__name__
        self._momentum = {}
        self._lr = 0.05
        self._momentum_factor = 0.9
        self._max_weight = 3.0

    def learn(self, lr=0.05, momentum=0.9):
        self._lr = lr
        self._momentum_factor = momentum
        return self

    def _hebbian(self, error, inp, weight, lr_eff=1.0):
        """NLMS Hebbian update: dW = sum_j (error_j / ||inp_j||^2) * inp_j^T
        weight.data += lr_eff * momentum(dW)
        weight.data.clamp_(-max_weight, max_weight)
        """
        e_2d = error.reshape(-1, error.size(-1))
        i_2d = inp.reshape(-1, inp.size(-1))
        inp_pow = (i_2d ** 2).sum(dim=1, keepdim=True) + 1e-8
        dW = (e_2d / inp_pow).T @ i_2d
        if weight not in self._momentum:
            self._momentum[weight] = dW.clone()
        else:
            m = self._momentum_factor
            self._momentum[weight] = m * self._momentum[weight] + (1 - m) * dW
        weight.data.add_(lr_eff * self._momentum[weight])
        weight.data.clamp_(-self._max_weight, self._max_weight)


class SensoryInput(CogModule):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self._max_weight = 2.0
    def forward(self, ids):
        return self.embed(ids)
    
    def learn_step(self, input_ids, error_in_d_model):
        """Hebbian update for embeddings based on projected decoder error."""
        # error_in_d_model shape: [batch, seq, d_model]
        # We update the embeddings for the actual target tokens
        batch, seq, d = error_in_d_model.shape
        # Aggregate error per token in the batch
        # For simplicity, we use the error at the current position to update the embedding
        # We clamp to prevent explosion
        lr_eff = self._lr * 0.1 # Smaller LR for embeddings to maintain stability
        
        # Update embeddings: E[target] += lr * error
        # We use scatter_add_ to accumulate updates for each token ID
        updates = error_in_d_model.reshape(-1, d) * lr_eff
        ids_flat = input_ids.reshape(-1)
        
        # Clamp updates to prevent instability
        updates = torch.clamp(updates, -0.1, 0.1)
        
        self.embed.weight.data.scatter_add_(0, ids_flat.unsqueeze(1).expand(-1, d), updates)
        self.embed.weight.data.clamp_(-self._max_weight, self._max_weight)


class SparseEncoder(CogModule):
    def __init__(self, input_dim, d_sparse, sparsity=0.02):
        super().__init__()
        self.d_sparse = d_sparse
        self.sparsity = sparsity
        self.proj = nn.Linear(input_dim, d_sparse, bias=False)
        self.norm = nn.LayerNorm(d_sparse)
    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        k = max(1, int(self.d_sparse * self.sparsity))
        vals, idx = torch.topk(x, k, dim=-1)
        mask = torch.zeros_like(x)
        mask.scatter_(-1, idx, torch.sigmoid(vals))
        return mask


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

    def forward(self, x, context=None, learn=True):
        with torch.no_grad():
            batch, seq, d = x.shape
            device = x.device
            ctx = context if context is not None else torch.zeros(batch, seq, self.d_context, device=device)
            state = self.state.expand(batch, seq, -1).contiguous()
            inp = torch.cat([state, ctx], dim=-1)
            prediction = self.W_pred(inp)
            error = x - prediction

            if learn:
                lr_eff = self._lr / (batch * seq)
                # W_pred: NLMS Hebbian (stabil)
                self._hebbian(error, inp, self.W_pred.weight, lr_eff)
                # W_error: Adaptive LMS Hebbian (stabilisiert durch Error-Norm-Scaling)
                delta = self.W_error(error)
                e_flat = error.reshape(-1, d)
                d_flat = delta.reshape(-1, self.d_state)
                
                # Adaptive scaling based on error norm
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
                
                # Hebbian update for W_gate: dW = gate_error * gate_input^T
                # gate_error is approximated by how much the state changed vs delta
                gate_error = (new_state - state) # How much the gate actually let through
                g_flat = gate_in.reshape(-1, gate_in.size(-1))
                ge_flat = gate_error.reshape(-1, self.d_state)
                dw_gate = (ge_flat.T @ g_flat) / (g_flat.size(0))
                
                if self.W_gate.weight not in self._momentum:
                    self._momentum[self.W_gate.weight] = dw_gate.clone()
                else:
                    m = self._momentum_factor
                    self._momentum[self.W_gate.weight] = m * self._momentum[self.W_gate.weight] + (1 - m) * dw_gate
                
                # Very small LR for gate to maintain stability
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
    def __init__(self, d_model, n_layers=4, d_state=64, d_context=128):
        super().__init__()
        self.layers = nn.ModuleList([PredictiveLayer(d_model, d_state, d_context) for _ in range(n_layers)])
        self.pred_mixer = nn.Parameter(torch.ones(n_layers) / n_layers)

    def forward(self, x, context=None, learn=True):
        errors, states, preds = [], [], []
        current = x
        for layer in self.layers:
            s, e, p = layer(current, context, learn=learn)
            errors.append(e); states.append(s); preds.append(p)
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
        return d_hidden # Return error in d_model space for embedding update


class CogLang:
    def __init__(self):
        self.modules = nn.ModuleList()
        self._sensory = None
        self._encoder = None
        self._stack = None
        self._decoder = None
        self._context_embed = None

    def SensoryInput(self, vocab_size, d_model):
        m = SensoryInput(vocab_size, d_model); self.modules.append(m); self._sensory = m; return m
    def SparseEncoder(self, input_dim, d_sparse, sparsity=0.02):
        m = SparseEncoder(input_dim, d_sparse, sparsity); self.modules.append(m); self._encoder = m; return m
    def PredictiveStack(self, d_model, n_layers, d_state, d_context, lr=0.05):
        m = PredictiveStack(d_model, n_layers, d_state, d_context); self.modules.append(m); self._stack = m
        for layer in m.layers: layer._lr = lr
        return m
    def OutputDecoder(self, d_sparse, d_model, vocab_size, lr=0.05):
        m = OutputDecoder(d_sparse, d_model, vocab_size); self.modules.append(m); self._decoder = m
        m._lr = lr; return m

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
            ce = nn.Embedding(8192, 512).to(device)
            ce.weight.requires_grad_(False)
            self._context_embed = ce
        positions = torch.arange(seq, device=device).unsqueeze(0).expand(batch, -1)
        context = self._context_embed(positions)
        errors, states, predictions = self._stack(sparse_x, context, learn=learn)
        pred = self._stack.mixed_prediction(predictions)
        output, hidden = self._decoder(pred)
        return output, {'errors': errors, 'predictions': predictions, 'hidden': hidden, 'pred': pred, 'sparse': sparse_x}

    def learn(self, input_ids):
        with torch.no_grad():
            output, info = self.forward(input_ids, learn=True)
            d_hidden_error = self._decoder.learn_step(output, info['hidden'], info['pred'], input_ids)
            # Update embeddings with projected error
            self._sensory.learn_step(input_ids, d_hidden_error)
            loss = F.cross_entropy(output.view(-1, output.size(-1)), input_ids.view(-1))
        return loss.item(), info

    def parameter_count(self):
        return sum(p.numel() for p in self.modules.parameters())


def build_anima(vocab_size=62, device='cuda', d_model=512, d_sparse=4096, n_layers=8, d_state=256, d_context=512, lr=0.05):
    """Anima in CogLang — Parameterisierbare Architektur."""
    brain = CogLang()
    brain.SensoryInput(vocab_size=vocab_size, d_model=d_model)
    brain.SparseEncoder(input_dim=d_model, d_sparse=d_sparse, sparsity=0.02)
    brain.PredictiveStack(d_model=d_sparse, n_layers=n_layers, d_state=d_state, d_context=d_context, lr=lr)
    brain.OutputDecoder(d_sparse=d_sparse, d_model=d_model, vocab_size=vocab_size, lr=lr)
    brain.to(device)
    print(f'CogLang: {brain.parameter_count()/1e6:.1f}M Parameter | d_model={d_model}, n_layers={n_layers}')
    return brain
