"""
Anima Network — skalierte Version (50M Parameter).
Kein Backprop. NLMS + Momentum + Embedding-Training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from collections import deque

from .sparse import SparseEncoder
from .predictive import PredictiveStack
from .memory import SparseAssociativeMemory
from .meta import MetaController, SelfMonitor
from .safety import SafetyCore


class Anima(nn.Module):
    def __init__(self, vocab_size=256, d_model=512, d_sparse=4096,
                 d_state=256, d_context=512, n_layers=8, sparsity=0.02,
                 mem_capacity=20000, max_seq_len=8192):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.embed = nn.Embedding(vocab_size, d_model)
        self.sparse_enc = SparseEncoder(d_model, d_sparse, sparsity=sparsity)
        self.predictive_stack = PredictiveStack(d_sparse, n_layers, d_state, d_context)
        self.memory = SparseAssociativeMemory(d_sparse, capacity=mem_capacity)

        # Output-Decoder mit Mischung aller Prediction-Layer
        self.out_proj = nn.Linear(d_sparse, d_model, bias=False)
        self.out_head = nn.Linear(d_model, vocab_size, bias=False)
        self.pred_mixer = nn.Parameter(torch.ones(n_layers) / n_layers)

        self.self_monitor = SelfMonitor(d_model)
        self.meta_controller = MetaController(d_model, n_layers=n_layers)
        self.meta_controller.set_monitor(self.self_monitor)
        self.context_embed = nn.Embedding(max_seq_len, d_context)

        self.safety = SafetyCore()
        self.experience_buffer = deque(maxlen=1000)
        self.generation_count = 0
        self.sparsity = sparsity
        self.online_lr = 0.05

    def encode(self, input_ids):
        x = self.embed(input_ids)
        return self.sparse_enc(x)

    def forward(self, input_ids, learn=True, slow=False):
        batch, seq = input_ids.shape
        if seq > self.max_seq_len:
            input_ids = input_ids[:, -self.max_seq_len:]
            seq = self.max_seq_len

        x = self.embed(input_ids)
        sparse_x = self.sparse_enc(x)
        positions = torch.arange(seq, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        context = self.context_embed(positions)

        t0 = time.time()
        with torch.no_grad():
            errors, states, predictions = self.predictive_stack(sparse_x, context, learn=learn)
        fast_time = time.time() - t0

        # Gelernte Mischung aller Layer-Vorhersagen
        mix = torch.softmax(self.pred_mixer, dim=-1)
        pred = sum(w * p for w, p in zip(mix, predictions))

        hidden = torch.tanh(self.out_proj(pred))
        output = self.out_head(hidden)

        # Output-Decoder Lernen
        if learn and self.training:
            with torch.no_grad():
                logits = output
                probs = F.softmax(logits, dim=-1)
                V = logits.size(-1)
                smooth = 0.1
                target_probs = (1 - smooth) * F.one_hot(input_ids, num_classes=V).float() + smooth / V
                error_out = probs - target_probs

                lr_out = self.online_lr

                d_hidden = error_out @ self.out_head.weight
                dw_proj = (d_hidden.reshape(-1, hidden.size(-1)).T @
                          pred.reshape(-1, pred.size(-1))) / (batch * seq)

                dw_head = (error_out.reshape(-1, logits.size(-1)).T @
                          hidden.reshape(-1, hidden.size(-1))) / (batch * seq)
                self.out_head.weight.data -= lr_out * dw_head
                self.out_proj.weight.data -= lr_out * dw_proj

                self.out_head.weight.data.clamp_(-2.0, 2.0)
                self.out_proj.weight.data.clamp_(-2.0, 2.0)

                # Embedding-Training: projiziere Output-Fehler zurück ins Embedding
                # error_out: [B, S, V] → d_hidden: [B, S, d_model]
                # Aktualisiere Embedding jedes Tokens basierend auf gemitteltem Fehlersignal
                d_hidden = error_out @ self.out_head.weight
                unique_tokens = torch.unique(input_ids)
                for v in unique_tokens:
                    mask = (input_ids == v)
                    if mask.sum() > 0:
                        avg = d_hidden[mask].mean(dim=0)
                        self.embed.weight.data[v] += self.online_lr * 0.01 * avg

        self.generation_count += 1

        layer_errs = [e.abs().mean().item() for e in errors]
        density = (sparse_x != 0).float().mean().item()
        mem_usage = self.memory.size() / self.memory.capacity
        self.self_monitor.record_step(layer_errs, mem_usage, density, fast_time,
            -F.cross_entropy(output.view(-1, output.size(-1)), input_ids.view(-1)).item())

        return output, {
            'errors': errors, 'states': states, 'predictions': predictions,
            'sparse': sparse_x, 'density': density, 'mem_usage': mem_usage,
            'fast_time': fast_time,
        }

    @torch.no_grad()
    def generate(self, input_ids, max_new=200, temp=0.8):
        self.eval()
        for step in range(max_new):
            out, info = self(input_ids[:, -self.max_seq_len:], learn=False)
            probs = F.softmax(out[:, -1, :] / temp, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, next_tok], dim=-1)
            if step % 5 == 0:
                self.memory.write(info['sparse'][:, -1:], info['predictions'][0][:, -1:])
        return input_ids

    def learn_from_interaction(self, input_ids):
        self.train()
        input_ids = input_ids.to(next(self.parameters()).device)
        out, info = self(input_ids, learn=True)
        loss = F.cross_entropy(out.view(-1, out.size(-1)), input_ids.view(-1))
        self.experience_buffer.append((input_ids.cpu(), loss.item()))
        return loss.item()

    def self_reflect(self, loss=None):
        if loss is not None and (loss > 50 or loss != loss):
            self.online_lr = max(0.001, self.online_lr * 0.1)
            self.sparsity = 0.02
            return {'suggestions': [f'CRASH detected (loss={loss})'], 'changes': ['LR×0.1, sparsity reset']}
        suggestions = self.self_monitor.get_optimization_suggestions()
        changes = []
        for s in suggestions:
            if 'Fehler' in s and 'Layer' in s:
                if 'inf' in s.lower():
                    self.online_lr = max(0.001, self.online_lr * 0.5)
                    changes.append(f'LR reduced: {s}')
                else:
                    self.online_lr = min(0.1, self.online_lr * 1.05)
                    changes.append(f'LR fine-tuned: {s}')
            elif 'Dichte' in s:
                self.sparsity = min(0.08, self.sparsity * 1.2)
                changes.append(f'Sparsity raised: {s}')
            elif 'Speicher' in s:
                self.memory.consolidate()
                changes.append(f'Consolidated: {s}')
        return {'suggestions': suggestions, 'changes': changes}

    def get_status(self):
        s = self.meta_controller.get_status()
        s.update({'sparsity': self.sparsity, 'online_lr': self.online_lr,
                  'memory_items': self.memory.size(), 'total_generations': self.generation_count,
                  'experience_buffer': len(self.experience_buffer)})
        return s

    def add_user_rule(self, rule):
        self.safety.add_user_rule(rule)

    def reset_states(self, batch_size=1, device='cpu'):
        self.predictive_stack.reset_all_states(batch_size, device)

    def get_efficiency_report(self):
        return {
            'Architecture': 'Anima (Predictive Coding + Hebbian + Momentum)',
            'Parameters': f'{sum(p.numel() for p in self.parameters())/1e6:.2f}M',
            'vs Transformer': 'O(seq_len²) for Attention',
            'Activation Sparsity': f'{self.sparsity:.1%}',
            'Training': 'Backprop-FREI (nur Hebbian + Delta-Regeln)',
            'Safety Core': 'IMMUTABLE',
        }
