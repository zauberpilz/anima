"""
Predictive Coding Layer — Hebbian Forward-Learning mit Momentum.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictiveCodingLayer(nn.Module):
    def __init__(self, d_model, d_state=64, d_context=128, lr=0.05, momentum=0.9):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_context = d_context
        self.lr = lr
        self.momentum = momentum

        self.W_pred = nn.Linear(d_state + d_context, d_model, bias=False)
        self.W_error = nn.Linear(d_model, d_state, bias=False)
        self.W_gate = nn.Linear(d_state + d_model + d_context, d_state)

        self.register_buffer('state', torch.zeros(1, 1, d_state))
        self.register_buffer('error_trace', torch.zeros(1, d_model))

        # Momentum-Puffer (initialisiert bei erstem Forward)
        self.register_buffer('m_pred', torch.zeros(d_model, d_state + d_context))
        self.register_buffer('m_error', torch.zeros(d_state, d_model))
        self.register_buffer('m_gate', torch.zeros(d_state, d_state + d_model + d_context))
        self._initialized = False

    def forward(self, x, context=None, learn=True):
        batch, seq, d = x.shape
        ctx = context if context is not None else torch.zeros(batch, seq, 64, device=x.device)

        state = self.state[0:1, -1:, :].expand(batch, seq, -1).contiguous()

        inp = torch.cat([state, ctx], dim=-1)
        prediction = self.W_pred(inp)
        error = x - prediction

        if learn and self.training:
            with torch.no_grad():
                e_2d = error.reshape(-1, d)
                i_2d = inp.reshape(-1, inp.size(-1))
                inp_pow = (i_2d ** 2).sum(dim=1, keepdim=True) + 1e-8
                lr_eff = self.lr / (batch * seq)

                # Gradienten
                dw_pred = (e_2d / inp_pow).T @ i_2d
                d_local = (e_2d / inp_pow) @ self.W_error.weight.T
                dw_err = d_local.T @ e_2d

                g_in = torch.cat([state, error, ctx], dim=-1
                ).reshape(-1, self.d_state + self.d_model + self.d_context)
                g_flat = torch.sigmoid(self.W_gate(g_in))
                target = torch.sigmoid(error.abs().mean(dim=-1, keepdim=True)).reshape(-1, 1)
                gate_err = g_flat - target.expand(-1, self.d_state)
                dw_gate = (gate_err / inp_pow).T @ g_in

                # Momentum anwenden
                if not self._initialized:
                    self.m_pred.data = dw_pred
                    self.m_error.data = dw_err
                    self.m_gate.data = dw_gate
                    self._initialized = True
                else:
                    self.m_pred.data = self.momentum * self.m_pred + (1 - self.momentum) * dw_pred
                    self.m_error.data = self.momentum * self.m_error + (1 - self.momentum) * dw_err
                    self.m_gate.data = self.momentum * self.m_gate + (1 - self.momentum) * dw_gate

                self.W_pred.weight.data += lr_eff * self.m_pred
                self.W_error.weight.data += lr_eff * self.m_error
                self.W_gate.weight.data -= lr_eff * 0.1 * self.m_gate

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


class PredictiveStack(nn.Module):
    def __init__(self, d_model, n_layers=4, d_state=64, d_context=128):
        super().__init__()
        self.layers = nn.ModuleList([
            PredictiveCodingLayer(d_model, d_state, d_context)
            for _ in range(n_layers)
        ])

    def forward(self, x, context=None, learn=True):
        errors, states, preds = [], [], []
        current = x
        for layer in self.layers:
            state, error, pred = layer(current, context, learn=learn)
            errors.append(error)
            states.append(state)
            preds.append(pred)
            current = torch.tanh(error)
        return errors, states, preds

    def reset_all_states(self, batch_size=1, device='cpu'):
        for l in self.layers:
            if hasattr(l, 'reset_state'):
                l.reset_state(batch_size, device)
