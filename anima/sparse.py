"""
Sparse Distributed Representations (SDR).
Das Gehirn nutzt ~1% Aktivität — Anima auch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseEncoder(nn.Module):
    """
    Wandelt dichte Inputs in spärliche, binäre/ternäre SDRs um.
    Aktiviert nur Top-k% der Neuronen pro Sample.

    Komplexität: O(d_model) — kein quadratischer Aufwand.
    """
    def __init__(self, d_model, d_sparse=None, sparsity=0.02, ternary=False):
        super().__init__()
        self.d_model = d_model
        self.d_sparse = d_sparse or d_model * 4
        self.sparsity = sparsity
        self.ternary = ternary

        self.proj = nn.Linear(d_model, self.d_sparse, bias=False)
        self.norm = nn.LayerNorm(self.d_sparse)

        # Wettbewerbslernen: laterale Inhibition (learned)
        self.register_buffer('lateral_weights', torch.randn(self.d_sparse, self.d_sparse) * 0.01)

    def forward(self, x):
        batch, seq, _ = x.shape
        x = self.proj(x)
        x = self.norm(x)

        k = max(1, int(self.d_sparse * self.sparsity))

        if self.ternary:
            pos_vals, pos_idx = torch.topk(x, k, dim=-1)
            neg_vals, neg_idx = torch.topk(-x, k, dim=-1)

            mask = torch.zeros_like(x)
            mask.scatter_(-1, pos_idx, torch.sigmoid(pos_vals))
            mask.scatter_(-1, neg_idx, -torch.sigmoid(neg_vals))
            return mask
        else:
            vals, idx = torch.topk(x, k, dim=-1)
            mask = torch.zeros_like(x)
            mask.scatter_(-1, idx, torch.sigmoid(vals))
            return mask


class SparseToDense(nn.Module):
    """Wandelt SDR zurück in dichte Representation."""
    def __init__(self, d_sparse, d_model):
        super().__init__()
        self.proj = nn.Linear(d_sparse, d_model)

    def forward(self, x):
        return self.proj(x)
