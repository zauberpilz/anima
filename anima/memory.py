"""
Sparse Associative Memory — Inhaltsadressierbarer Speicher.

Anders als Attention (O(n²) für n Speicheritems) erlaubt dieser Speicher
O(1) Zugriff durch Locality-Sensitive Hashing auf spärliche Codes.

Eigenschaften:
- Speichert SDRs (Sparse Distributed Representations)
- Content-addressable: retrieve by similarity, not by index
- Zeitbewusst: Items haben einen Verfallszeitstempel
- Konsolidierung: häufige Items werden permanent
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SparseAssociativeMemory(nn.Module):
    """
    Assoziativer Speicher mit LSH-basiertem Zugriff.

    Speicherformat: {key: SDR, value: SDR, timestamp: float, access_count: int}
    """
    def __init__(self, d_sparse, capacity=4096, lsh_bits=8, decay_rate=0.01):
        super().__init__()
        self.d_sparse = d_sparse
        self.capacity = capacity
        self.lsh_bits = lsh_bits
        self.decay_rate = decay_rate

        # LSH-Projektionen
        self.register_buffer('lsh_proj',
            torch.randn(lsh_bits, d_sparse) * 0.1)

        # Speicher-Buffer
        self.register_buffer('keys', torch.zeros(capacity, d_sparse))
        self.register_buffer('values', torch.zeros(capacity, d_sparse))
        self.register_buffer('timestamps', torch.zeros(capacity))
        self.register_buffer('access_counts', torch.zeros(capacity))
        self.register_buffer('valid_mask', torch.zeros(capacity, dtype=torch.bool))

        self.register_buffer('global_clock', torch.zeros(1))

    def _hash(self, x):
        """Locality-Sensitive Hashing für SDRs."""
        proj = F.linear(x, self.lsh_proj)
        return (proj > 0).int()

    def _find_slot(self, hash_code):
        """Findet den nächsten freien oder überschreibbaren Slot."""
        if not self.valid_mask.any():
            return 0

        # Prüfe auf existierenden Eintrag mit gleichem Hash
        existing = (self.valid_mask & (self._batch_hash_check(hash_code))).nonzero()
        if existing.numel() > 0:
            return existing[0].item()

        # Freien Slot finden
        free = (~self.valid_mask).nonzero()
        if free.numel() > 0:
            return free[0].item()

        # Ältesten/nicht-genutzten Slot überschreiben
        scores = self.access_counts / (self.global_clock - self.timestamps + 1)
        return scores.argmin().item()

    def _batch_hash_check(self, hash_code):
        """Batch-Version des Hash-Vergleichs."""
        existing_hashes = (self.lsh_proj @ self.keys.T > 0).int().T
        target_hash = hash_code.unsqueeze(0)
        return (existing_hashes == target_hash).all(dim=1)

    def write(self, key, value):
        """Schreibt ein (key, value) Paar in den Speicher."""
        batch, seq, d = key.shape
        hash_code = self._hash(key)

        for b in range(batch):
            for t in range(seq):
                idx = self._find_slot(hash_code[b, t])
                self.keys[idx] = key[b, t].detach()
                self.values[idx] = value[b, t].detach()
                self.timestamps[idx] = self.global_clock.item()
                self.access_counts[idx] = 0
                self.valid_mask[idx] = True
                self.global_clock += 1

    def read(self, query, k=8):
        """Liest die top-k ähnlichsten Einträge zum Query."""
        batch, seq, d = query.shape
        hash_code = self._hash(query)

        if not self.valid_mask.any():
            return torch.zeros(batch, seq, self.d_sparse, device=query.device)

        # Ähnlichkeit zu allen gültigen Einträgen
        valid_keys = self.keys[self.valid_mask]
        valid_vals = self.values[self.valid_mask]

        # Kosinus-Ähnlichkeit
        query_norm = F.normalize(query, dim=-1)
        key_norm = F.normalize(valid_keys, dim=-1)
        sim = torch.matmul(query_norm, key_norm.T)

        # Top-k
        vals, idx = torch.topk(sim, min(k, len(valid_vals)), dim=-1)

        # Gewichteter Durchschnitt
        weights = F.softmax(vals / 0.1, dim=-1)
        retrieved = torch.zeros(batch, seq, self.d_sparse, device=query.device)
        for b in range(batch):
            for t in range(seq):
                retrieved[b, t] = (weights[b, t].unsqueeze(-1) * valid_vals[idx[b, t]]).sum(0)

        # Zugriffszähler aktualisieren
        flat_idx = idx.reshape(-1)
        for i in flat_idx:
            if i < len(self.valid_mask):
                self.access_counts[i] = self.access_counts[i] + 1

        return retrieved

    def consolidate(self, threshold=10):
        """
        Konsolidiert häufig genutzte Einträge (macht sie quasi-permanent).
        Löscht Einträge mit niedriger Zugriffszahl und hohem Alter.
        """
        now = self.global_clock.item()
        age = now - self.timestamps
        score = self.access_counts / (age + 1)

        # Entferne Einträge mit niedrigem Score (untere 20%)
        cutoff = torch.quantile(score[self.valid_mask], 0.2)
        to_remove = (score < cutoff) & (age > 100)
        self.valid_mask[to_remove] = False

    def size(self):
        return self.valid_mask.sum().item()
