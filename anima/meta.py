"""
MetaController + SelfMonitor — Die Meta-Ebene von Anima.

Beobachtet das gesamte System, findet Optimierungen,
und implementiert sie autonom. Das ist Animas Fähigkeit,
sich selbst zu verbessern.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import json
import os


class SelfMonitor(nn.Module):
    """
    Überwacht Systemmetriken und erkennt Optimierungspotential.

    Trackt:
    - Prediction Error pro Layer
    - Speicherauslastung
    - Aktivierungsdichte
    - Verarbeitungszeit pro Schritt
    - Fehlerrate
    """
    def __init__(self, d_model, history_len=1000):
        super().__init__()
        self.history_len = history_len

        self.register_buffer('layer_errors', torch.zeros(4, history_len))
        self.register_buffer('memory_usage', torch.zeros(history_len))
        self.register_buffer('activation_density', torch.zeros(history_len))
        self.register_buffer('step_times', torch.zeros(history_len))
        self.register_buffer('output_scores', torch.zeros(history_len))

        self.step_counter = 0
        self.optimization_log = []

    def record_step(self, layer_errors, mem_usage, density, step_time, score):
        idx = self.step_counter % self.history_len

        for i, err in enumerate(layer_errors):
            if i < self.layer_errors.shape[0]:
                self.layer_errors[i, idx] = err

        self.memory_usage[idx] = mem_usage
        self.activation_density[idx] = density
        self.step_times[idx] = step_time
        self.output_scores[idx] = score
        self.step_counter += 1

    def get_optimization_suggestions(self):
        """Analysiert Metriken und gibt Optimierungsvorschläge."""
        suggestions = []

        if self.step_counter < 100:
            return ["Sammle mehr Daten für Optimierung..."]

        recent = slice(max(0, self.step_counter - 100), self.step_counter)

        # Hoher Fehler in bestimmten Layern?
        for i in range(self.layer_errors.shape[0]):
            recent_errors = self.layer_errors[i, recent]
            if recent_errors.numel() > 0 and recent_errors.mean() > 2.0:
                suggestions.append(f"Layer {i}: Hoher Fehler ({recent_errors.mean():.3f}) -> mehr Kapazität")

        # Aktivierungsdichte
        recent_density = self.activation_density[recent]
        if recent_density.numel() > 0:
            avg_density = recent_density.mean()
            if avg_density > 0.1:
                suggestions.append(f"Hohe Aktivierungsdichte ({avg_density:.3f}) -> Sparsity erhöhen")
            elif avg_density < 0.005:
                suggestions.append(f"Sehr niedrige Dichte ({avg_density:.3f}) -> evtl. zu spärlich")

        # Speichernutzung
        recent_mem = self.memory_usage[recent]
        if recent_mem.numel() > 0 and recent_mem.mean() > 0.8:
            suggestions.append(f"Hohe Speichernutzung ({recent_mem.mean():.1%}) -> konsolidieren")

        # Score-Trend
        if self.step_counter >= 200:
            early_scores = self.output_scores[max(0, self.step_counter-200):self.step_counter-100]
            late_scores = self.output_scores[self.step_counter-100:self.step_counter]
            if early_scores.numel() > 0 and late_scores.numel() > 0:
                if late_scores.mean() < early_scores.mean() * 0.9:
                    suggestions.append("Score sinkt -> Lernrate oder Architektur prüfen")

        return suggestions if suggestions else ["System läuft stabil."]

    def log_optimization(self, description, metric_before, metric_after):
        """Loggt eine durchgeführte Optimierung."""
        entry = {
            'step': self.step_counter,
            'description': description,
            'before': metric_before,
            'after': metric_after,
            'improvement': metric_before - metric_after,
        }
        self.optimization_log.append(entry)
        return entry


class MetaController(nn.Module):
    """
    Meta-Controller: Steuert Compute-Allokation und Lernprozesse.

    Entscheidet dynamisch:
    - Wie viel Rechenleistung pro Input (adaptive compute)
    - Welche Module aktiviert werden (conditional computation)
    - Wann der Speicher konsolidiert wird
    - Wann neue Konzepte gelernt werden sollen (explore vs exploit)
    """
    def __init__(self, d_model, n_layers=4):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        # Compute-Controller (entscheidet über Tiefe pro Token)
        self.compute_router = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, n_layers),
        )

        # Modulationsparameter (lernbar)
        self.learning_rate = nn.Parameter(torch.tensor(0.01))
        self.exploration_rate = nn.Parameter(torch.tensor(0.05))
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # Zustand
        self.register_buffer('performance_history', torch.zeros(500))
        self.register_buffer('compute_budget', torch.tensor(1.0))
        self.self_monitor = None

    def set_monitor(self, monitor):
        self.self_monitor = monitor

    def get_activation_mask(self, x):
        """
        Entscheidet pro Token, welche Layer aktiviert werden.
        Gibt einen [batch, seq, n_layers] bool-Maske zurück.
        """
        # Durchschnitt über Embeddings
        if x.dim() > 2:
            pooled = x.mean(dim=-1)
        else:
            pooled = x

        logits = self.compute_router(pooled)
        mask = torch.sigmoid(logits) > 0.5  # Binary decision
        return mask

    def get_compute_ratio(self, x):
        """Gibt zurück, wie viel Compute verwendet werden soll (0-1)."""
        if x.dim() > 2:
            pooled = x.mean(dim=-1)
        else:
            pooled = x

        logits = self.compute_router(pooled.mean(dim=-1, keepdim=True) if pooled.dim() > 1 else pooled.unsqueeze(0))
        return torch.sigmoid(logits.mean()).item()

    def should_explore(self):
        """Soll das System neue Konzepte erkunden?"""
        if self.performance_history.numel() < 50:
            return True
        recent = self.performance_history[-50:]
        if recent.std() < 0.01:  # Plateau erreicht
            return True
        return torch.rand(1).item() < self.exploration_rate

    def apply_optimization(self, suggestion, metric_fn):
        """
        Wendet einen Optimierungsvorschlag an.
        metric_fn: () -> float (aktueller Metrikwert)
        """
        if self.self_monitor is None:
            return None

        before = metric_fn()
        desc = suggestion

        # Optimierung durchführen
        if "Sparsity" in suggestion:
            self.exploration_rate.data = torch.tensor(max(0.01, self.exploration_rate * 1.1))
        elif "Lernrate" in suggestion or "Lern" in suggestion:
            self.learning_rate.data = self.learning_rate * 0.8
        elif "Speicher" in suggestion:
            self.compute_budget.data = torch.tensor(max(0.3, self.compute_budget * 0.9))

        after = metric_fn()
        return self.self_monitor.log_optimization(desc, before, after)

    def get_status(self):
        """Gibt den aktuellen Systemstatus zurück."""
        return {
            'learning_rate': self.learning_rate.item(),
            'exploration_rate': self.exploration_rate.item(),
            'temperature': self.temperature.item(),
            'compute_budget': self.compute_budget.item(),
            'should_explore': self.should_explore(),
        }
