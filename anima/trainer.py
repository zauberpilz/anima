"""Anima Trainer — 100% Backprop-frei. Nur Hebbian + Delta-Regeln."""
import torch
import torch.nn.functional as F
import math

class AnimaTrainer:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.history = []

    def train_step(self, input_ids):
        self.model.train()
        input_ids = input_ids.to(self.device)
        loss = self.model.learn_from_interaction(input_ids)
        self.history.append(loss)
        return {'loss': loss}

    @torch.no_grad()
    def evaluate(self, input_ids):
        self.model.eval()
        input_ids = input_ids.to(self.device)
        out, info = self.model(input_ids, learn=False)
        loss = F.cross_entropy(out.view(-1, out.size(-1)), input_ids.view(-1))
        return {'loss': loss.item(), 'ppl': math.exp(loss.item()), 'density': info['density']}

    def reflect(self):
        return self.model.self_reflect()
