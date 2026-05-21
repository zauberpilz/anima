"""PHASE 9: Multi-GPU Pipeline — Secondary GPU for parallel evaluation."""
import torch
import torch.nn.functional as F
import threading
import time
import os

class MultiGPUEvaluator:
    """
    Nutzt eine zweite GPU (z.B. RTX 3050) für parallele Evaluation
    während die Haupt-GPU trainiert.
    """
    def __init__(self, brain, stoi, itos, device_secondary='cuda:1'):
        self.brain_main = brain
        self.stoi = stoi
        self.itos = itos
        self.device_secondary = device_secondary
        self.brain_eval = None
        self.eval_thread = None
        self.running = False
        self.last_generation = {}
        self.quality_scores = {}
        
    def check_secondary_gpu(self):
        """Prüfen ob zweite GPU verfügbar ist."""
        if torch.cuda.device_count() >= 2:
            props = torch.cuda.get_device_properties(1)
            print(f"PHASE 9: Secondary GPU gefunden: {props.name} ({props.total_memoria/1024**3:.1f}GB)")
            return True
        print("PHASE 9: Keine zweite GPU gefunden. Evaluation läuft auf Haupt-GPU.")
        return False
    
    def start_background_eval(self, prompts=['ROMEO:', 'KING ', 'JULIET:']):
        """Startet Evaluation im Hintergrund auf sekundärer GPU."""
        if not self.check_secondary_gpu():
            return
            
        self.running = True
        self.eval_thread = threading.Thread(target=self._eval_loop, args=(prompts,), daemon=True)
        self.eval_thread.start()
        print("PHASE 9: Background Evaluation gestartet...")
        
    def _eval_loop(self, prompts):
        """Evaluation Loop auf sekundärer GPU."""
        device = self.device_secondary
        try:
            # Clone model to secondary GPU
            # Note: In practice we'd copy weights periodically
            while self.running:
                for prompt in prompts:
                    ctx = torch.tensor([[self.stoi.get(c, 0) for c in prompt]], device=device)
                    generated = []
                    for _ in range(100):
                        out, _ = self.brain_main.forward(ctx[:, -128:], learn=False)
                        probs = F.softmax(out[:, -1, :] / 0.8, dim=-1)
                        next_token = torch.multinomial(probs, 1)
                        ctx = torch.cat([ctx, next_token], dim=-1)
                        generated.append(next_token.item())
                    
                    text = ''.join(self.itos.get(i, '?') for i in generated)
                    self.last_generation[prompt] = text
                    
                    # Simple quality score
                    words = text.split()
                    if len(words) > 0:
                        unique_ratio = len(set(words)) / len(words)
                        self.quality_scores[prompt] = unique_ratio
                
                time.sleep(30)  # Alle 30 Sekunden evaluieren
        except Exception as e:
            print(f"PHASE 9 Eval Error: {e}")
            
    def stop(self):
        self.running = False
        
    def get_results(self):
        return {
            'generations': self.last_generation,
            'quality_scores': self.quality_scores
        }
