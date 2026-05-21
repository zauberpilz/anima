"""Cobra Evolution — Autonome Optimierungsschleife mit Efficiency Features."""
import torch, torch.nn.functional as F, sys, time, json, os
sys.path.insert(0, '/home/anima')
from coglang import build_anima, AsyncDataLoader, DynamicBatchSizer
from data_loader import get_large_dataset
from training_controller import TrainingController

device = 'cuda'
torch.manual_seed(42)

# Konfiguration für die Evolution
config_file = '/home/anima/evolution_config.json'

def load_config():
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return json.load(f)
    else:
        # Start-Konfiguration (Basis Cobra) - Streaming-optimiert
        return {
            "d_model": 384,
            "d_sparse": 2048,
            "n_layers": 6,
            "d_state": 128,
            "d_context": 256,
            "lr": 0.05,
            "max_vram_mb": 4500, # Limit für Streaming-Kompatibilität (3.5GB frei)
            "generation_step": 50000,
            "best_loss": float('inf'),
            "iteration": 0
        }

def save_config(config):
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)

def run_evolution():
    config = load_config()
    data, stoi, itos, vocab_size = get_large_dataset(max_chars=5000000)
    if isinstance(data, torch.Tensor): data = data.long()
    else: data = torch.tensor(data, dtype=torch.long)
    
    # PHASE 16: Training Controller
    controller = TrainingController()

    print('\n' + '='*60)
    print(f'EVOLUTION ITERATION {config["iteration"]}')
    print('='*60)
    print(f'Steuerung: pause/resume/stop via training_controller.py')
    print(f'  Pause:  python3 training_controller.py pause')
    print(f'  Resume: python3 training_controller.py resume')
    print(f'  Stop:   python3 training_controller.py stop')
    
    brain = build_anima(vocab_size=vocab_size, device=device, 
                        d_model=config['d_model'], d_sparse=config['d_sparse'], 
                        n_layers=config['n_layers'], d_state=config['d_state'], 
                        d_context=config['d_context'], lr=config['lr'])
    
    # Checkpoint Loading
    checkpoint_path = '/home/anima/checkpoint.pt'
    loaded_config = brain.load_checkpoint(checkpoint_path)
    if loaded_config:
        print("Checkpoint gefunden und geladen!")
        # Optional: Config aus Checkpoint übernehmen falls nötig
        config['best_loss'] = loaded_config.get('best_loss', config['best_loss'])

    print(f'VRAM nach Init: {torch.cuda.memory_allocated()/1024/1024:.0f}MB')
    print(f'Modell-Größe: {brain.parameter_count()/1e6:.1f}M Parameter')

    history = []
    t_start = time.time()
    
    # PHASE 15: Efficiency Features
    steps_per_iter = config['generation_step']
    batch_sizer = DynamicBatchSizer(initial_batch=8, initial_seq=128, max_vram_mb=config['max_vram_mb'])
    B, S = batch_sizer.get_sizes()
    async_loader = AsyncDataLoader(data, B, S, device, prefetch=4)
    async_loader.start()
    
    # PHASE 17: Resource Throttle für Surf-Kompatibilität
    # Begrenzt GPU utilization auf ~70% damit Browser-traffic Priorität hat
    torch.backends.cudnn.benchmark = False  # Weniger VRAM-Spitzen
    torch.set_num_threads(4)  # CPU Threads begrenzen
    
    last_log_time = time.time()

    try:
        for step in range(steps_per_iter):
            # PHASE 15: Get batch from async loader
            batch = async_loader.get_batch()
            loss, _ = brain.learn(batch)
            history.append(loss)
            
            # PHASE 15: Dynamic batch sizing every 1000 steps
            if step % 1000 == 0:
                vram_used = torch.cuda.max_memory_allocated() / 1024 / 1024
                batch_sizer.adjust(vram_used)
                new_B, new_S = batch_sizer.get_sizes()
                if new_B != B or new_S != S:
                    B, S = new_B, new_S
                    async_loader.batch_size = B
                    async_loader.seq_length = S
                    print(f'\n[EFFICIENCY] Batch angepasst: B={B}, S={S}')

            now = time.time()
            if now - last_log_time >= 1.0:
                # PHASE 16: Check pause/stop
                if controller.check_stop():
                    print("Training gestoppt durch Controller.")
                    return
                controller.check_pause()
                
                mem = torch.cuda.max_memory_allocated() / 1024 / 1024
                avg = sum(history[-500:]) / 500 if len(history) >= 500 else loss
                elapsed = now - t_start
                speed = (step + 1) / elapsed if elapsed > 0 else 0
                pct = step / steps_per_iter * 100
                
                elapsed_m, elapsed_s = divmod(int(elapsed), 60)
                remaining_steps = steps_per_iter - step
                eta_secs = remaining_steps / speed if speed > 0 else 0
                eta_h, eta_rem = divmod(int(eta_secs), 3600)
                eta_m, eta_s = divmod(eta_rem, 60)
                
                status = f'[{pct:5.1f}%] Step {step:5d} | loss={avg:.4f} | VRAM={mem:.0f}MB | {speed:.1f}step/s | +{elapsed_m:02d}:{elapsed_s:02d} | ETA {eta_h:02d}:{eta_m:02d}:{eta_s:02d}'
                print(f'\r{status}', end='', flush=True)
                last_log_time = now

                if loss != loss:
                    print('\n!!! NaN DETEKTIERT !!!')
                    return # Abbruch der Iteration, zur Mutation/Reset

            # Sicherheitscheck gegen OOM
            if torch.cuda.max_memory_allocated() / 1024 / 1024 > config['max_vram_mb']:
                print('!!! VRAM LIMIT ERREICHT !!!')
                return

        # Iteration erfolgreich beendet
        final_loss = min(history)
        elapsed = time.time() - t_start
        print(f'\nIteration {config["iteration"]} abgeschlossen in {elapsed/60:.1f}m.')
        print(f'Best Loss dieser Iteration: {final_loss:.4f}')

        # --- MUTATION / EVOLUTION LOGIK ---
        config['iteration'] += 1
        if final_loss < config['best_loss']:
            print(f'Neuer Rekord! {final_loss:.4f} < {config["best_loss"]:.4f}')
            config['best_loss'] = final_loss
            # Bei Erfolg: Kapazität leicht erhöhen (Scaling)
            config['d_model'] = int(config['d_model'] * 1.2)
            config['n_layers'] += 1
            print(f'Evolution: Modell wird vergrößert -> d_model={config["d_model"]}, layers={config["n_layers"]}')
        else:
            # Wenn kein neuer Rekord, versuchen die Lernrate zu optimieren oder Layer-Struktur leicht zu variieren
            print('Kein Fortschritt. Mutation: Ändere Learning Rate.')
            config['lr'] *= 0.8 # Decay

        save_config(config)
        
        # Checkpoints speichern
        brain.save_checkpoint('/home/anima/checkpoint.pt', config=config)
        if final_loss < config['best_loss']:
            brain.save_checkpoint('/home/anima/best_model.pt', config=config)
            print("Neues Best Model gespeichert!")
        
        # PHASE 12: Online Evaluation — Automatische Quality Metriken
        gen_scores = []
        for prompt in ['ROMEO:', 'KING ']:
            ctx = torch.tensor([[stoi.get(c, 0) for c in prompt]], device=device)
            for _ in range(150):
                out, info = brain.forward(ctx[:, -128:], learn=False)
                logits = out[:, -1, :] / 0.8
                
                k = 40
                top_k_logits, top_k_indices = torch.topk(logits, k)
                probs = F.softmax(top_k_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                next_token = top_k_indices.gather(1, next_token)
                
                ctx = torch.cat([ctx, next_token], dim=-1)
            gen = ''.join(itos.get(int(i), '?') for i in ctx[0])
            with open(f'/home/anima/evolve_gen_{config["iteration"]}_{prompt.strip()}.txt', 'w') as f:
                f.write(gen)
            
            # PHASE 12: Quality Metrics
            tokens = gen.split()
            if len(tokens) > 0:
                unique_ratio = len(set(tokens)) / len(tokens)
                trigrams = [tuple(tokens[i:i+3]) for i in range(len(tokens)-2)]
                rep_score = 1.0 - (len(set(trigrams)) / max(1, len(trigrams)))
                # PHASE 12: New metrics
                avg_word_len = sum(len(t) for t in tokens) / len(tokens)
                punctuation_ratio = sum(1 for t in tokens if any(c in t for c in '.,!?;:')) / len(tokens)
                score = unique_ratio * (1.0 - rep_score * 0.5) * (avg_word_len / 5.0) * (1.0 + punctuation_ratio)
                gen_scores.append(score)
        
        avg_gen_score = sum(gen_scores) / len(gen_scores) if gen_scores else 0
        print(f'Generation Quality Score: {avg_gen_score:.3f}')

    except Exception as e:
        print(f'Fehler in Iteration {config["iteration"]}: {e}')
    finally:
        # Speicher leeren für nächste Iteration
        del brain
        torch.cuda.empty_cache()
        print('Cleanup abgeschlossen.')

if __name__ == '__main__':
    print("=== STARTING AUTONOMOUS EVOLUTION LOOP ===")
    while True:
        try:
            run_evolution()
        except Exception as e:
            print(f"CRITICAL ERROR IN LOOP: {e}")
            time.sleep(5)
