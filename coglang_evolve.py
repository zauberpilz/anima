"""Cobra Evolution — Autonome Optimierungsschleife mit Efficiency Features."""
import torch, torch.nn.functional as F, sys, time, json, os, math, gc
sys.path.insert(0, '/home/anima/src')
from coglang import build_anima, AsyncDataLoader, DynamicBatchSizer
from data_loader import get_large_dataset, get_mixed_dataset
from training_controller import TrainingController

# Clean directory paths
CHECKPOINT_DIR = '/home/anima/checkpoints'
CONFIG_FILE = '/home/anima/evolution_config.json'
GENERATION_DIR = '/home/anima/generations'
CONTROL_DIR = '/home/anima/control'

device = 'cuda'
torch.manual_seed(42)

# Konfiguration für die Evolution
config_file = CONFIG_FILE

def load_config():
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return json.load(f)
    else:
        # Start-Konfiguration (SURF MODE - stark gedrosselt)
        return {
            "d_model": 384,
            "d_sparse": 2048,
            "n_layers": 6,
            "d_state": 128,
            "d_context": 256,
            "lr": 0.05,
            "max_vram_mb": 3000, # Stark reduziert für Surf-Kompatibilität
            "generation_step": 50000,
            "best_loss": float('inf'),
            "iteration": 0,
            "use_code_data": False
        }

def save_config(config):
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)


def cosine_anneal_lr(base_lr, step, total_steps, cycle_steps=50000):
    """Cosine Annealing Learning Rate Schedule."""
    progress = (step % max(1, cycle_steps)) / max(1, cycle_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

def run_evolution():
    config = load_config()
    
    # PHASE 20: Multi-Source Data Pipeline
    use_code_data = config.get('use_code_data', False)
    if use_code_data:
        print("[DATA] Lade Mixed Dataset (Code + Text)...")
        data, stoi, itos, vocab_size = get_mixed_dataset(max_chars=5000000, code_ratio=0.3)
    else:
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
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'checkpoint.pt')
    try:
        loaded_config = brain.load_checkpoint(checkpoint_path)
        if loaded_config:
            print("Checkpoint gefunden und geladen!")
            config['best_loss'] = loaded_config.get('best_loss', config['best_loss'])
    except RuntimeError as e:
        if "size mismatch" in str(e) or "Missing key" in str(e):
            print(f"[EVOLUTION] Architektur geändert ({e}). Starte mit frischen Gewichten.")
        else:
            raise e

    print(f'VRAM nach Init: {torch.cuda.memory_allocated()/1024/1024:.0f}MB')
    print(f'Modell-Größe: {brain.parameter_count()/1e6:.1f}M Parameter')

    history = []
    t_start = time.time()
    
    # PHASE 15: Efficiency Features (SURF MODE)
    steps_per_iter = config['generation_step']
    batch_sizer = DynamicBatchSizer(initial_batch=4, initial_seq=64, max_vram_mb=config['max_vram_mb'])
    B, S = batch_sizer.get_sizes()
    async_loader = AsyncDataLoader(data, B, S, device, prefetch=2)  # Weniger Prefetch
    async_loader.start()
    
    # PHASE 17: Resource Throttle für Surf-Kompatibilität (STARK GEDROSSELT)
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(2)  # CPU Threads stark begrenzt
    
    # PHASE 22: System-Priorität senken (Nice Level 19 = niedrigste Prio)
    try:
        os.system("renice -n 19 -p $$ > /dev/null 2>&1")
        os.system("ionice -c 3 -p $$ > /dev/null 2>&1")  # Idle I/O priority
        print("[SURF MODE] CPU/IO-Priorität gesenkt für maximale Browser-Performance")
    except:
        pass
        
    # PHASE 23: Hard VRAM Limit (3GB)
    torch.cuda.set_per_process_memory_fraction(0.45, device=0)  # Max 45% of 8GB ~ 3.6GB
    
    last_log_time = time.time()

    try:
        for step in range(steps_per_iter):
            # PHASE 15: Get batch from async loader
            batch = async_loader.get_batch()
            # PHASE 24: Cosine Annealing LR Scheduler
            current_lr = config['lr']
            if step > 1000:
                current_lr = cosine_anneal_lr(config['lr'], step, steps_per_iter * 2)
            for layer_idx, layer in enumerate(brain._stack.layers):
                layer._lr = current_lr * (0.95 ** layer_idx)
            
            loss, _ = brain.learn(batch)
            history.append(loss)
            
            # PHASE 23: GPU Throttle - kleine Pause um GPU für Browser freizugeben
            time.sleep(0.05)  # 50ms Pause pro Step -> reduziert GPU Load signifikant
            
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
                
                status = f'[{pct:5.1f}%] Step {step:5d} | loss={avg:.4f} | LR={current_lr:.6f} | VRAM={mem:.0f}MB | {speed:.1f}step/s | +{elapsed_m:02d}:{elapsed_s:02d} | ETA {eta_h:02d}:{eta_m:02d}:{eta_s:02d}'
                print(f'\r{status}', end='', flush=True)
                last_log_time = now

                if loss != loss or torch.isnan(torch.tensor(loss)):
                    print('\n!!! NaN DETEKTIERT - Recovery !!!')
                    config['lr'] *= 0.5
                    save_config(config)
                    print(f'  LR auf {config["lr"]:.6f} reduziert')
                    return

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
        if final_loss < config['best_loss'] * 0.99:
            print(f'Neuer Rekord! {final_loss:.4f} < {config["best_loss"]:.4f}')
            config['best_loss'] = final_loss
            if config['d_model'] < 1024:
                config['d_model'] = min(1024, int(config['d_model'] * 1.15))
            config['n_layers'] = min(18, config['n_layers'] + 1)
            config['lr'] *= 0.95
            print(f'Evolution: d_model={config["d_model"]}, layers={config["n_layers"]}, lr={config["lr"]:.6f}')
        elif final_loss < config['best_loss'] * 1.05:
            print(f'Leichter Fortschritt {final_loss:.4f}. Optimiere LR.')
            config['best_loss'] = min(config['best_loss'], final_loss)
            config['lr'] *= 0.9
        else:
            print(f'Kein Fortschritt ({final_loss:.4f} vs {config["best_loss"]:.4f}). Mutation.')
            config['lr'] *= 0.5
            if config['d_sparse'] > 2048:
                config['d_sparse'] = int(config['d_sparse'] * 0.85)
        save_config(config)
        
        # Checkpoints speichern
        brain.save_checkpoint(os.path.join(CHECKPOINT_DIR, 'checkpoint.pt'), config=config)
        if final_loss < config['best_loss']:
            brain.save_checkpoint(os.path.join(CHECKPOINT_DIR, 'best_model.pt'), config=config)
            print("Neues Best Model gespeichert!")
        
        # PHASE 12: Online Evaluation — Automatische Quality Metriken
        gen_scores = []
        for prompt in ['ROMEO:', 'KING ']:
            try:
                ctx = torch.tensor([[stoi.get(c, 0) for c in prompt]], device=device)
                generated = brain.generate_safe(ctx, max_new=150, temperature=0.7, top_k=30)
                gen = ''.join(itos.get(int(i), '?') for i in generated[0])
            except Exception as e:
                print(f'  [WARN] Generation fehlgeschlagen für "{prompt}": {e}')
                gen = f'[GENERATION FAILED]'
            
            with open(os.path.join(GENERATION_DIR, f'evolve_gen_{config["iteration"]}_{prompt.strip()}.txt'), 'w') as f:
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
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print('Cleanup abgeschlossen.')

if __name__ == '__main__':
    print("=== STARTING AUTONOMOUS EVOLUTION LOOP ===")
    while True:
        try:
            run_evolution()
        except Exception as e:
            print(f"CRITICAL ERROR IN LOOP: {e}")
            time.sleep(5)
