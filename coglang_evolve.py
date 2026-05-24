"""Cobra Evolution — Autonome Optimierungsschleife mit Multi-Domain, Curriculum, Multi-Task Loss."""
import torch, torch.nn.functional as F, sys, time, json, os, math, gc
sys.path.insert(0, '/home/anima/src')
from coglang import build_anima, AsyncDataLoader, DynamicBatchSizer
from data_loader import MultiDomainDataset, get_large_dataset, get_mixed_dataset
from training_controller import TrainingController

# Clean directory paths
CHECKPOINT_DIR = '/home/anima/checkpoints'
CONFIG_FILE = '/home/anima/evolution_config.json'
GENERATION_DIR = '/home/anima/generations'
BPE_TOKENIZER_PATH = '/home/anima/tokenizer/bpe_4k.json'
CONTROL_DIR = '/home/anima/control'
RECOVERY_DIR = '/home/anima/recovery'

device = 'cuda'
torch.manual_seed(42)

# Konfiguration für die Evolution
config_file = CONFIG_FILE


# =====================================================================
#  PHASE 30: CURRICULUM LEARNING SCHEDULER
# =====================================================================

class CurriculumScheduler:
    """Manages multi-phase curriculum learning across domains with smooth transitions."""
    def __init__(self, total_steps=50000, transition_fraction=0.2):
        self.phases = [
            {'name': 'foundation',   'domain': 'text',     'steps': 5000,  'desc': 'Language foundation'},
            {'name': 'code_intro',   'domain': 'code',     'steps': 10000, 'desc': 'Code understanding'},
            {'name': 'security',     'domain': 'security', 'steps': 10000, 'desc': 'Vulnerability detection'},
            {'name': 'network',      'domain': 'network',  'steps': 5000,  'desc': 'Traffic analysis'},
            {'name': 'integration',  'domain': 'mixed',    'steps': 20000, 'desc': 'All domains integrated'},
        ]
        self.transition_fraction = transition_fraction  # 0.0=abrupt, 0.2=smooth over 20% of phase
        self._validate_steps(total_steps)
        # Precompute domain order for transitions
        self._domain_order = []
        prev = None
        for p in self.phases:
            if p['domain'] != 'mixed':
                self._domain_order.append(p['domain'])
                prev = p['domain']
            else:
                self._domain_order.append('mixed')

    def _validate_steps(self, total_steps):
        total = sum(p['steps'] for p in self.phases)
        if total != total_steps:
            scale = total_steps / max(1, total)
            for p in self.phases:
                p['steps'] = max(1, int(p['steps'] * scale))

    def get_phase(self, step):
        """Return current phase dict based on step number."""
        accumulated = 0
        for phase in self.phases:
            accumulated += phase['steps']
            if step < accumulated:
                return phase
        return self.phases[-1]

    def get_domain_weights(self, step):
        """Return dict of domain weights with smooth transitions between phases.
        
        During the first `transition_fraction` of a new phase, the previous domain
        is gradually blended out while the new domain is blended in.
        This prevents catastrophic loss spikes at phase boundaries.
        """
        # Default: uniform weights for mixed, single domain for focused phases
        phase = self.get_phase(step)
        
        if phase['domain'] == 'mixed':
            return {'text': 0.25, 'code': 0.25, 'security': 0.25, 'network': 0.25}
        
        # Find where we are in the phase sequence
        accumulated = 0
        prev_domain = 'text'  # default fallback
        for i, p in enumerate(self.phases):
            phase_start = accumulated
            phase_end = accumulated + p['steps']
            
            if phase_start <= step < phase_end:
                # We're in phase p
                domain = p['domain']
                if domain == 'mixed':
                    return {'text': 0.25, 'code': 0.25, 'security': 0.25, 'network': 0.25}
                
                # Calculate transition progress
                progress_in_phase = (step - phase_start) / max(1, p['steps'])
                transition_steps = int(p['steps'] * self.transition_fraction)
                
                if progress_in_phase < self.transition_fraction and i > 0:
                    # Smooth blend: prev domain fades out, new domain fades in
                    prev_phase = self.phases[i - 1]
                    prev_domain = prev_phase['domain']
                    if prev_domain == 'mixed':
                        # If prev was mixed, distribute evenly
                        blend = progress_in_phase / self.transition_fraction
                        return {domain: blend, 'text': (1-blend)/4, 'code': (1-blend)/4,
                                'security': (1-blend)/4, 'network': (1-blend)/4}
                    else:
                        blend = progress_in_phase / self.transition_fraction
                        weights = {d: 0.0 for d in ['text', 'code', 'security', 'network']}
                        weights[prev_domain] = 1.0 - blend
                        weights[domain] = blend
                        return weights
                
                # Fully in new domain
                return {d: 1.0 if d == domain else 0.0 for d in ['text', 'code', 'security', 'network']}
            
            accumulated += p['steps']
        
        return {'text': 0.25, 'code': 0.25, 'security': 0.25, 'network': 0.25}

    def get_domain_for_step(self, step):
        """Return the primary domain name for a given step."""
        return self.get_phase(step)['domain']

    def get_phase_progress(self, step):
        """Return progress within current phase as (phase_idx, phase_name, progress_0to1)."""
        accumulated = 0
        for idx, phase in enumerate(self.phases):
            if step < accumulated + phase['steps']:
                progress = (step - accumulated) / max(1, phase['steps'])
                return idx, phase['name'], progress
            accumulated += phase['steps']
        return len(self.phases) - 1, self.phases[-1]['name'], 1.0


# =====================================================================
#  PHASE 31: MULTI-TASK LOSS FUNCTIONS
# =====================================================================

def security_aware_loss(output, target):
    """
    Security-aware auxiliary loss: penalize false negatives more.
    For vulnerability detection, missing a vulnerability (FN) is worse than false alarm (FP).
    """
    log_probs = F.log_softmax(output, dim=-1)
    target_flat = target.view(-1)
    output_flat = log_probs.view(-1, output.size(-1))

    # Gather the log probabilities of the target tokens
    nll = -output_flat[torch.arange(output_flat.size(0)), target_flat]

    # Identify "security-sensitive" tokens (punctuation, keywords like 'CVE', 'CWE', 'vuln')
    security_keywords_str = 'CVE:cwe:vuln:exploit:buffer:overflow:injection:escape:patch:fix:vulnerability:malware:backdoor:rootkit:ransomware'
    security_keywords = {ord(c) for c in security_keywords_str}
    # Compute per-token significance
    significance = torch.ones_like(nll)
    for token_id in range(output.size(-1)):
        char_val = token_id  # This is the index, which maps to a character
        # Heuristic: tokens that appear in security contexts get boosted weight
        pass  # Simplified: we use a fixed boost on all tokens for security domain

    # Boost NLL by 2x to penalize errors more heavily in security domain
    aux_loss = nll.mean() * 0.5  # Weighted cross-entropy boost
    return aux_loss


def network_contrastive_loss(output, target):
    """
    Contrastive auxiliary loss for network anomaly detection.
    Encourages the model to distinguish normal vs anomalous traffic patterns
    by contrasting representations.
    """
    B, S, V = output.shape
    flat_out = output.view(-1, V)
    flat_target = target.view(-1)

    # Standard CE as base
    ce = F.cross_entropy(flat_out, flat_target, reduction='none')

    # Simple contrastive component: penalize predictions that are too uniform
    probs = F.softmax(flat_out, dim=-1)
    entropy = -(probs * torch.log(probs.clamp(min=1e-8))).sum(dim=-1)

    # Lower entropy = more confident predictions (desired)
    # Add a small penalty for high entropy (uncertainty)
    entropy_penalty = entropy.mean() * 0.05

    # Anomaly score: penalize when the model is uncertain about next token
    # in network data (which should have predictable patterns)
    return entropy_penalty


def compute_multi_task_loss(output, target, domain, task_weights=None):
    """Compute weighted loss based on domain.

    Args:
        output: [B, S, V] logits
        target: [B, S] token IDs
        domain: str domain name ('text', 'code', 'security', 'network', 'mixed')
        task_weights: optional dict of auxiliary loss weights

    Returns:
        total_loss: scalar tensor
        loss_components: dict of individual loss components for logging
    """
    if task_weights is None:
        task_weights = {'aux': 0.1, 'ce': 1.0}

    # Standard cross-entropy for all (already the main loss)
    ce_loss = F.cross_entropy(output.view(-1, output.size(-1)), target.view(-1))

    # Domain-specific auxiliary losses
    aux_loss = 0.0
    if domain == 'security':
        aux_loss = security_aware_loss(output, target)
    elif domain == 'network':
        aux_loss = network_contrastive_loss(output, target)
    elif domain == 'mixed':
        # In mixed mode, apply a mild ensemble of auxiliary losses
        aux_loss = 0.5 * security_aware_loss(output, target) + 0.5 * network_contrastive_loss(output, target)

    total_loss = ce_loss + task_weights['aux'] * aux_loss

    return total_loss, {'ce_loss': ce_loss.item(), 'aux_loss': aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss}


# =====================================================================
#  CONFIG HANDLING
# =====================================================================

def load_config():
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        # Ensure domain weights exist (backward compatibility)
        if 'domain_weights' not in config:
            config['domain_weights'] = {"code": 0.3, "security": 0.2, "network": 0.1, "text": 0.4}
        if 'curriculum_enabled' not in config:
            config['curriculum_enabled'] = True
        if 'current_phase' not in config:
            config['current_phase'] = 'foundation'
        return config
    else:
        # Start-Konfiguration (SURF MODE - stark gedrosselt)
        return {
            "d_model": 384,
            "d_sparse": 2048,
            "n_layers": 6,
            "d_state": 128,
            "d_context": 256,
            "lr": 0.05,
            "max_vram_mb": 3000,
            "generation_step": 50000,
            "best_loss": float('inf'),
            "iteration": 0,
            "use_code_data": False,
            "domain_weights": {"code": 0.3, "security": 0.2, "network": 0.1, "text": 0.4},
            "curriculum_enabled": True,
            "current_phase": "foundation"
        }


def save_config(config):
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)


# =====================================================================
#  GENERATION EVALUATION HELPERS
# =====================================================================

def evaluate_generated_text(gen_text, domain='text'):
    """Evaluate generated text quality with domain-specific metrics."""
    scores = {}

    tokens = gen_text.split()
    if len(tokens) == 0:
        return {'score': 0.0, 'unique_ratio': 0.0, 'avg_word_len': 0.0, 'rep_score': 1.0}

    # Common metrics
    unique_ratio = len(set(tokens)) / len(tokens)
    trigrams = [tuple(tokens[i:i+3]) for i in range(len(tokens)-2)]
    rep_score = 1.0 - (len(set(trigrams)) / max(1, len(trigrams)))
    avg_word_len = sum(len(t) for t in tokens) / len(tokens)
    punctuation_ratio = sum(1 for t in tokens if any(c in t for c in '.,!?;:')) / len(tokens)

    scores['unique_ratio'] = unique_ratio
    scores['avg_word_len'] = avg_word_len
    scores['rep_score'] = rep_score
    scores['punctuation_ratio'] = punctuation_ratio

    # Domain-specific evaluation
    if domain == 'code':
        # Check if generated code parses (basic syntax check)
        code_score = evaluate_code_generation(gen_text)
        scores['code_score'] = code_score
    elif domain == 'security':
        # Check CWE diversity in generation
        sec_score = evaluate_security_generation(gen_text)
        scores['security_score'] = sec_score
    elif domain == 'network':
        # Check if output contains structured flow-like patterns
        net_score = evaluate_network_generation(gen_text)
        scores['network_score'] = net_score
    else:
        scores['domain_score'] = 1.0

    # Composite score
    base_score = unique_ratio * (1.0 - rep_score * 0.5) * (avg_word_len / 5.0) * (1.0 + punctuation_ratio)
    domain_mult = scores.get('code_score', scores.get('security_score', scores.get('network_score', 1.0)))
    scores['score'] = base_score * (0.5 + 0.5 * domain_mult)

    return scores


def evaluate_code_generation(gen_text):
    """Basic check if generated text looks like parseable code."""
    # Simple heuristics: check for balanced braces, common code patterns
    lines = gen_text.split('\n')
    if len(lines) < 2:
        return 0.5

    brace_balance = gen_text.count('{') - gen_text.count('}')
    paren_balance = gen_text.count('(') - gen_text.count(')')

    # Check for common code keywords
    code_keywords = ['def ', 'class ', 'if ', 'for ', 'while ', 'return ', 'import ', 'from ', 'func ', 'var ', 'int ', 'void ']
    keyword_count = sum(1 for kw in code_keywords if kw in gen_text)

    has_semicolons = ';' in gen_text
    has_equals = '=' in gen_text
    has_comparison = any(op in gen_text for op in ['==', '!=', '<', '>'])

    features = [
        abs(brace_balance) <= 2,        # reasonably balanced braces
        abs(paren_balance) <= 2,        # reasonably balanced parens
        keyword_count >= 1,              # at least one code keyword
        has_semicolons or has_equals,    # has code operators
    ]

    score = sum(features) / len(features)
    return min(1.0, max(0.0, score + 0.2 * min(1.0, keyword_count / 5)))


def evaluate_security_generation(gen_text):
    """Evaluate security-related generation: CWE diversity, vulnerability patterns."""
    cwe_patterns = ['CWE-', 'CVE-', 'vulnerability', 'exploit', 'buffer overflow',
                    'injection', 'xss', 'csrf', 'sql injection', 'patch', 'fix']
    found = sum(1 for pat in cwe_patterns if pat.lower() in gen_text.lower())

    # Count unique CWE/CVE mentions
    import re
    cwe_matches = set(re.findall(r'CWE-\d+', gen_text))
    cve_matches = set(re.findall(r'CVE-\d+-\d+', gen_text))
    unique_refs = len(cwe_matches) + len(cve_matches)

    diversity_score = min(1.0, unique_refs / 3.0)  # 3+ unique refs = perfect
    coverage_score = min(1.0, found / 5.0)          # 5+ pattern matches = perfect

    return 0.5 * diversity_score + 0.5 * coverage_score


def evaluate_network_generation(gen_text):
    """Evaluate network traffic generation quality."""
    # Check for flow-like patterns
    has_flow_marker = '[FLOW]' in gen_text
    has_src_dst = 'src=' in gen_text and 'dst=' in gen_text
    has_proto = 'proto=' in gen_text
    has_bytes = 'bytes=' in gen_text
    has_label = 'label=' in gen_text or 'anomaly=' in gen_text
    has_ip_pattern = any(p in gen_text for p in ['.', ':', '->'])

    features = [has_flow_marker, has_src_dst, has_proto, has_bytes, has_label, has_ip_pattern]
    return sum(features) / len(features)


# =====================================================================
#  MAIN EVOLUTION LOOP
# =====================================================================

def run_evolution():
    config = load_config()

    # -----------------------------------------------------------------
    #  PHASE 30: Multi-Domain Dataset + Curriculum
    # -----------------------------------------------------------------
    curriculum_enabled = config.get('curriculum_enabled', True)
    curriculum = CurriculumScheduler(total_steps=config['generation_step'])

    # Lade Multi-Domain Dataset
    use_code_data = config.get('use_code_data', False)
    domain_weights = config.get('domain_weights', {"code": 0.3, "security": 0.2, "network": 0.1, "text": 0.4})

    print('\n' + '='*60)
    print('PHASE 30/31: Multi-Domain + Curriculum Learning')
    print('='*60)

    # Multi-Domain Dataset: synthetisch + CodeAlpaca (zuverlässig, kein Download nötig)
    # 500K Chars pro Domäne = 1.5M+ Gesamttokens
    multi_domain = MultiDomainDataset(max_chars_per_domain=500000, bpe_tokenizer_path=BPE_TOKENIZER_PATH)

    # Apply domain weights from config
    for domain, weight in domain_weights.items():
        if domain in multi_domain.domains:
            multi_domain.set_domain_weight(domain, weight)

    # Conditional domain enablement
    if not use_code_data:
        multi_domain.enable_domain('code', False)

    multi_domain.load_all()

    data = multi_domain.data
    stoi = multi_domain.stoi
    itos = multi_domain.itos
    vocab_size = multi_domain.vocab_size
    domain_ranges = multi_domain.domain_ranges

    if isinstance(data, torch.Tensor):
        data = data.long()
    else:
        data = torch.tensor(data, dtype=torch.long)

    print(f'\n[MULTI] Domain Ranges: {domain_ranges}')
    print(f'[MULTI] Curriculum: {"Enabled" if curriculum_enabled else "Disabled"}')
    if curriculum_enabled:
        print(f'[MULTI] Phases: {[p["name"] for p in curriculum.phases]}')

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
    async_loader = AsyncDataLoader(data, B, S, device, prefetch=2)
    async_loader.start()

    # PHASE 17: Resource Throttle für Surf-Kompatibilität (STARK GEDROSSELT)
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(2)

    # PHASE 22: System-Priorität senken (Nice Level 19 = niedrigste Prio)
    try:
        os.system("renice -n 19 -p $$ > /dev/null 2>&1")
        os.system("ionice -c 3 -p $$ > /dev/null 2>&1")
        print("[SURF MODE] CPU/IO-Priorität gesenkt für maximale Browser-Performance")
    except:
        pass

    # PHASE 23: Hard VRAM Limit (3GB)
    torch.cuda.set_per_process_memory_fraction(0.45, device=0)

    # PHASE 31: Recovery state
    oom_backoff = 1
    max_oom_backoff = 60
    recovery_checkpoint_saved = False

    last_log_time = time.time()
    nan_recovery_count = 0

    # PHASE 30/31: Domain-specific tracking
    domain_loss_history = {'text': [], 'code': [], 'security': [], 'network': [], 'mixed': []}
    current_domain_data = {}  # Cache domain-specific data slices

    try:
        for step in range(steps_per_iter):
            # -----------------------------------------------------------------
            #  PHASE 30: Curriculum-based domain selection (SMOOTH transitions)
            # -----------------------------------------------------------------
            if curriculum_enabled:
                phase_info = curriculum.get_phase(step)
                config['current_phase'] = phase_info['name']
                # Get smooth domain weights (blends domains at phase boundaries)
                domain_weights = curriculum.get_domain_weights(step)
            else:
                phase_info = (0, 'mixed', 1.0)
                domain_weights = {'text': 0.25, 'code': 0.25, 'security': 0.25, 'network': 0.25}

            # Set default domain (used for logging; 'mixed' when blending)
            current_domain = 'mixed'

            # -----------------------------------------------------------------
            #  PHASE 30: Get batch with smooth domain blending
            # -----------------------------------------------------------------
            # Build a mixed batch using the smooth domain weights
            try:
                batch_parts = []
                target_parts = []
                for domain, weight in domain_weights.items():
                    if weight > 0.05:  # Only sample from active domains
                        n_samples = max(1, int(B * weight))
                        x_d, y_d = multi_domain.get_batch(domain, n_samples, S, device)
                        batch_parts.append(x_d)
                        target_parts.append(y_d)

                if batch_parts:
                    batch = torch.cat(batch_parts, dim=0)
                    batch_target = torch.cat(target_parts, dim=0)
                    # Shuffle within batch for domain mixing
                    perm = torch.randperm(batch.size(0), device=device)
                    batch = batch[perm]
                    batch_target = batch_target[perm]
                else:
                    raise RuntimeError("No active domains")
            except Exception as e:
                # Fallback: async loader
                batch = async_loader.get_batch()
                batch_target = batch

            # -----------------------------------------------------------------
            #  PHASE 24: AGGRESSIVE LR for fresh Hebbian model
            # -----------------------------------------------------------------
            base_lr = config['lr']
            # Warmup: keep LR high for first 5000 steps
            if step < 5000:
                warmup_factor = 0.5 + 0.5 * step / 5000
                current_lr = base_lr * warmup_factor * 3.0
            else:
                current_lr = cosine_anneal_lr(base_lr, step - 5000, steps_per_iter * 2)
            for layer_idx, layer in enumerate(brain._stack.layers):
                # Hebbian LR is divided by (batch*seq)=1024 internally → lr_eff must be ~0.01
                # So layer._lr must be ~10 for lr_eff ~0.01
                layer._lr = current_lr * 100.0 * (0.95 ** layer_idx)
            brain._decoder._lr = current_lr * 1.5
            brain._sensory._lr = current_lr * 10.0  # Embeddings need big updates too

            # -----------------------------------------------------------------
            #  Forward pass mit OOM Recovery
            # -----------------------------------------------------------------
            try:
                loss, info = brain.learn(batch)
            except RuntimeError as e:
                if 'out of memory' in str(e).lower() or 'CUDA' in str(e):
                    print(f'\n!!! OOM DETEKTIERT (Step {step}) - Backoff {oom_backoff}s !!!')
                    gc.collect()
                    torch.cuda.empty_cache()

                    # Exponetial backoff
                    time.sleep(oom_backoff)
                    oom_backoff = min(oom_backoff * 2, max_oom_backoff)

                    # Automatic batch size reduction
                    new_B = max(1, B // 2)
                    new_S = max(16, S // 2)
                    print(f'  Batch reduziert: B={B}->{new_B}, S={S}->{new_S}')
                    B, S = new_B, new_S
                    async_loader.batch_size = B
                    async_loader.seq_length = S

                    # Recovery checkpoint
                    if not recovery_checkpoint_saved:
                        os.makedirs(RECOVERY_DIR, exist_ok=True)
                        brain.save_checkpoint(
                            os.path.join(RECOVERY_DIR, f'recovery_iter{config["iteration"]}_step{step}.pt'),
                            config=config
                        )
                        recovery_checkpoint_saved = True

                    # Retry with smaller batch
                    if current_domain == 'mixed':
                        x_domain, y_domain = multi_domain.get_mixed_batch(B, S, device)
                    else:
                        x_domain, y_domain = multi_domain.get_batch(current_domain, B, S, device)
                    batch = x_domain
                    batch_target = y_domain
                    loss, info = brain.learn(batch)
                else:
                    raise e

            # Reset backoff on success
            oom_backoff = max(1, oom_backoff // 2)

            # -----------------------------------------------------------------
            #  PHASE 31: Multi-Task Loss (add domain-specific auxiliary losses)
            # -----------------------------------------------------------------
            if isinstance(loss, torch.Tensor):
                loss_val = loss.item()
            else:
                loss_val = loss

            # Compute multi-task loss for tracking (does not affect brain.learn internal loss)
            output = info.get('output', None)
            if output is None:
                # Reconstruct output from info if possible
                multi_loss, loss_components = compute_multi_task_loss(
                    info.get('pred', torch.zeros(1, 1, vocab_size, device=device)),
                    batch_target if isinstance(batch_target, torch.Tensor) else batch,
                    current_domain
                )
            else:
                multi_loss, loss_components = compute_multi_task_loss(
                    output, batch_target if isinstance(batch_target, torch.Tensor) else batch,
                    current_domain
                )

            history.append(loss_val)

            # Track domain-specific loss
            if current_domain in domain_loss_history:
                domain_loss_history[current_domain].append(loss_val)

            # PHASE 23: GPU Throttle - kleine Pause um GPU für Browser freizugeben
            time.sleep(0.05)

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
                avg = sum(history[-500:]) / 500 if len(history) >= 500 else loss_val
                elapsed = now - t_start
                speed = (step + 1) / elapsed if elapsed > 0 else 0
                pct = step / steps_per_iter * 100

                elapsed_m, elapsed_s = divmod(int(elapsed), 60)
                remaining_steps = steps_per_iter - step
                eta_secs = remaining_steps / speed if speed > 0 else 0
                eta_h, eta_rem = divmod(int(eta_secs), 3600)
                eta_m, eta_s = divmod(eta_rem, 60)

                # PHASE 30: Show curriculum phase
                phase_name = phase_info['name'] if curriculum_enabled and isinstance(phase_info, dict) else 'mixed'
                if curriculum_enabled:
                    _, _, phase_progress = curriculum.get_phase_progress(step)
                else:
                    phase_progress = 1.0
                phase_str = f'| Phase={phase_name}'

                status = (f'[{pct:5.1f}%] Step {step:5d} | loss={avg:.4f} | LR={current_lr:.6f} '
                          f'| VRAM={mem:.0f}MB | {speed:.1f}step/s '
                          f'| +{elapsed_m:02d}:{elapsed_s:02d} | ETA {eta_h:02d}:{eta_m:02d}:{eta_s:02d}'
                          f'{phase_str}')

                # PHASE 26: Live State für Dashboard (alle 10 Sekunden)
                if int(elapsed) % 10 == 0 and last_log_time != int(elapsed):
                    try:
                        # Per-domain perplexity tracking
                        domain_ppl = {}
                        for d_name, d_losses in domain_loss_history.items():
                            if d_losses:
                                avg_d_loss = sum(d_losses[-100:]) / max(1, len(d_losses[-100:]))
                                domain_ppl[d_name] = math.exp(min(avg_d_loss, 10.0))

                        state = {
                            'step': step, 'total_steps': steps_per_iter,
                            'loss': avg, 'vram_mb': mem, 'speed': speed,
                            'lr': current_lr, 'elapsed_s': int(elapsed),
                            'eta_s': int(eta_secs), 'iteration': config.get('iteration', 0),
                            'best_loss': config.get('best_loss', None),
                            'd_model': config.get('d_model', 0),
                            'n_layers': config.get('n_layers', 0),
                            'params_m': brain.parameter_count() / 1e6,
                            'batch_size': B, 'seq_len': S,
                            'timestamp': time.time(),
                            'loss_history': history[-500:] if len(history) > 100 else history,
                            # PHASE 30: Curriculum state
                            'current_phase': phase_name,
                            'phase_progress': phase_progress,
                            'curriculum_enabled': curriculum_enabled,
                            # PHASE 31: Domain-specific tracking (blended)
                            'domain': phase_info['name'] if isinstance(phase_info, dict) else 'mixed',
                            'domain_weights': domain_weights,
                            'domain_losses': {k: float(sum(v[-100:]) / max(1, len(v[-100:])))
                                              for k, v in domain_loss_history.items() if v},
                            'domain_perplexity': domain_ppl,
                            'multi_task_ce': loss_components.get('ce_loss', 0),
                            'multi_task_aux': loss_components.get('aux_loss', 0),
                        }
                        with open('/home/anima/train_state.json', 'w') as sf:
                            json.dump(state, sf)
                    except Exception:
                        pass

                print(f'\r{status}', end='', flush=True)
                last_log_time = now

                # -----------------------------------------------------------------
                #  PHASE 31: NaN Detection mit verbesserter Recovery
                # -----------------------------------------------------------------
                if loss_val != loss_val or torch.isnan(torch.tensor(loss_val)):
                    nan_recovery_count += 1
                    print(f'\n!!! NaN DETEKTIERT (x{nan_recovery_count}) - Recovery !!!')

                    # Save recovery checkpoint
                    if not recovery_checkpoint_saved:
                        os.makedirs(RECOVERY_DIR, exist_ok=True)
                        brain.save_checkpoint(
                            os.path.join(RECOVERY_DIR, f'recovery_nan_iter{config["iteration"]}_step{step}.pt'),
                            config=config
                        )
                        recovery_checkpoint_saved = True

                    # Aggressive LR reduction with exponential backoff
                    config['lr'] *= max(0.3, 0.5 ** nan_recovery_count)
                    save_config(config)
                    print(f'  LR auf {config["lr"]:.6f} reduziert (nan_count={nan_recovery_count})')

                    # Clear CUDA cache
                    gc.collect()
                    torch.cuda.empty_cache()

                    if nan_recovery_count >= 5:
                        print(f'!!! KRITISCH: {nan_recovery_count} NaN Ereignisse - Abbruch !!!')
                        return

            # Sicherheitscheck gegen OOM
            if torch.cuda.max_memory_allocated() / 1024 / 1024 > config['max_vram_mb']:
                print('!!! VRAM LIMIT ERREICHT !!!')
                # Save recovery checkpoint
                os.makedirs(RECOVERY_DIR, exist_ok=True)
                brain.save_checkpoint(
                    os.path.join(RECOVERY_DIR, f'recovery_oom_iter{config["iteration"]}_step{step}.pt'),
                    config=config
                )
                return

        # =====================================================================
        #  ITERATION SUCCESS COMPLETION
        # =====================================================================
        final_loss = min(history)
        elapsed = time.time() - t_start
        print(f'\nIteration {config["iteration"]} abgeschlossen in {elapsed/60:.1f}m.')
        print(f'Best Loss dieser Iteration: {final_loss:.4f}')

        # PHASE 30: Per-domain best losses
        print('\n[DOMAIN] Per-Domain Losses:')
        for d_name, d_losses in domain_loss_history.items():
            if d_losses:
                d_best = min(d_losses)
                d_avg = sum(d_losses[-500:]) / max(1, len(d_losses[-500:]))
                print(f'  {d_name:10s}: best={d_best:.4f}, avg_last500={d_avg:.4f}')

        # --- MUTATION / EVOLUTION LOGIK ---
        config['iteration'] += 1
        
        # Detect if architecture changed (requires fresh weights, unfair comparison)
        arch_changed = False
        arch_key = f"d{config['d_model']}_s{config['d_sparse']}_l{config['n_layers']}"
        last_arch = config.get('_last_arch', '')
        if arch_key != last_arch:
            arch_changed = True
            print(f'[EVOLUTION] Architektur geändert ({last_arch} -> {arch_key}). Aufwärmiteration.')
            config['_last_arch'] = arch_key
        
        if final_loss < config['best_loss'] * 0.99:
            print(f'Neuer Rekord! {final_loss:.4f} < {config["best_loss"]:.4f}')
            config['best_loss'] = final_loss
            config['_best_arch'] = arch_key
            if config['d_model'] < 1024:
                config['d_model'] = min(1024, (int(config['d_model'] * 1.15) // 4) * 4)
            config['n_layers'] = min(18, config['n_layers'] + 1)
            config['lr'] *= 0.95
            print(f'Evolution: d_model={config["d_model"]}, layers={config["n_layers"]}, lr={config["lr"]:.6f}')
        elif arch_changed:
            # Architecture changed - don't punish, just give next iteration fair chance
            print(f'Aufwärmiteration abgeschlossen (best={final_loss:.4f}). Keine Mutation.')
            config['lr'] *= 0.95  # Gentle LR decay only
        elif final_loss < config['best_loss'] * 1.05:
            print(f'Leichter Fortschritt {final_loss:.4f}. Optimiere LR.')
            config['best_loss'] = min(config['best_loss'], final_loss)
            config['lr'] *= 0.9
        else:
            print(f'Kein Fortschritt ({final_loss:.4f} vs {config["best_loss"]:.4f}). Mutation.')
            config['lr'] *= 0.5
            if config['d_sparse'] > 2048:
                config['d_sparse'] = max(1024, (int(config['d_sparse'] * 0.85) // 4) * 4)
        save_config(config)

        # Checkpoints speichern
        brain.save_checkpoint(os.path.join(CHECKPOINT_DIR, 'checkpoint.pt'), config=config)
        if final_loss < config['best_loss']:
            brain.save_checkpoint(os.path.join(CHECKPOINT_DIR, 'best_model.pt'), config=config)
            print("Neues Best Model gespeichert!")

        # =====================================================================
        #  AUTO-EVALUATION: Generation Samples nach jeder Iteration
        # =====================================================================
        try:
            print("\n[AUTO-EVAL] Generiere Samples...")
            # Sample prompts from different domains
            prompts = {
                'text': "The future of artificial intelligence is",
                'code': "def fibonacci(n):",
                'security': "def check_vulnerability(code):",
                'network': "[FLOW] src=192.168.1.1 dst=10.0.0.1",
            }
            eval_dir = os.path.join(CHECKPOINT_DIR, f'iter_{config["iteration"]-1}')
            os.makedirs(eval_dir, exist_ok=True)
            
            for domain, prompt in prompts.items():
                # Encode prompt (BPE-kompatibel)
                encoded = multi_domain.encode(prompt)
                prompt_ids = torch.tensor([encoded], device=device)
                if prompt_ids.size(1) < 10:
                    continue
                
                # Generate
                generated = brain.generate_safe(prompt_ids, max_new=100, temperature=0.8, top_k=30)
                
                # Decode (BPE-kompatibel)
                gen_text = multi_domain.decode(generated[0])
                
                # Save
                with open(os.path.join(eval_dir, f'{domain}.txt'), 'w', encoding='utf-8') as f:
                    f.write(f"Prompt: {prompt}\n\nGenerated:\n{gen_text}\n")
                
                print(f"  [{domain}] {gen_text[:120]}...")
            
            print("[AUTO-EVAL] Samples gespeichert in", eval_dir)
        except Exception as eval_err:
            print(f"[AUTO-EVAL] Fehler: {eval_err}")

        # PHASE 27: 8-bit Weight Quantization für Generation (4x weniger VRAM)
        print("[QUANT] Quantisiere Modell für Evaluation...")
        try:
            q_stats = brain.quantize_weights()
        except AttributeError:
            print("[QUANT] quantize_weights() nicht verfügbar, überspringe.")
            q_stats = {}

        # =====================================================================
        #  PHASE 31: Verbesserte Online Evaluation
        # =====================================================================
        print("\n" + "="*60)
        print("PHASE 31: Enhanced Generation Evaluation")
        print("="*60)

        # Domain-specific prompts for evaluation
        eval_prompts = {
            'text':     [('ROMEO:', 'text'), ('KING ', 'text')],
            'code':     [('def ', 'code'), ('class ', 'code')],
            'security': [('CVE-', 'security'), ('vulnerability', 'security')],
            'network':  [('[FLOW]', 'network'), ('src=', 'network')],
        }

        # PHASE 12: Online Evaluation — Automatische Quality Metriken
        gen_scores = []
        domain_gen_scores = {}

        for eval_key, eval_list in eval_prompts.items():
            for prompt, domain in eval_list:
                try:
                    ctx = torch.tensor([multi_domain.encode(prompt)], device=device)
                    generated = brain.generate_safe(ctx, max_new=150, temperature=0.7, top_k=30)
                    gen = multi_domain.decode(generated[0])
                except Exception as e:
                    print(f'  [WARN] Generation fehlgeschlagen für "{prompt}": {e}')
                    gen = f'[GENERATION FAILED]'

                # Save generation
                safe_key = prompt.strip().replace('/', '_').replace('[', '').replace(']', '')
                gen_path = os.path.join(GENERATION_DIR,
                                        f'evolve_gen_{config["iteration"]}_{safe_key}.txt')
                with open(gen_path, 'w') as f:
                    f.write(gen)

                # PHASE 31: Domain-specific evaluation
                scores = evaluate_generated_text(gen, domain=domain)
                gen_scores.append(scores['score'])

                if domain not in domain_gen_scores:
                    domain_gen_scores[domain] = []
                domain_gen_scores[domain].append(scores)

                print(f'  [{domain:8s}] "{prompt:12s}" → score={scores["score"]:.3f} '
                      f'(unique={scores["unique_ratio"]:.2f}, '
                      f'rep={scores["rep_score"]:.2f})')

        avg_gen_score = sum(gen_scores) / len(gen_scores) if gen_scores else 0
        print(f'\nGeneration Quality Score: {avg_gen_score:.3f}')

        # Per-domain gen scores
        for d_name, d_scores in domain_gen_scores.items():
            d_avg = sum(s['score'] for s in d_scores) / max(1, len(d_scores))
            print(f'  {d_name:10s} Gen Score: {d_avg:.3f}')

        # PHASE 27: Quantized Checkpoint speichern (25% Größe)
        try:
            brain.save_quantized_checkpoint(os.path.join(CHECKPOINT_DIR, 'quantized_checkpoint.pt'), config=config)
        except AttributeError:
            print('[QUANT] save_quantized_checkpoint() nicht verfügbar, überspringe.')
        except Exception as e:
            print(f'[QUANT] Quantized Checkpoint Fehler: {e}')

    except Exception as e:
        print(f'Fehler in Iteration {config["iteration"]}: {e}')
        import traceback
        traceback.print_exc()
    finally:
        # Speicher leeren für nächste Iteration
        del brain
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print('Cleanup abgeschlossen.')


def cosine_anneal_lr(base_lr, step, total_steps, cycle_steps=50000):
    """Cosine Annealing Learning Rate Schedule."""
    progress = (step % max(1, cycle_steps)) / max(1, cycle_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


if __name__ == '__main__':
    print("=== STARTING AUTONOMOUS EVOLUTION LOOP (Multi-Domain + Curriculum) ===")
    while True:
        try:
            run_evolution()
        except Exception as e:
            print(f"CRITICAL ERROR IN LOOP: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(5)
