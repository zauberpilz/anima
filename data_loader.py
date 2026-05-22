"""
Erweiterter Multi-Domain Data Loader für große Korpora + Code + Security + Network.
v4: Multi-Domain Dataset Manager mit HuggingFace Integration.
"""
import torch
import os
import urllib.request
import json
import random
import re
from code_scraper import CodeScraper
from code_tokenizer import CodeTokenizer

DATA_DIR = '/home/anima/data'
DATASET_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
SHAKESPEARE_PATH = os.path.join(DATA_DIR, 'input.txt')

# ===== HUGGINGFACE DATASETS INTEGRATION =====
try:
    import datasets
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


# =====================================================================
#  UTILITY FUNCTIONS
# =====================================================================

def download_file(url, dest, max_retries=2):
    """Download mit Retry-Logik."""
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        print(f'OK: {dest} ({os.path.getsize(dest)/1e6:.1f}MB)')
        return True

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for attempt in range(max_retries + 1):
        try:
            print(f'Download {url} (Versuch {attempt+1})...')
            urllib.request.urlretrieve(url, dest)
            size = os.path.getsize(dest)
            if size > 1000:
                print(f'OK: {size/1e6:.1f}MB')
                return True
            os.remove(dest)
        except Exception as e:
            print(f'Fehler: {e}')
    return False


def get_character_mapping(text):
    """Erstellt Character-to-Index Mapping aus Text."""
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos, vocab_size


def load_shakespeare_fallback(max_chars=None):
    """Ultimativer Fallback: Shakespeare."""
    file_path = SHAKESPEARE_PATH
    if not os.path.exists(file_path):
        download_file(DATASET_URL, file_path)
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        if max_chars:
            text = text[:max_chars]
        print(f'[FALLBACK] Shakespeare geladen: {len(text):,} Chars')
        return text
    return None


def load_tinystories_fallback(max_chars=None):
    """Fallback auf TinyStories wenn HF nicht verfügbar."""
    if not HF_AVAILABLE:
        return None
    try:
        return download_hf_dataset('roneneldan/TinyStories', 'train', max_chars, 'text')
    except Exception:
        return None


# =====================================================================
#  HUGGINGFACE DOWNLOAD CORE
# =====================================================================

def download_hf_dataset(dataset_name, split='train', max_chars=None, text_column='text'):
    """
    Lädt ein Dataset von HuggingFace und bereitet es als Char-Level Text auf.
    Fallback auf Shakespeare wenn das Dataset nicht verfügbar ist.
    """
    if not HF_AVAILABLE:
        print('[DATA] HuggingFace datasets nicht installiert. Fallback auf Shakespeare.')
        return None

    try:
        print(f'[DATA] Lade HF Dataset: {dataset_name} ({split})...')
        ds = datasets.load_dataset(dataset_name, split=split, streaming=True)

        text = ''
        count = 0
        for example in ds:
            if text_column in example and example[text_column]:
                text += str(example[text_column]) + '\n'
                count += 1
                if max_chars and len(text) >= max_chars:
                    text = text[:max_chars]
                    break
                if count >= 100000:
                    break

        if len(text) < 1000:
            print(f'[DATA] Zu wenig Text von {dataset_name}: {len(text)} Chars')
            return None

        print(f'[DATA] HF Dataset geladen: {len(text):,} Chars aus {count} Beispielen')
        return text
    except Exception as e:
        print(f'[DATA] HF Dataset Fehler ({dataset_name}): {e}')
        return None


# =====================================================================
#  DOMAIN-SPECIFIC LOADERS
# =====================================================================

def load_code_datasets(max_chars=None):
    """Lädt Code-Domänen-Datasets mit Fallbacks."""
    if not HF_AVAILABLE:
        print('[CODE] HF nicht verfügbar, fallback auf CodeScraper')
        return _load_code_scraper_fallback(max_chars)

    datasets_config = [
        ('bigcode/the-stack-v2-smol', 'train', 'content'),
        ('codeparrot/codeparrot-clean', 'train', 'code'),
        ('sahil2801/CodeAlpaca-20k', 'train', None),  # special handling
    ]

    all_text = ''
    total_examples = 0

    for ds_name, split, col in datasets_config:
        try:
            if ds_name == 'sahil2801/CodeAlpaca-20k':
                text = _load_codealpaca(max_chars)
            else:
                text = download_hf_dataset(ds_name, split, max_chars, col)

            if text:
                marker = f'\n--- CODE:{ds_name.split("/")[-1]} ---\n'
                all_text += marker + text
                total_examples += 1
                print(f'[CODE] {ds_name}: {len(text):,} Chars geladen')

                if max_chars and len(all_text) >= max_chars:
                    all_text = all_text[:max_chars]
                    break
        except Exception as e:
            print(f'[CODE] Fehler bei {ds_name}: {e}')
            continue

    if len(all_text) < 1000:
        print('[CODE] Keine HF Code-Datasets verfügbar, fallback auf CodeScraper')
        return _load_code_scraper_fallback(max_chars)

    return all_text


def _load_codealpaca(max_chars=None):
    """Spezieller Loader für CodeAlpaca-20k (instruction+input+output)."""
    try:
        ds = datasets.load_dataset('sahil2801/CodeAlpaca-20k', split='train', streaming=True)
        text = ''
        for example in ds:
            instr = example.get('instruction', '')
            inp = example.get('input', '')
            out = example.get('output', '')
            combined = f'### Instruction:\n{instr}\n### Input:\n{inp}\n### Output:\n{out}\n\n'
            text += combined
            if max_chars and len(text) >= max_chars:
                text = text[:max_chars]
                break
        if len(text) > 1000:
            print(f'[CODE] CodeAlpaca-20k: {len(text):,} Chars geladen')
            return text
    except Exception as e:
        print(f'[CODE] CodeAlpaca-20k Fehler: {e}')
    return None


def _load_code_scraper_fallback(max_chars=None):
    """Fallback: CodeScraper wenn HF Datasets nicht laden."""
    try:
        scraper = CodeScraper(data_dir=DATA_DIR + '/code')
        code_files = list(scraper.data_dir.glob('*'))
        if len(code_files) < 5:
            print('[CODE] Starte Code-Scraper...')
            scraper.scrape_popular_repos()
            scraper.scrape_stackoverflow()
        return scraper.get_combined_dataset(max_chars=max_chars)
    except Exception as e:
        print(f'[CODE] CodeScraper Fallback Fehler: {e}')
        return None


def load_security_datasets(max_chars=None):
    """Lädt Security-Domänen-Datasets (Vulnerabilities, CVEs, Patches)."""
    if not HF_AVAILABLE:
        print('[SECURITY] HF nicht verfügbar, fallback auf Shakespeare')
        return load_shakespeare_fallback(max_chars)

    datasets_config = [
        ('starsofchance/PrimeVul', 'train', 'func'),
        ('starsofchance/CVEfixes_v1.0.8', 'train', None),  # special
        ('morinoppp/CyberSecurity-1M', 'train', 'text'),
        ('CIRCL/vulnerability-cwe-patch', 'train', 'patch'),
    ]

    all_text = ''
    total_examples = 0

    for ds_name, split, col in datasets_config:
        try:
            if ds_name == 'starsofchance/CVEfixes_v1.0.8':
                text = _load_cvefixes(max_chars)
            elif ds_name == 'morinoppp/CyberSecurity-1M':
                text = download_hf_dataset(ds_name, split, max_chars, col)
            elif ds_name == 'starsofchance/PrimeVul':
                text = _load_primevul(max_chars)
            elif ds_name == 'CIRCL/vulnerability-cwe-patch':
                text = _load_cwe_patch(max_chars)
            else:
                text = download_hf_dataset(ds_name, split, max_chars, col)

            if text:
                marker = f'\n--- SECURITY:{ds_name.split("/")[-1]} ---\n'
                all_text += marker + text
                total_examples += 1
                print(f'[SECURITY] {ds_name}: {len(text):,} Chars geladen')

                if max_chars and len(all_text) >= max_chars:
                    all_text = all_text[:max_chars]
                    break
        except Exception as e:
            print(f'[SECURITY] Fehler bei {ds_name}: {e}')
            continue

    if len(all_text) < 1000:
        print('[SECURITY] Keine Security-Datasets verfügbar, fallback auf Shakespeare')
        return load_shakespeare_fallback(max_chars)

    return all_text


def _load_primevul(max_chars=None):
    """Lädt PrimeVul (vulnerable code) und formatiert es."""
    try:
        ds = datasets.load_dataset('starsofchance/PrimeVul', split='train', streaming=True)
        text = ''
        for example in ds:
            func = example.get('func', '')
            cve = example.get('CVE', '')
            cwe = example.get('CWE', '')
            severity = example.get('severity', '')
            if func:
                header = f'[CVE: {cve}] [CWE: {cwe}] [SEVERITY: {severity}]\n'
                text += header + func + '\n\n'
                if max_chars and len(text) >= max_chars:
                    text = text[:max_chars]
                    break
        if len(text) > 1000:
            return text
    except Exception as e:
        print(f'[SECURITY] PrimeVul Fehler: {e}')
    return None


def _load_cvefixes(max_chars=None):
    """Lädt CVEfixes und konvertiert in unified Format."""
    try:
        ds = datasets.load_dataset('starsofchance/CVEfixes_v1.0.8', split='train', streaming=True)
        text = ''
        for example in ds:
            cve_id = example.get('cve_id', '')
            cwe = example.get('cwe', '')
            severity = example.get('severity', '')
            code_before = example.get('code_before', '')
            code_after = example.get('code_after', '')

            if code_before and code_after:
                entry = f'[{cve_id}] [CWE: {cwe}] [SEVERITY: {severity}]\n'
                entry += f'BEFORE: {code_before}\n'
                entry += f'AFTER: {code_after}\n\n'
                text += entry
                if max_chars and len(text) >= max_chars:
                    text = text[:max_chars]
                    break
        if len(text) > 1000:
            return text
    except Exception as e:
        print(f'[SECURITY] CVEfixes Fehler: {e}')
    return None


def _load_cwe_patch(max_chars=None):
    """Lädt CIRCL vulnerability-cwe-patch Dataset."""
    try:
        ds = datasets.load_dataset('CIRCL/vulnerability-cwe-patch', split='train', streaming=True)
        text = ''
        for example in ds:
            patch = example.get('patch', '')
            cwe = example.get('cwe', '')
            summary = example.get('summary', '')
            if patch:
                entry = f'[CWE: {cwe}] {summary}\nPATCH: {patch}\n\n'
                text += entry
                if max_chars and len(text) >= max_chars:
                    text = text[:max_chars]
                    break
        if len(text) > 1000:
            return text
    except Exception as e:
        print(f'[SECURITY] CWE-Patch Fehler: {e}')
    return None


def load_network_datasets(max_chars=None):
    """Lädt Network-Domänen-Datasets (Traffic, Anomalien)."""
    if not HF_AVAILABLE:
        print('[NETWORK] HF nicht verfügbar, fallback auf Shakespeare')
        return load_shakespeare_fallback(max_chars)

    datasets_config = [
        ('bvsam/cic-ids-2017', 'train', None),  # special
        ('Mireu-Lab/UNSW-NB15', 'train', None),  # special
    ]

    all_text = ''

    for ds_name, split, col in datasets_config:
        try:
            if ds_name == 'bvsam/cic-ids-2017':
                text = _load_cicids2017(max_chars)
            elif ds_name == 'Mireu-Lab/UNSW-NB15':
                text = _load_unsw_nb15(max_chars)
            else:
                text = download_hf_dataset(ds_name, split, max_chars, col)

            if text:
                marker = f'\n--- NETWORK:{ds_name.split("/")[-1]} ---\n'
                all_text += marker + text
                print(f'[NETWORK] {ds_name}: {len(text):,} Chars geladen')

                if max_chars and len(all_text) >= max_chars:
                    all_text = all_text[:max_chars]
                    break
        except Exception as e:
            print(f'[NETWORK] Fehler bei {ds_name}: {e}')
            continue

    if len(all_text) < 1000:
        print('[NETWORK] Keine Network-Datasets verfügbar, fallback auf Shakespeare')
        return load_shakespeare_fallback(max_chars)

    return all_text


def _load_cicids2017(max_chars=None):
    """Lädt CIC-IDS-2017 und konvertiert in Text-Format."""
    try:
        ds = datasets.load_dataset('bvsam/cic-ids-2017', split='train', streaming=True)
    except Exception:
        try:
            ds = datasets.load_dataset('bvsam/cic-ids-2017', split='train', streaming=True, trust_remote_code=True)
        except Exception as e:
            print(f'[NETWORK] CIC-IDS-2017 Fehler: {e}')
            return None

    text = ''
    count = 0
    try:
        for example in ds:
            # Versuche, Flow/Packet-ähnliche Felder zu erkennen
            src_ip = example.get('Source IP', example.get('src_ip', example.get('src', '')))
            dst_ip = example.get('Destination IP', example.get('dst_ip', example.get('dst', '')))
            src_port = example.get('Source Port', example.get('src_port', example.get('sport', '')))
            dst_port = example.get('Destination Port', example.get('dst_port', example.get('dport', '')))
            protocol = example.get('Protocol', example.get('proto', ''))
            label = example.get('Label', example.get('label', example.get('attack', '')))
            bytes_val = example.get('Total Length of Fwd Packets', example.get('totlen', ''))
            pkts = example.get('Total Fwd Packets', example.get('pkts', ''))
            duration = example.get('Flow Duration', example.get('duration', ''))

            # Formatiere als strukturierten Text
            entry = f'[FLOW] src={src_ip}:{src_port} -> dst={dst_ip}:{dst_port} '
            entry += f'proto={protocol} bytes={bytes_val} pkts={pkts} duration={duration} '
            entry += f'label={label}\n'
            text += entry
            count += 1

            if max_chars and len(text) >= max_chars:
                text = text[:max_chars]
                break
            if count >= 50000:
                break

        if len(text) > 1000:
            print(f'[NETWORK] CIC-IDS-2017: {len(text):,} Chars aus {count} Einträgen')
            return text
    except Exception as e:
        print(f'[NETWORK] CIC-IDS-2017 Verarbeitungsfehler: {e}')

    return None


def _load_unsw_nb15(max_chars=None):
    """Lädt UNSW-NB15 und konvertiert in Text-Format."""
    try:
        ds = datasets.load_dataset('Mireu-Lab/UNSW-NB15', split='train', streaming=True)
    except Exception:
        try:
            ds = datasets.load_dataset('Mireu-Lab/UNSW-NB15', split='train', streaming=True, trust_remote_code=True)
        except Exception as e:
            print(f'[NETWORK] UNSW-NB15 Fehler: {e}')
            return None

    text = ''
    count = 0
    try:
        for example in ds:
            src_ip = example.get('srcip', example.get('src_ip', ''))
            dst_ip = example.get('dstip', example.get('dst_ip', ''))
            src_port = example.get('sport', example.get('src_port', ''))
            dst_port = example.get('dsport', example.get('dst_port', ''))
            proto = example.get('proto', example.get('Protocol', ''))
            dur = example.get('dur', example.get('duration', ''))
            bytes_val = example.get('bytes', example.get('spkts', ''))
            label = example.get('label', example.get('attack_cat', ''))
            is_anomaly = example.get('is_anomaly', example.get('Label', ''))

            entry = f'[FLOW] src={src_ip}:{src_port} -> dst={dst_ip}:{dst_port} '
            entry += f'proto={proto} duration={dur}s bytes={bytes_val} '
            entry += f'anomaly={is_anomaly} label={label}\n'
            text += entry
            count += 1

            if max_chars and len(text) >= max_chars:
                text = text[:max_chars]
                break
            if count >= 50000:
                break

        if len(text) > 1000:
            print(f'[NETWORK] UNSW-NB15: {len(text):,} Chars aus {count} Einträgen')
            return text
    except Exception as e:
        print(f'[NETWORK] UNSW-NB15 Verarbeitungsfehler: {e}')

    return None


def load_text_datasets(max_chars=None):
    """Lädt Text-Domänen-Datasets (TinyStories, OpenWebText, FineWeb)."""
    if not HF_AVAILABLE:
        print('[TEXT] HF nicht verfügbar, fallback auf Shakespeare')
        return load_shakespeare_fallback(max_chars)

    datasets_config = [
        ('roneneldan/TinyStories', 'train', 'text'),
        ('stas/openwebtext-10k', 'train', 'text'),
        ('HuggingFaceFW/fineweb-edu', 'train', 'text'),
    ]

    all_text = ''

    for ds_name, split, col in datasets_config:
        try:
            text = download_hf_dataset(ds_name, split, max_chars, col)
            if text:
                marker = f'\n--- TEXT:{ds_name.split("/")[-1]} ---\n'
                all_text += marker + text
                print(f'[TEXT] {ds_name}: {len(text):,} Chars geladen')

                if max_chars and len(all_text) >= max_chars:
                    all_text = all_text[:max_chars]
                    break
        except Exception as e:
            print(f'[TEXT] Fehler bei {ds_name}: {e}')
            continue

    if len(all_text) < 1000:
        print('[TEXT] Keine Text-Datasets verfügbar, fallback auf Shakespeare')
        return load_shakespeare_fallback(max_chars)

    return all_text


# =====================================================================
#  MULTI-DOMAIN DATASET MANAGER
# =====================================================================

class MultiDomainDataset:
    """
    Multi-Domain Dataset Manager — Code + Security + Network + Text.
    Verwaltet separate Domänen mit Gewichtung und shared Character Mapping.
    """
    def __init__(self, max_chars_per_domain=5000000):
        self.max_chars_per_domain = max_chars_per_domain
        self.domains = {
            'code': {'weight': 0.3, 'enabled': True, 'data': None, 'source': None},
            'security': {'weight': 0.2, 'enabled': True, 'data': None, 'source': None},
            'network': {'weight': 0.1, 'enabled': True, 'data': None, 'source': None},
            'text': {'weight': 0.4, 'enabled': True, 'data': None, 'source': None},
        }
        self.stoi = {}
        self.itos = {}
        self.vocab_size = 0
        self.data = None
        self.domain_ranges = {}
        self._loaded = False

    # -----------------------------------------------------------------
    #  LOADING
    # -----------------------------------------------------------------

    def load_all(self, max_chars_per_domain=None):
        """Lädt alle aktiven Domänen."""
        if max_chars_per_domain is not None:
            self.max_chars_per_domain = max_chars_per_domain

        print('=' * 60)
        print('[MULTI] Multi-Domain Dataset Loader')
        print('=' * 60)

        domain_loaders = {
            'code': load_code_datasets,
            'security': load_security_datasets,
            'network': load_network_datasets,
            'text': load_text_datasets,
        }

        all_texts = {}
        for domain_name, loader_fn in domain_loaders.items():
            if not self.domains[domain_name]['enabled']:
                print(f'[MULTI] Domäne {domain_name} ist deaktiviert, überspringe.')
                continue

            print(f'\n[MULTI] Lade Domäne: {domain_name} ...')
            text = loader_fn(max_chars=self.max_chars_per_domain)
            if text and len(text) > 1000:
                all_texts[domain_name] = text
                self.domains[domain_name]['data'] = text
                print(f'[MULTI] ✓ {domain_name}: {len(text):,} Chars')
            else:
                print(f'[MULTI] ✗ {domain_name}: Keine Daten, fallback auf TinyStories')
                fallback = load_tinystories_fallback(self.max_chars_per_domain)
                if fallback:
                    all_texts[domain_name] = fallback
                    self.domains[domain_name]['data'] = fallback
                    self.domains[domain_name]['source'] = 'fallback'
                else:
                    sh = load_shakespeare_fallback(self.max_chars_per_domain)
                    if sh:
                        all_texts[domain_name] = sh
                        self.domains[domain_name]['data'] = sh
                        self.domains[domain_name]['source'] = 'shakespeare_fallback'

        if not all_texts:
            raise RuntimeError('[MULTI] KEINE DATEN VERFUEGBAR!')

        # Shared Character Mapping über alle Domänen
        combined_text_for_vocab = ''.join(all_texts.values())
        self.stoi, self.itos, self.vocab_size = get_character_mapping(combined_text_for_vocab)

        # Tensor-Konvertierung mit shared mapping
        all_tensors = []
        current_offset = 0
        for domain_name in ['code', 'security', 'network', 'text']:
            if domain_name in all_texts:
                domain_text = all_texts[domain_name]
                domain_tensor = torch.tensor(
                    [self.stoi.get(c, 0) for c in domain_text],
                    dtype=torch.long
                )
                start_idx = current_offset
                end_idx = current_offset + len(domain_tensor)
                self.domain_ranges[domain_name] = (start_idx, end_idx)
                all_tensors.append(domain_tensor)
                current_offset += len(domain_tensor)
                weight = self.domains[domain_name]['weight']
                print(f'[MULTI]   {domain_name:10s}: {len(domain_tensor):>10,} Tokens (weight={weight:.1f})')
            else:
                self.domain_ranges[domain_name] = (current_offset, current_offset)

        self.data = torch.cat(all_tensors) if all_tensors else torch.tensor([], dtype=torch.long)
        self._loaded = True

        print(f'\n[MULTI] GESAMT: {len(self.data):,} Tokens, Vocab: {self.vocab_size}')
        print(f'[MULTI] Domain-Ranges: {self.domain_ranges}')
        print('=' * 60)

        return self

    def load_domain(self, domain_name):
        """Lädt eine einzelne Domäne mit Fallbacks."""
        loaders = {
            'code': load_code_datasets,
            'security': load_security_datasets,
            'network': load_network_datasets,
            'text': load_text_datasets,
        }

        if domain_name not in loaders:
            raise ValueError(f'Unbekannte Domäne: {domain_name}. Valid: {list(loaders.keys())}')

        text = loaders[domain_name](max_chars=self.max_chars_per_domain)
        if not text or len(text) < 1000:
            print(f'[MULTI] Fallback für {domain_name} auf TinyStories')
            text = load_tinystories_fallback(self.max_chars_per_domain)
        if not text or len(text) < 1000:
            print(f'[MULTI] Fallback für {domain_name} auf Shakespeare')
            text = load_shakespeare_fallback(self.max_chars_per_domain)

        if text:
            self.domains[domain_name]['data'] = text

            # Stelle sicher dass das mapping existiert
            if not self.stoi:
                self.stoi, self.itos, self.vocab_size = get_character_mapping(text)

            domain_tensor = torch.tensor(
                [self.stoi.get(c, 0) for c in text],
                dtype=torch.long
            )
            return domain_tensor

        return None

    # -----------------------------------------------------------------
    #  BATCHING
    # -----------------------------------------------------------------

    def get_batch(self, domain, batch_size, seq_len, device='cpu'):
        """Holt einen Batch aus einer spezifischen Domäne."""
        if domain not in self.domain_ranges:
            raise ValueError(f'Unbekannte Domäne: {domain}')

        start, end = self.domain_ranges[domain]
        domain_len = end - start

        if domain_len < seq_len + 1:
            print(f'[BATCH] {domain} zu klein ({domain_len}), verwende gesamte Domäne')
            # Wiederhole Daten wenn nötig
            domain_data = self.data[start:end]
            if len(domain_data) < seq_len + 1:
                repeats = (seq_len + 1) // len(domain_data) + 1
                domain_data = domain_data.repeat(repeats)
            domain_data = domain_data[:seq_len + 1]
            ix = 0
        else:
            domain_data = self.data[start:end]
            ix = torch.randint(len(domain_data) - seq_len, (batch_size,))

        if domain_len >= seq_len + 1:
            x = torch.stack([domain_data[i:i+seq_len] for i in ix])
            y = torch.stack([domain_data[i+1:i+seq_len+1] for i in ix])
        else:
            x = domain_data[:seq_len].unsqueeze(0).expand(batch_size, -1)
            y = domain_data[1:seq_len+1].unsqueeze(0).expand(batch_size, -1)

        return x.to(device), y.to(device)

    def get_mixed_batch(self, batch_size, seq_len, device='cpu'):
        """
        Erzeugt einen Batch gemischt nach Domänen-Gewichtung.
        Jede Domäne liefert proportional zu ihrem weight.
        """
        if not self._loaded or self.data is None:
            raise RuntimeError('[MULTI]Dataset noch nicht geladen. Rufe load_all() auf.')

        # Berechne Samples pro Domäne basierend auf Gewichtung
        total_weight = sum(
            d['weight'] for d in self.domains.values()
            if d['enabled'] and d['data'] is not None
        )

        if total_weight == 0:
            raise RuntimeError('[MULTI] Keine aktiven Domänen mit Daten!')

        samples_per_domain = {}
        remaining = batch_size
        active_domains = [
            name for name, d in self.domains.items()
            if d['enabled'] and d['data'] is not None
        ]

        # Erste Zuteilung nach Gewicht
        for i, name in enumerate(active_domains):
            if i == len(active_domains) - 1:
                samples_per_domain[name] = remaining
            else:
                n = max(1, int(batch_size * self.domains[name]['weight'] / total_weight))
                samples_per_domain[name] = n
                remaining -= n

        # Sammle Batches aus jeder Domäne
        x_parts = []
        y_parts = []

        for name, n_samples in samples_per_domain.items():
            if n_samples <= 0:
                continue
            try:
                x_domain, y_domain = self.get_batch(name, n_samples, seq_len, device)
                x_parts.append(x_domain)
                y_parts.append(y_domain)
            except Exception as e:
                print(f'[BATCH] Fehler bei Domäne {name}: {e}')
                continue

        if not x_parts:
            # Fallback: erste verfügbare Domäne
            fallback_domain = active_domains[0] if active_domains else 'text'
            print(f'[BATCH] Fallback auf {fallback_domain}')
            return self.get_batch(fallback_domain, batch_size, seq_len, device)

        x = torch.cat(x_parts, dim=0)
        y = torch.cat(y_parts, dim=0)

        # Shuffle innerhalb des Batches
        perm = torch.randperm(x.size(0))
        x = x[perm]
        y = y[perm]

        return x, y

    def get_domain_weights(self):
        """Gibt die aktuellen Gewichte der Domänen zurück."""
        return {name: d['weight'] for name, d in self.domains.items()}

    def set_domain_weight(self, domain, weight):
        """Setzt das Gewicht einer Domäne (0.0 - 1.0)."""
        if domain in self.domains:
            self.domains[domain]['weight'] = max(0.0, min(1.0, weight))
            print(f'[MULTI] Gewicht {domain} -> {self.domains[domain]["weight"]:.2f}')
        else:
            raise ValueError(f'Unbekannte Domäne: {domain}')

    def enable_domain(self, domain, enabled=True):
        """Aktiviert oder deaktiviert eine Domäne."""
        if domain in self.domains:
            self.domains[domain]['enabled'] = enabled
            print(f'[MULTI] Domäne {domain} {"aktiviert" if enabled else "deaktiviert"}')
        else:
            raise ValueError(f'Unbekannte Domäne: {domain}')

    # -----------------------------------------------------------------
    #  STATISTICS
    # -----------------------------------------------------------------

    def get_vocab_stats(self):
        """Gibt Vokabular-Statistiken aus."""
        if not self.stoi:
            print('[STATS] Kein Vokabular geladen.')
            return

        print('\n' + '=' * 50)
        print('VOCABULARY STATISTICS')
        print('=' * 50)
        print(f'Vocab Size: {self.vocab_size}')
        print(f'Total Tokens: {len(self.data):,}' if self.data is not None else 'Total Tokens: N/A')

        for name in ['code', 'security', 'network', 'text']:
            d = self.domains[name]
            data_len = 0
            start, end = self.domain_ranges.get(name, (0, 0))
            if self.data is not None and end > start:
                data_len = end - start

            status = '✓' if d['data'] is not None else '✗'
            source = d.get('source', 'hf')
            print(f'  {name:10s} [{status}] weight={d["weight"]:.1f} '
                  f'tokens={data_len:>10,} source={source}')

        if self.stoi:
            # Top 10 häufigste Zeichen
            if self.data is not None and len(self.data) > 0:
                counts = torch.bincount(self.data, minlength=self.vocab_size)
                top_chars = counts.topk(min(10, self.vocab_size))
                print('\nTop 10 Characters:')
                for i in range(top_chars.indices.size(0)):
                    idx = top_chars.indices[i].item()
                    char = self.itos.get(idx, repr(idx))
                    count = top_chars.values[i].item()
                    pct = 100.0 * count / len(self.data)
                    print(f'  {i+1}. {repr(char):>6} -> {count:>8,} ({pct:.2f}%)')

        print('=' * 50)

    def get_domain_info(self):
        """Gibt strukturierte Infos über alle Domänen zurück."""
        info = {}
        for name in ['code', 'security', 'network', 'text']:
            d = self.domains[name]
            start, end = self.domain_ranges.get(name, (0, 0))
            info[name] = {
                'weight': d['weight'],
                'enabled': d['enabled'],
                'has_data': d['data'] is not None,
                'tokens': (end - start) if self.data is not None else 0,
                'source': d.get('source', 'hf'),
            }
        info['total_tokens'] = len(self.data) if self.data is not None else 0
        info['vocab_size'] = self.vocab_size
        return info


# =====================================================================
#  CONVENIENCE FUNCTIONS (backward compatible)
# =====================================================================

def get_multi_domain_data(max_chars=3000000):
    """
    Lädt und kombiniert alle Domänen in ein unified Dataset.
    Gibt MultiDomainDataset-Instanz zurück.
    """
    loader = MultiDomainDataset(max_chars_per_domain=max_chars)
    loader.load_all()
    return loader


def get_large_dataset(max_chars=None):
    """
    Lädt großen Text-Korpus via HuggingFace Datasets (Multi-Domain).
    Multi-Source Fallback: TinyStories -> Shakespeare -> ...
    Fällt zurück auf Shakespeare (input.txt) wenn alles fehlschlägt.

    Returns: (data_tensor, stoi, itos, vocab_size)
    """
    try:
        loader = get_multi_domain_data(max_chars=max_chars or 3000000)
        if loader is not None and loader.data is not None and len(loader.data) > 1000:
            print(f'[DATA] Multi-Domain Dataset: {len(loader.data):,} Tokens, '
                  f'Vocab: {loader.vocab_size}')
            return loader.data, loader.stoi, loader.itos, loader.vocab_size
    except Exception as e:
        print(f'[DATA] Multi-Domain Fehler, Fallback auf single-source: {e}')

    # Legacy Fallback
    os.makedirs(DATA_DIR, exist_ok=True)

    HF_DATASETS = [
        ('roneneldan/TinyStories', 'train', 'text'),
        ('shakespeare', 'train', 'text'),
        ('tiny_shakespeare', 'train', 'text'),
        ('stas/openwebtext-10k', 'train', 'text'),
        ('bigcode/the-stack-smol', 'train', 'content'),
        ('roneneldan/TinyStoriesV2', 'train', 'text'),
        ('HuggingFaceFW/fineweb-edu', 'train', 'text'),
    ]

    text = None
    source_name = None

    for dataset_name, split, col in HF_DATASETS:
        result = download_hf_dataset(dataset_name, split, max_chars, col)
        if result:
            text = result
            source_name = dataset_name
            break

    if text is None:
        print('[DATA] HF Datasets nicht verfügbar. Versuche Raw URLs...')
        RAW_URLS = [
            (DATASET_URL, 'input.txt'),
            ("https://raw.githubusercontent.com/brunoklein99/deep-learning-notes/master/shakespeare.txt", 'shakespeare2.txt'),
            ("https://gist.githubusercontent.com/phillipj/4944029/raw/75ba2243dd5ec2875febbfa7c2bb1e2e5da5ef3e/shakespeare.txt", 'shakespeare3.txt'),
        ]

        for url, fname in RAW_URLS:
            file_path = os.path.join(DATA_DIR, fname)
            if download_file(url, file_path):
                try:
                    for enc in ['utf-8', 'latin-1']:
                        try:
                            with open(file_path, 'r', encoding=enc) as f:
                                text = f.read()
                            source_name = url
                            break
                        except UnicodeDecodeError:
                            continue
                    if text:
                        break
                except Exception:
                    continue

    if text is None:
        file_path = os.path.join(DATA_DIR, 'input.txt')
        if os.path.exists(file_path):
            print(f'[DATA] Fallback auf vorhandene Datei: {file_path}')
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            source_name = 'shakespeare_fallback'

    if text is None:
        raise RuntimeError("[DATA] KEINE DATASETS VERFUEGBAR! Training kann nicht starten.")

    if max_chars:
        text = text[:max_chars]

    if len(text) < 500000:
        multiplier = min(8, 500000 // max(1, len(text)))
        if multiplier > 1:
            print(f'[DATA] Dataset klein ({len(text):,} Chars). Vervielfache x{multiplier}')
            text = text * multiplier

    stoi, itos, vocab_size = get_character_mapping(text)
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    print(f'[DATA] Dataset: {len(data):,} Tokens, Vocab: {vocab_size}, Quelle: {source_name}')

    return data, stoi, itos, vocab_size


def get_code_dataset(max_chars=None, use_code_tokenizer=False):
    """
    Lädt Code-Daten von GitHub/StackOverflow.
    Optional mit speziellem Code-Tokenizer.
    Verwendet jetzt Multi-Domain Code-Loader.
    """
    text = load_code_datasets(max_chars=max_chars)
    if text is None:
        raise RuntimeError('[CODE] Keine Code-Daten verfuegbar.')

    if use_code_tokenizer:
        tokenizer = CodeTokenizer()
        data = tokenizer.encode(text)
        stoi = tokenizer.stoi
        itos = tokenizer.itos
        vocab_size = tokenizer.vocab_size
    else:
        stoi, itos, vocab_size = get_character_mapping(text)
        data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    print(f'[CODE] Dataset: {len(data):,} Tokens, Vocab: {vocab_size}')
    return data, stoi, itos, vocab_size


def get_mixed_dataset(max_chars=None, code_ratio=0.3):
    """
    Kombiniert Code-Daten mit Text-Daten.
    code_ratio: Anteil Code-Daten (0.0-1.0)
    Verwendet jetzt Multi-Domain Mix.
    """
    if max_chars is None:
        max_chars = 3000000

    try:
        loader = MultiDomainDataset(max_chars_per_domain=max_chars)
        loader.domains['code']['weight'] = code_ratio
        loader.domains['text']['weight'] = 1.0 - code_ratio
        loader.domains['security']['enabled'] = False
        loader.domains['network']['enabled'] = False
        loader.load_all()

        print(f'[DATA] Mixed Dataset: {len(loader.data):,} Tokens, Vocab: {loader.vocab_size}')
        for name in ['code', 'text']:
            start, end = loader.domain_ranges.get(name, (0,0))
            print(f'  {name}: {end-start:,} Tokens')

        return loader.data, loader.stoi, loader.itos, loader.vocab_size
    except Exception as e:
        print(f'[DATA] Multi-Domain Mixed Fehler, fallback: {e}')

    # Legacy Fallback
    text_data, text_stoi, text_itos, text_vocab = get_large_dataset(
        max_chars=int(max_chars * (1 - code_ratio))
    )

    code_data, code_stoi, code_itos, code_vocab = get_code_dataset(
        max_chars=int(max_chars * code_ratio)
    )

    combined_stoi = {**text_stoi, **code_stoi}
    combined_itos = {**text_itos, **code_itos}
    vocab_size = len(combined_stoi)

    combined_data = torch.cat([text_data, code_data])

    print(f'[DATA] Mixed Dataset: {len(combined_data):,} Tokens, Vocab: {vocab_size}')
    print(f'  Text: {len(text_data):,} Tokens, Code: {len(code_data):,} Tokens')

    return combined_data, combined_stoi, combined_itos, vocab_size
