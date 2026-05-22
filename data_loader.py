"""
Erweiterter Data Loader für große Korpora + Code-Daten.
v3: HuggingFace Datasets Integration für tausende Trainingsquellen.
"""
import torch
import os
import urllib.request
import json
import random
from code_scraper import CodeScraper
from code_tokenizer import CodeTokenizer

DATA_DIR = '/home/anima/data'
DATASET_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

# ===== HUGGINGFACE DATASETS INTEGRATION =====
try:
    import datasets
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

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
                if count >= 100000:  # Sicherheitslimit
                    break
        
        if len(text) < 1000:
            print(f'[DATA] Zu wenig Text von {dataset_name}: {len(text)} Chars')
            return None
            
        print(f'[DATA] HF Dataset geladen: {len(text):,} Chars aus {count} Beispielen')
        return text
    except Exception as e:
        print(f'[DATA] HF Dataset Fehler ({dataset_name}): {e}')
        return None

def get_character_mapping(text):
    """Erstellt Character-to-Index Mapping aus Text."""
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos, vocab_size

def get_code_dataset(max_chars=None, use_code_tokenizer=False):
    """
    Lädt Code-Daten von GitHub/StackOverflow.
    Optional mit speziellem Code-Tokenizer.
    """
    scraper = CodeScraper(data_dir=DATA_DIR + '/code')
    
    code_files = list(scraper.data_dir.glob('*'))
    if len(code_files) < 5:
        print("[DATA] Starte Code-Scraper...")
        scraper.scrape_popular_repos()
        scraper.scrape_stackoverflow()
    
    return scraper.get_combined_dataset(max_chars=max_chars)

def get_mixed_dataset(max_chars=None, code_ratio=0.3):
    """
    Kombiniert Code-Daten mit Text-Daten.
    code_ratio: Anteil Code-Daten (0.0-1.0)
    """
    text_data, text_stoi, text_itos, text_vocab = get_large_dataset(
        max_chars=int(max_chars * (1 - code_ratio)) if max_chars else None
    )
    
    code_data, code_stoi, code_itos, code_vocab = get_code_dataset(
        max_chars=int(max_chars * code_ratio) if max_chars else None
    )
    
    combined_stoi = {**text_stoi, **code_stoi}
    combined_itos = {**text_itos, **code_itos}
    vocab_size = len(combined_stoi)
    
    combined_data = torch.cat([text_data, code_data])
    
    print(f'[DATA] Mixed Dataset: {len(combined_data):,} Tokens, Vocab: {vocab_size}')
    print(f'  Text: {len(text_data):,} Tokens, Code: {len(code_data):,} Tokens')
    
    return combined_data, combined_stoi, combined_itos, vocab_size

def get_large_dataset(max_chars=None):
    """
    Lädt großen Text-Korpus via HuggingFace Datasets (10 Quellen).
    Multi-Source Fallback: TinyStories → Shakespeare → OpenWebText → Wiki → ...
    Fällt zurück auf Shakespeare (input.txt) wenn alles fehlschlägt.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # === Phase 1: HuggingFace Datasets (streaming, kein Download nötig) ===
    HF_DATASETS = [
        # Kindergeschichten (einfach, gut für Foundation)
        ('roneneldan/TinyStories', 'train', 'text'),
        # Shakespeare
        ('shakespeare', 'train', 'text'),
        ('tiny_shakespeare', 'train', 'text'),
        # OpenWebText (qualitativ hochwertige Web-Texte)
        ('stas/openwebtext-10k', 'train', 'text'),
        # Code (für Hybrid-Training)
        ('bigcode/the-stack-smol', 'train', 'content'),
        # Geschichten
        ('roneneldan/TinyStoriesV2', 'train', 'text'),
        # FineWeb (hochwertig gefiltertes Web)
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
    
    # === Phase 2: Raw URL Fallback ===
    if text is None:
        print('[DATA] HF Datasets nicht verfügbar. Versuche Raw URLs...')
        RAW_URLS = [
            ("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", 'input.txt'),
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
                except:
                    continue
    
    # === Phase 3: Ultimate Fallback (vorhandene Shakespeare-Datei) ===
    if text is None:
        file_path = os.path.join(DATA_DIR, 'input.txt')
        if os.path.exists(file_path):
            print(f'[DATA] Fallback auf vorhandene Datei: {file_path}')
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            source_name = 'shakespeare_fallback'
    
    if text is None:
        raise RuntimeError("[DATA] KEINE DATASETS VERFÜGBAR! Training kann nicht starten.")
    
    if max_chars:
        text = text[:max_chars]
    
    # Data Augmentation: Dataset vervielfachen wenn sehr klein
    if len(text) < 500000:
        multiplier = min(8, 500000 // max(1, len(text)))
        if multiplier > 1:
            print(f'[DATA] Dataset klein ({len(text):,} Chars). Vervielfache x{multiplier}')
            text = text * multiplier
    
    stoi, itos, vocab_size = get_character_mapping(text)
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    
    print(f'[DATA] Dataset: {len(data):,} Tokens, Vocab: {vocab_size}, Quelle: {source_name}')
    
    return data, stoi, itos, vocab_size
