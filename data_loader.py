"""Erweiterter Data Loader für größere Korpora + Code-Daten."""
import torch
import os
import urllib.request
from code_scraper import CodeScraper
from code_tokenizer import CodeTokenizer

DATA_DIR = '/home/anima/data'
DATASET_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

# TinyStories: Mehrere Mirrors
TINYSTORIES_URLS = [
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_train.txt",
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.txt",
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2_train.txt",
]

# Fallback-Datasets
SHAKESPEARE_FULL_URL = "https://raw.githubusercontent.com/Phylliade/ShakespeareDataset/master/shakespeare.txt"
PG19_URL = "https://raw.githubusercontent.com/google-research-datasets/pg19/master/train_set.txt"

def download_file(url, dest, max_retries=2):
    """Download mit Retry-Logik."""
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        print(f'Datei existiert: {dest} ({os.path.getsize(dest)/1e6:.1f}MB)')
        return True
    
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for attempt in range(max_retries + 1):
        try:
            print(f'Download {url} -> {dest} (Versuch {attempt+1})')
            urllib.request.urlretrieve(url, dest)
            size = os.path.getsize(dest)
            if size > 1000:
                print(f'OK: {size/1e6:.1f}MB')
                return True
            os.remove(dest)
        except Exception as e:
            print(f'Fehler: {e}')
    return False

def get_code_dataset(max_chars=None, use_code_tokenizer=False):
    """
    Lädt Code-Daten von GitHub/StackOverflow.
    Optional mit speziellem Code-Tokenizer.
    """
    scraper = CodeScraper(data_dir=DATA_DIR + '/code')
    
    # Prüfen ob bereits Code-Daten vorhanden
    code_files = list(scraper.data_dir.glob('*'))
    if len(code_files) < 5:
        print("[DATA] Zu wenige Code-Dateien. Starte Scraper...")
        scraper.scrape_popular_repos()
        scraper.scrape_stackoverflow()
    
    return scraper.get_combined_dataset(max_chars=max_chars)

def get_mixed_dataset(max_chars=None, code_ratio=0.3):
    """
    Kombiniert Code-Daten mit Text-Daten (Shakespeare/TinyStories).
    code_ratio: Anteil der Code-Daten (0.0-1.0)
    """
    # Text-Daten
    text_data, text_stoi, text_itos, text_vocab = get_large_dataset(max_chars=int(max_chars * (1 - code_ratio)) if max_chars else None)
    
    # Code-Daten
    code_data, code_stoi, code_itos, code_vocab = get_code_dataset(max_chars=int(max_chars * code_ratio) if max_chars else None)
    
    # Kombiniere Vocabulars
    combined_stoi = {**text_stoi, **code_stoi}
    combined_itos = {**text_itos, **code_itos}
    vocab_size = len(combined_stoi)
    
    # Kombiniere Daten
    combined_data = torch.cat([text_data, code_data])
    
    print(f'[DATA] Mixed Dataset: {len(combined_data):,} Tokens, Vocab: {vocab_size}')
    print(f'  Text: {len(text_data):,} Tokens, Code: {len(code_data):,} Tokens')
    
    return combined_data, combined_stoi, combined_itos, vocab_size

def get_large_dataset(max_chars=None):
    """Lädt großen Text-Korpus mit Multi-URL Fallback."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Phase 1: TinyStories via mehrere Mirrors
    file_path = os.path.join(DATA_DIR, 'tinystories_train.txt')
    downloaded = False
    for url in TINYSTORIES_URLS:
        if download_file(url, file_path):
            downloaded = True
            break
    
    if not downloaded:
        print('[DATA] TinyStories nicht verfuegbar. Versuche PG-19 oder erweiterten Shakespeare...')
        file_path = os.path.join(DATA_DIR, 'shakespeare_full.txt')
        if not download_file(SHAKESPEARE_FULL_URL, file_path):
            file_path = os.path.join(DATA_DIR, 'input.txt')
            download_file(DATASET_URL, file_path)
    
    # Datei einlesen (verschiedene Encodings)
    text = None
    for enc in ['utf-8', 'latin-1', 'cp1252']:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                text = f.read()
            break
        except UnicodeDecodeError:
            continue
    
    if text is None:
        raise RuntimeError(f"Konnte {file_path} nicht lesen")
    
    if max_chars:
        text = text[:max_chars]
    
    # Data Augmentation: Dataset vervielfachen wenn klein
    if len(text) < 500000:
        multiplier = min(5, 500000 // max(1, len(text)))
        if multiplier > 1:
            print(f'[DATA] Dataset klein ({len(text):,} Chars). Vervielfache x{multiplier}')
            text = text * multiplier
    
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    print(f'[DATA] Dataset: {len(data):,} Tokens, Vocab: {vocab_size}, Datei: {os.path.basename(file_path)}')
    
    return data, stoi, itos, vocab_size
