"""Erweiterter Data Loader für größere Korpora + Code-Daten."""
import torch
import os
import urllib.request
from code_scraper import CodeScraper
from code_tokenizer import CodeTokenizer

DATA_DIR = '/home/anima/data'
DATASET_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
LARGE_DATASET_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_train.txt"

def download_file(url, dest):
    if not os.path.exists(dest):
        print(f'Downloading {url} to {dest}...')
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        urllib.request.urlretrieve(url, dest)
        print('Download abgeschlossen.')
    else:
        print(f'Datei existiert bereits: {dest}')

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
    """Lädt einen großen Text-Korpus herunter und bereitet ihn auf."""
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, 'tinystories_train.txt')
    
    # Versuch TinyStories zu laden, falls nicht verfügbar, fallback
    try:
        download_file(LARGE_DATASET_URL, file_path)
    except Exception as e:
        print(f'Fehler beim Download von TinyStories: {e}')
        print('Fallback auf Shakespeare...')
        file_path = os.path.join(DATA_DIR, 'input.txt')
        download_file(DATASET_URL, file_path)

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    if max_chars:
        text = text[:max_chars]
        
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    print(f'Dataset geladen: {len(data):,} Tokens, Vocab: {vocab_size}')
    
    return data, stoi, itos, vocab_size
