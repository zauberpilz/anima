"""Erweiterter Data Loader für größere Korpora."""
import torch
import os
import urllib.request

DATA_DIR = '/home/anima/data'
DATASET_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt" # Fallback
# Für einen größeren Korpus könnten wir hier z.B. TinyStories nutzen
LARGE_DATASET_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_train.txt"

def download_file(url, dest):
    if not os.path.exists(dest):
        print(f'Downloading {url} to {dest}...')
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        urllib.request.urlretrieve(url, dest)
        print('Download abgeschlossen.')
    else:
        print(f'Datei existiert bereits: {dest}')

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
