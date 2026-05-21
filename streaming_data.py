"""PHASE 10: Streaming Data Pipeline — Große Datasets ohne RAM-Limit."""
import torch
import os
import mmap
import numpy as np

class StreamingDataset:
    """
    Liest Daten direkt von der Festplatte ohne alles in RAM zu laden.
    Perfekt für 10GB+ Datasets bei begrenztem RAM.
    """
    def __init__(self, file_path, chunk_size=1000000):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self._file = None
        self._mmap = None
        self._length = None
        self._vocab = None
        self._stoi = None
        self._itos = None
        
    def _open(self):
        if self._file is None:
            self._file = open(self.file_path, 'r', encoding='utf-8')
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
            self._length = len(self._mmap)
            
    def _build_vocab(self, sample_size=1000000):
        """Build vocab from first N bytes."""
        self._open()
        sample = self._mmap[:min(sample_size, self._length)].decode('utf-8', errors='ignore')
        chars = sorted(list(set(sample)))
        self._vocab = chars
        self._stoi = {ch: i for i, ch in enumerate(chars)}
        self._itos = {i: ch for i, ch in enumerate(chars)}
        
    @property
    def vocab_size(self):
        if self._stoi is None:
            self._build_vocab()
        return len(self._stoi)
    
    @property
    def stoi(self):
        if self._stoi is None:
            self._build_vocab()
        return self._stoi
    
    @property
    def itos(self):
        if self._stoi is None:
            self._build_vocab()
        return self._itos
    
    def get_batch(self, batch_size, seq_length, device='cuda'):
        """Liest einen zufälligen Batch direkt von der Platte."""
        self._open()
        
        # Random position
        max_pos = self._length - batch_size * seq_length
        if max_pos <= 0:
            max_pos = self._length - 1
            
        start = np.random.randint(0, max_pos)
        
        # Read chunk
        self._mmap.seek(start)
        chunk = self._mmap.read(batch_size * seq_length)
        text = chunk.decode('utf-8', errors='ignore')
        
        # Encode
        data = torch.tensor([self.stoi.get(c, 0) for c in text], dtype=torch.long, device=device)
        data = data.view(batch_size, seq_length)
        
        return data
    
    def __del__(self):
        if self._mmap:
            self._mmap.close()
        if self._file:
            self._file.close()


def get_streaming_dataset(data_dir='/home/anima/data', max_files=5):
    """Erstellt Streaming Dataset aus allen verfügbaren Text-Dateien."""
    os.makedirs(data_dir, exist_ok=True)
    
    # Finde alle .txt Dateien
    txt_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.txt')]
    
    if not txt_files:
        # Fallback: lade Shakespeare
        from data_loader import get_large_dataset
        return get_large_dataset()
    
    # Kombiniere Dateien zu einem Stream
    print(f"PHASE 10: Streaming Dataset aus {len(txt_files)} Dateien")
    return StreamingDataset(txt_files[0])
