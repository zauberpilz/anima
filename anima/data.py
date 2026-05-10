"""
Data utilities for Anima.
"""
import torch
import urllib.request


def get_shakespeare_data(max_chars=None):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    with urllib.request.urlopen(url, timeout=30) as f:
        text = f.read().decode('utf-8')
    if max_chars:
        text = text[:max_chars]

    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    return data, stoi, itos, len(chars)


class SequenceLoader:
    def __init__(self, data, block_size=128, batch_size=16):
        self.data = data
        self.block_size = block_size
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        ix = torch.randint(len(self.data) - self.block_size, (self.batch_size,))
        x = torch.stack([self.data[i:i+self.block_size] for i in ix])
        return x

    def __len__(self):
        return len(self.data) // (self.block_size * self.batch_size)
