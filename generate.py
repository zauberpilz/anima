import torch
from hybrid_model import HybridLM

def load_data():
    import urllib.request
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    with urllib.request.urlopen(url) as f:
        text = f.read().decode('utf-8')
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos, len(chars)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    stoi, itos, vocab_size = load_data()

    model = HybridLM(
        vocab_size=vocab_size,
        d_model=384,
        n_layers=6,
        d_state=16,
        n_heads=4,
        window_size=64,
    ).to(device)

    model.load_state_dict(torch.load("hybrid_model.pt", map_location=device))
    model.eval()

    prompt = "ROMEO:"
    context = torch.tensor([[stoi[c] for c in prompt]], device=device)
    out = model.generate(context, max_new_tokens=500, temperature=0.8)
    print(''.join([itos[int(i)] for i in out[0]]))

if __name__ == "__main__":
    main()
