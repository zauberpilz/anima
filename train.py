import torch
import torch.nn.functional as F
import time
from hybrid_model import HybridLM

def get_shakespeare_data():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    import urllib.request
    print("Lade Shakespeare-Daten...")
    with urllib.request.urlopen(url) as f:
        text = f.read().decode('utf-8')
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    return data, stoi, itos, len(chars)

@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, eval_iters=10):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

def main():
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data, stoi, itos, vocab_size = get_shakespeare_data()
    print(f"Vokabular: {vocab_size} Zeichen, {len(data)} Tokens")

    n_train = int(0.9 * len(data))
    train_data, val_data = data[:n_train], data[n_train:]

    model = HybridLM(
        vocab_size=vocab_size,
        d_model=256,
        n_layers=4,
        d_state=16,
        n_heads=4,
        window_size=64,
        max_seq_len=256,
    ).to(device)



    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nParameter: {n_params/1e6:.2f}M")
    print(f"Größe: {n_params*4/1024/1024:.1f} MB (FP32)")
    print(f"Speicherkomplexität pro Token: O(d_model*d_state) statt O(n*d_model)")
    print(f"Attention-Komplexität: O(window_size) statt O(seq_len) pro Token\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    batch_size = 32
    block_size = 128
    max_iters = 1500
    eval_interval = 200

    model.train()
    print(f"{'Step':>5} | {'Train Loss':>10} | {'Val Loss':>9} | {'Zeit':>8} | {'VRAM':>6}")
    t0 = time.time()
    for step in range(max_iters):
        ix = torch.randint(len(train_data) - block_size, (batch_size,))
        x = torch.stack([train_data[i:i+block_size] for i in ix]).to(device)
        y = torch.stack([train_data[i+1:i+block_size+1] for i in ix]).to(device)

        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % eval_interval == 0 or step == max_iters - 1:
            val_loss = estimate_loss(model, val_data, block_size, batch_size, eval_iters=5)
            dt = time.time() - t0
            mem = torch.cuda.max_memory_allocated()/1024/1024 if torch.cuda.is_available() else 0
            print(f"{step:5d} | {loss.item():10.4f} | {val_loss:9.4f} | {dt:7.1f}s | {mem:5.0f}MB")
            t0 = time.time()

            context = data[:50].unsqueeze(0).to(device)
            out = model.generate(context, max_new_tokens=100, temperature=1.0)
            print(f"Sample: {''.join([itos[int(i)] for i in out[0][50:]])}\n")

    torch.save(model.state_dict() if not hasattr(model, 'module') else model.module.state_dict(), "hybrid_model.pt")
    print("Gespeichert als hybrid_model.pt")

if __name__ == "__main__":
    main()
