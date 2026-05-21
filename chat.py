"""Interaktive Chat-Schnittstelle für Cobra."""
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/home/anima')
from coglang import build_anima
from data_loader import get_large_dataset

device = 'cuda'

def load_best_model(vocab_size):
    brain = build_anima(vocab_size=vocab_size, device=device)
    path = '/home/anima/best_model.pt'
    loaded = brain.load_checkpoint(path)
    if loaded:
        print(f"Best Model geladen (Iteration {loaded.get('iteration', '?')})")
    else:
        print("Kein Best Model gefunden, nutze aktuellen Checkpoint.")
        brain.load_checkpoint('/home/anima/checkpoint.pt')
    return brain

def chat_loop():
    data, stoi, itos, vocab_size = get_large_dataset()
    brain = load_best_model(vocab_size)
    brain.eval() # Set modules to eval mode if applicable (though no dropout etc)

    print("\n=== COBRA CHAT ===")
    print("Tippe 'exit' zum Beenden.")
    print("Tippe 'reset' um den Kontext zu löschen.")
    
    context = []
    
    while True:
        user_input = input("\nDu: ")
        if user_input.lower() == 'exit':
            break
        if user_input.lower() == 'reset':
            context = []
            print("Kontext gelöscht.")
            continue
            
        context.extend([stoi.get(c, 0) for c in user_input])
        ctx_tensor = torch.tensor([context], device=device)
        
        # Generate response
        print("Cobra: ", end='', flush=True)
        response = []
        for _ in range(100):
            out, info = brain.forward(ctx_tensor[:, -256:], learn=False)
            logits = out[:, -1, :] / 0.8
            k = 40
            top_k_logits, top_k_indices = torch.topk(logits, k)
            probs = F.softmax(top_k_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            next_token = top_k_indices.gather(1, next_token)
            
            token_id = next_token.item()
            char = itos.get(token_id, '?')
            print(char, end='', flush=True)
            response.append(token_id)
            context.append(token_id)
            ctx_tensor = torch.tensor([context], device=device)
        print() # Newline after generation

if __name__ == '__main__':
    chat_loop()
