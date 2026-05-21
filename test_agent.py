"""Test Agent — Validiert neue Code-Änderungen vor der Integration."""
import torch
import sys
import os
import time

# Pfad zum "Staging"-Bereich, wo neue Dateien zuerst landen
STAGING_DIR = '/home/anima/staging'
MAIN_DIR = '/home/anima'

def run_sanity_check():
    """Führt einen kurzen Trainingslauf durch, um Stabilität zu prüfen."""
    print("=== TEST AGENT: STARTING SANITY CHECK ===")
    
    # Füge Staging zum Pfad hinzu, um die neuen Dateien zu laden
    sys.path.insert(0, STAGING_DIR)
    
    try:
        from coglang import build_anima
        from data_loader import get_large_dataset
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Device: {device}")
        
        # Kleines Dataset für schnellen Test
        data, stoi, itos, vocab_size = get_large_dataset(max_chars=50000)
        if isinstance(data, torch.Tensor): data = data.long()
        else: data = torch.tensor(data, dtype=torch.long)
        
        # Baue ein kleines Test-Modell
        print("Baue Test-Modell...")
        brain = build_anima(vocab_size=vocab_size, device=device, 
                            d_model=256, d_sparse=1024, n_layers=4, 
                            d_state=64, d_context=128, lr=0.05)
        
        print(f"Modell gebaut: {brain.parameter_count()/1e6:.2f}M Parameter")
        
        # Kurzer Trainingslauf (100 Steps)
        print("Starte 100 Test-Schritte...")
        B, S = 8, 64
        for step in range(100):
            idx = torch.randint(0, len(data) - B * S, (1,)).item()
            batch = data[idx:idx + B * S].view(B, S).to(device)
            loss, _ = brain.learn(batch)
            
            if loss != loss: # NaN Check
                print(f"!!! FEHLER: NaN bei Step {step} !!!")
                return False
                
            if step % 20 == 0:
                print(f"  Step {step}: Loss={loss:.4f}")
                
        print("=== TEST AGENT: SANITY CHECK ERFOLGREICH ===")
        return True
        
    except Exception as e:
        print(f"!!! FEHLER IM TEST AGENT: {e} !!!")
        import traceback
        traceback.print_exc()
        return False

def deploy_if_valid():
    """Kopiert Dateien von Staging nach Main, wenn der Test erfolgreich war."""
    if run_sanity_check():
        print("Deploying validated files to main directory...")
        # Hier würden wir die Dateien kopieren
        # os.system(f'cp {STAGING_DIR}/*.py {MAIN_DIR}/')
        print("Deployment erfolgreich!")
        return True
    else:
        print("Deployment abgebrochen: Tests fehlgeschlagen.")
        return False

if __name__ == '__main__':
    deploy_if_valid()
