# start_training.ps1
# Startet Training + Watchdog für CogLang v3 ANIMA
# Kann als geplante Aufgabe oder manuell ausgeführt werden

$WSL_DISTRO = "Ubuntu-24.04"
$PROJECT_DIR = "C:\Users\admin\Documents\Besseres LLM"

Write-Host "=== CogLang v3 ANIMA Training Starter ===" -ForegroundColor Cyan
Write-Host ""

# 1. WSL sicher starten
Write-Host "[1/4] WSL starten..." -NoNewline
wsl --shutdown 2>$null
Start-Sleep -Seconds 2
wsl -d $WSL_DISTRO -e bash -l -c "echo OK" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host " FEHLER" -ForegroundColor Red
    exit 1
}
Write-Host " OK" -ForegroundColor Green

# 2. Quelle syncen
Write-Host "[2/4] Source syncen..." -NoNewline
wsl -d $WSL_DISTRO -e bash -l -c "cp '/mnt/c/Users/admin/Documents/Besseres LLM/coglang.py' /home/anima/src/coglang.py && cp '/mnt/c/Users/admin/Documents/Besseres LLM/coglang_evolve.py' /home/anima/src/coglang_evolve.py && echo SYNCED" 2>$null
Write-Host " OK" -ForegroundColor Green

# 3. Checkpoint-Health prüfen
Write-Host "[3/4] Checkpoint prüfen..." -NoNewline
$result = wsl -d $WSL_DISTRO -e bash -l -c @"
python3 -c "
import torch
ckpt = torch.load('/home/anima/checkpoints/checkpoint.pt', map_location='cpu')
total = sum(v.numel() for v in ckpt['model_state'].values())
nan = sum(torch.isnan(v).sum().item() for v in ckpt['model_state'].values())
print(f'{nan}/{total}')
" 2>/dev/null
"@
if ($result -match "(\d+)/\d+") {
    $nanCount = [int]$Matches[1]
    if ($nanCount -gt 0) {
        Write-Host " $nanCount NaN gefunden! Repariere..." -ForegroundColor Yellow
        wsl -d $WSL_DISTRO -e bash -l -c @"
python3 -c "
import torch
ckpt = torch.load('/home/anima/checkpoints/checkpoint.pt', map_location='cpu')
ms = ckpt['model_state']
for k, v in ms.items():
    nan_mask = torch.isnan(v)
    if nan_mask.any():
        rand_vals = torch.randn(v.shape, dtype=v.dtype) * 0.01
        v.data = torch.where(nan_mask, rand_vals, v)
        v.data = v.data.clamp_(-1.0, 1.0)
torch.save(ckpt, '/home/anima/checkpoints/checkpoint.pt')
print('Repariert')
" 2>/dev/null
"@
    } else {
        Write-Host " OK" -ForegroundColor Green
    }
} else {
    Write-Host " FEHLER" -ForegroundColor Red
}

# 4. Training starten (in WSL Background)
Write-Host "[4/4] Training starten..." -NoNewline
wsl -d $WSL_DISTRO -e bash -l -c @"
mkdir -p /home/anima/control
echo 'resume' > /home/anima/control/signal.txt
cd /home/anima/src && nohup nice -19 /home/anima/venv/bin/python3 -u coglang_evolve.py > /home/anima/evolve.log 2>&1 &
"@
Start-Sleep -Seconds 5
$running = wsl -d $WSL_DISTRO -e bash -l -c "ps aux | grep coglang_evolve | grep -v grep | wc -l" 2>$null
$running = $running.Trim()
if ($running -gt 0) {
    Write-Host " OK (PID läuft)" -ForegroundColor Green
} else {
    Write-Host " FEHLER" -ForegroundColor Red
    
    # Fallback: direkter Start
    Write-Host "  → Fallback-Modus..." -NoNewline
    $proc = Start-Process -FilePath "wsl" -ArgumentList "-d", $WSL_DISTRO, "bash", "-l", "-c", `
        "cd /home/anima/src && nohup nice -19 /home/anima/venv/bin/python3 -u coglang_evolve.py > /home/anima/evolve.log 2>&1 &" `
        -NoNewWindow -PassThru
    Start-Sleep -Seconds 5
    Write-Host " gestartet" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== Status ===" -ForegroundColor Cyan
wsl -d $WSL_DISTRO -e bash -l -c "tail -3 /home/anima/evolve.log 2>/dev/null | head -1" 2>$null
Write-Host ""

# Watchdog starten (im aktuellen Fenster oder neuem)
Write-Host "Watchdog starten... (Ctrl+C zum Beenden)" -ForegroundColor Cyan
Write-Host ""
python "$PROJECT_DIR\watchdog.py"
