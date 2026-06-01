# watchdog.ps1
# Einfacher PowerShell Watchdog für CogLang v3 Training
# Läuft als Hintergrundjob: überwacht evolve.py, startet bei Crash neu

param(
    [int]$IntervalSeconds = 120,
    [int]$MaxStalledChecks = 15
)

$WSLDistro = "Ubuntu-24-04"
$LogPath = "/home/anima/evolve.log"
$EvolveCmd = "cd /home/anima/src && nice -19 /home/anima/venv/bin/python3 -u coglang_evolve.py > /home/anima/evolve.log 2>&1"

Write-Host "=== ANIMA WATCHDOG ==="
Write-Host "Start: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "Intervall: ${IntervalSeconds}s"
Write-Host "Max Stalled: ${MaxStalledChecks}x (=$($IntervalSeconds*$MaxStalledChecks/60)min)"
Write-Host ""

$lastStep = 0
$stalledCount = 0
$restartCount = 0

while ($true) {
    Start-Sleep -Seconds $IntervalSeconds
    
    $now = Get-Date -Format "HH:mm:ss"
    
    try {
        # Check if process exists
        $procOut = & wsl -d $WSLDistro -e bash -l -c "ps aux | grep coglang_evolve | grep -v grep | wc -l" 2>$null
        $procCount = ($procOut | Out-String).Trim()
        
        if ($procCount -eq "0" -or [string]::IsNullOrWhiteSpace($procCount)) {
            # CRASH! Restart
            Write-Host "[$now] CRASH! evolve.py nicht gefunden. Neustart #$restartCount..."
            
            # Verify checkpoint health
            $nanCheck = & wsl -d $WSLDistro -e bash -l -c @"
python3 -c "
import torch
ckpt = torch.load('/home/anima/checkpoints/checkpoint.pt', map_location='cpu')
nan = sum(torch.isnan(v).sum().item() for v in ckpt['model_state'].values())
if nan > 0:
    for k, v in ckpt['model_state'].items():
        m = torch.isnan(v)
        if m.any():
            v.data = torch.where(m, torch.randn(v.shape, dtype=v.dtype)*0.01, v)
            v.data = v.data.clamp_(-1.0, 1.0)
    torch.save(ckpt, '/home/anima/checkpoints/checkpoint.pt')
    print(f'Fixed {nan} NaN')
else:
    print('OK')
" 2>/dev/null
"@
            Write-Host "  Checkpoint: $nanCheck"
            
            # Restart
            Start-Process -FilePath wsl -ArgumentList "-d", $WSLDistro, "bash", "-l", "-c", $EvolveCmd -WindowStyle Hidden
            Start-Sleep -Seconds 30
            
            # Verify restart
            $retryOut = & wsl -d $WSLDistro -e bash -l -c "ps aux | grep coglang_evolve | grep -v grep | wc -l" 2>$null
            $retryCount = ($retryOut | Out-String).Trim()
            
            if ($retryCount -eq "0") {
                Write-Host "  [FEHLER] Neustart fehlgeschlagen!"
            } else {
                $restartCount++
                Write-Host "  [OK] Neustart erfolgreich (#$restartCount)"
            }
            
            $lastStep = 0
            $stalledCount = 0
        } else {
            # Running - check progress
            $logLine = & wsl -d $WSLDistro -e bash -l -c "tail -5 $LogPath 2>/dev/null | grep -oP 'Step\s+\d+.*?Phase=\w+' | tail -1" 2>$null
            $step = & wsl -d $WSLDistro -e bash -l -c "tail -1 $LogPath 2>/dev/null | grep -oP 'Step\s+\K\d+' | tail -1" 2>$null
            $loss = & wsl -d $WSLDistro -e bash -l -c "tail -1 $LogPath 2>/dev/null | grep -oP 'loss=\K[\d.]+' | tail -1" 2>$null
            
            $stepNum = 0
            if ($step -and $step -match "^\d+$") {
                $stepNum = [int]$step
            }
            
            if ($stepNum -gt $lastStep) {
                $pctStr = & wsl -d $WSLDistro -e bash -l -c "tail -1 $LogPath 2>/dev/null | grep -oP '\[\s*\K[\d.]+(?=%\])' | tail -1" 2>$null
                Write-Host "[$now] Step $step/50000 ($pctStr%) | loss=$loss"
                $lastStep = $stepNum
                $stalledCount = 0
            } else {
                # Maybe the tail isn't updating because of \r chars
                $stepRange = & wsl -d $WSLDistro -e bash -l -c "grep -oP 'Step\s+\K\d+' $LogPath 2>/dev/null | tail -3" 2>$null
                $lastStepInLog = ($stepRange | Out-String).Trim().Split("`n") | Select-Object -Last 1
                if ($lastStepInLog -and $lastStepInLog -match "^\d+$") {
                    $stepNum2 = [int]$lastStepInLog
                    if ($stepNum2 -gt $lastStep) {
                        $lastStep = $stepNum2
                        $stalledCount = 0
                        Write-Host "[$now] Step $stepNum2 (via fallback)"
                    } else {
                        $stalledCount++
                    }
                } else {
                    $stalledCount++
                }
                
                if ($stalledCount -ge $MaxStalledChecks) {
                    Write-Host "[$now] STALLED! Kein Fortschritt seit $($IntervalSeconds*$stalledCount/60)min. Starte neu..."
                    & wsl -d $WSLDistro -e bash -l -c "pkill -f coglang_evolve" 2>$null
                    Start-Sleep -Seconds 10
                    Start-Process -FilePath wsl -ArgumentList "-d", $WSLDistro, "bash", "-l", "-c", $EvolveCmd -WindowStyle Hidden
                    $stalledCount = 0
                }
            }
        }
    }
    catch {
        Write-Host "[$now] Watchdog Error: $_"
    }
}
