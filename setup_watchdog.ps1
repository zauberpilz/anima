# setup_watchdog.ps1
# Erstellt eine geplante Windows-Aufgabe für das ANIMA-Training
# Läuft alle 30 Minuten und startet evolve.py falls abgestürzt

$TaskName = "AnimaTrainingWatchdog"
$ScriptPath = "C:\Users\admin\Documents\Besseres LLM\watchdog.ps1"
$WSLDistro = "Ubuntu-24.04"

# Prüfe ob PowerShell als Admin läuft
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "Admin-Rechte benötigt! Starte neu..." -ForegroundColor Yellow
    Start-Process powershell -Verb RunAs -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`""
    exit
}

Write-Host "=== ANIMA Training Watchdog Setup ===" -ForegroundColor Cyan
Write-Host ""

# 1. Task beenden falls vorhanden
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue

# 2. Task erstellen (alle 30 Minuten, auch im Leerlauf)
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -Command `
    `$step = wsl -d $WSLDistro -e bash -l -c 'ps aux | grep coglang_evolve | grep -v grep | wc -l' 2>`$null; `
    if (`$step.Trim() -eq '0') { `
        Write-Host ('[' + (Get-Date -Format 'HH:mm:ss') + '] evolve.py CRASHED! Restarting...'); `
        Start-Process -FilePath wsl -ArgumentList '-d', $WSLDistro, 'bash', '-l', '-c', 'cd /home/anima/src && nice -19 /home/anima/venv/bin/python3 -u coglang_evolve.py > /home/anima/evolve.log 2>&1' -WindowStyle Hidden; `
    } else { `
        `$loss = wsl -d $WSLDistro -e bash -l -c `"tail -1 /home/anima/evolve.log | grep -oP 'loss=\\K[\\d.]+' | tail -1`" 2>`$null; `
        `$step = wsl -d $WSLDistro -e bash -l -c `"tail -1 /home/anima/evolve.log | grep -oP 'Step\\s+\\K\\d+' | tail -1`" 2>`$null; `
        Write-Host ('[' + (Get-Date -Format 'HH:mm:ss') + '] Training OK: Step ' + `$step + ' | loss=' + `$loss); `
    }"

$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 30) -RepetitionDuration (New-TimeSpan -Days 365)
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -Compatibility Win8
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -RunLevel Limited

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force

Write-Host "Geplante Aufgabe '$TaskName' erstellt!" -ForegroundColor Green
Write-Host "Prüft alle 30 Minuten den evolve.py Prozess" -ForegroundColor Green
Write-Host ""

# 3. Prüfe ob training läuft
Write-Host "=== Aktueller Status ===" -ForegroundColor Cyan
$procCount = wsl -d $WSLDistro -e bash -l -c "ps aux | grep coglang_evolve | grep -v grep | wc -l" 2>$null
if ($procCount.Trim() -gt 0) {
    Write-Host "evolve.py läuft! (Prozesse: $($procCount.Trim()))" -ForegroundColor Green
    wsl -d $WSLDistro -e bash -l -c "tail -3 /home/anima/evolve.log | grep -oP 'Step\s+\d+.*?Phase=\w+' | tail -1" 2>$null
} else {
    Write-Host "evolve.py läuft NICHT! Starte..." -ForegroundColor Yellow
    Start-Process -FilePath wsl -ArgumentList "-d", $WSLDistro, "bash", "-l", "-c", "cd /home/anima/src && nice -19 /home/anima/venv/bin/python3 -u coglang_evolve.py > /home/anima/evolve.log 2>&1" -WindowStyle Hidden
    Start-Sleep -Seconds 15
    $verify = wsl -d $WSLDistro -e bash -l -c "ps aux | grep coglang_evolve | grep -v grep | wc -l" 2>$null
    if ($verify.Trim() -gt 0) {
        Write-Host "evolve.py gestartet!" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "Dashboard: http://172.21.214.125:8080" -ForegroundColor Cyan
Write-Host "Log: /home/anima/evolve.log" -ForegroundColor Gray
Write-Host "Steuerung: python3 training_controller.py pause/resume/stop" -ForegroundColor Gray
