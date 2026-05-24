$url = "https://www.station-drivers.com/index.php/en/component/remository/func-startdown/5832/lang,en-gb/"
$out = "$env:TEMP\Marvell_MSU.exe"
Write-Host "Downloading Marvell Storage Utility..."
try {
    Invoke-WebRequest -Uri $url -OutFile $out -UserAgent "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36" -ErrorAction Stop
    $f = Get-Item $out
    Write-Host "OK! $($f.Length / 1MB) MB heruntergeladen nach: $out"
    Write-Host "Starte Installation..."
    Start-Process -FilePath $out -Wait
    Write-Host "Installation abgeschlossen."
}
catch {
    Write-Host "FEHLER: $_"
}
