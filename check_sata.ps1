Get-PnpDevice | Where-Object { $_.Class -eq 'SCSIAdapter' } | Format-Table FriendlyName, Status -AutoSize
Write-Host "=== SATA im Namen ==="
Get-PnpDevice | Where-Object { $_.FriendlyName -like '*SATA*' } | Format-Table FriendlyName, Status -AutoSize
Write-Host "=== MARVELL/ASMEDIA ==="
Get-PnpDevice | Where-Object { $_.FriendlyName -like '*Marvell*' -or $_.FriendlyName -like '*ASMedia*' -or $_.FriendlyName -like '*Silicon*' } | Format-Table FriendlyName, Status -AutoSize
Write-Host "=== SOUND BLASTER ==="
Get-PnpDevice | Where-Object { $_.FriendlyName -like '*Sound*' -or $_.FriendlyName -like '*AE*' } | Format-Table FriendlyName, Status -AutoSize
Write-Host "=== PCI BRIDGES ==="
Get-PnpDevice | Where-Object { $_.Class -eq 'PCI' -and $_.FriendlyName -notlike '*Microsoft*' } | Format-Table FriendlyName, Status -AutoSize
