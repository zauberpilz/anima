# SATA-Karte Details + Soundblaster Config
Get-PnpDevice | Where-Object { $_.FriendlyName -like '*AHCI*' -or $_.FriendlyName -like '*SATA*' -or $_.FriendlyName -like '*ASMedia*' -or $_.FriendlyName -like '*Marvell*' -or $_.FriendlyName -like '*JMB*' -or $_.FriendlyName -like '*JMicron*' } | Select-Object FriendlyName, Class, Status, InstanceId | Format-Table -AutoSize

Write-Host "=== PCI Hardware IDs ==="
Get-PnpDeviceProperty -InstanceId (Get-PnpDevice | Where-Object { $_.FriendlyName -like '*AHCI*' } | Select-Object -First 1 -ExpandProperty InstanceId) -KeyName 'DEVPKEY_Device_HardwareIds' 2>$null | Select-Object -ExpandProperty Data

Write-Host "=== Alle PCI Speicher/Storage Controller ==="
Get-PnpDevice | Where-Object { $_.Class -eq 'SCSIAdapter' -and $_.FriendlyName -like '*AHCI*' } | ForEach-Object {
    Write-Host $_.FriendlyName
    $hw = Get-PnpDeviceProperty -InstanceId $_.InstanceId -KeyName 'DEVPKEY_Device_HardwareIds'
    $hw.Data
    Write-Host "---"
}
