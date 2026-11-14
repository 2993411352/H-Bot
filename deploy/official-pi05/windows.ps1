#Requires -Version 5.1
<#
.SYNOPSIS
Install and run the LoongForge official evaluation service in WSL2 Ubuntu.
.EXAMPLE
.\windows.ps1 Setup -Checkpoint /home/alice/checkpoints/steps_30000
.EXAMPLE
.\windows.ps1 SelfTest
.EXAMPLE
.\windows.ps1 Start
#>
[CmdletBinding()]
param(
    [Parameter(Position=0, Mandatory=$true)]
    [ValidateSet('Setup','Configure','Doctor','Check','SelfTest','Start','Attach','Status','ClearSession')]
    [string]$Action,
    [string]$Distribution = 'Ubuntu-22.04',
    [string]$InstallDirectory = '',
    [string]$Checkpoint = '',
    [ValidateRange(0,128)][int]$Gpu = 0,
    [ValidateSet('direct','inherit')][string]$Network = 'direct',
    [string]$ProxyUrl = ''
)
$ErrorActionPreference = 'Stop'
Get-Command wsl.exe -ErrorAction Stop | Out-Null
foreach ($value in @($Distribution, $InstallDirectory, $Checkpoint, $ProxyUrl, $PSScriptRoot)) {
    if ($value -match '["\r\n]') { throw 'Arguments must not contain double quotes or line breaks.' }
}
if ($InstallDirectory -and -not $InstallDirectory.StartsWith('/')) {
    throw 'InstallDirectory must be an absolute Linux path, for example /home/alice/hbot-official.'
}

# wsl.exe list output is UTF-16 even in some UTF-8 PowerShell sessions.
$distros = @(& wsl.exe --list --quiet)
if ($LASTEXITCODE -ne 0) {
    if ($Action -ne 'Setup') { throw 'Cannot query WSL. Run the Setup action in an administrator PowerShell.' }
    & wsl.exe --install -d $Distribution
    if ($LASTEXITCODE -ne 0) { throw "Run in an administrator PowerShell: wsl --install -d $Distribution" }
    Write-Host 'Complete WSL installation and Ubuntu user initialization, then rerun Setup after any requested reboot.'
    exit 0
}
$distros = @($distros | ForEach-Object { ($_ -replace "`0", '').Trim() })
if ($Distribution -notin $distros) {
    if ($Action -ne 'Setup') { throw "WSL distribution $Distribution is missing. Run the Setup action first." }
    & wsl.exe --install -d $Distribution
    if ($LASTEXITCODE -ne 0) { throw "WSL installation requires an administrator terminal. Run: wsl --install -d $Distribution" }
    Write-Host 'Complete Ubuntu user initialization (and reboot if requested), then rerun the same Setup command.'
    exit 0
}

$linuxSource = @(& wsl.exe -d $Distribution --exec wslpath -a -u $PSScriptRoot)
if ($LASTEXITCODE -ne 0) { throw 'Cannot translate the checkout path into WSL.' }
$linuxSource = ($linuxSource -join "`n").Trim()
if ($Checkpoint -and -not $Checkpoint.StartsWith('/')) {
    $resolved = (Resolve-Path -LiteralPath $Checkpoint).Path
    $converted = @(& wsl.exe -d $Distribution --exec wslpath -a -u $resolved)
    if ($LASTEXITCODE -ne 0) { throw 'Cannot translate the checkpoint path into WSL.' }
    $Checkpoint = ($converted -join "`n").Trim()
}
# Encode values separately to preserve spaces, apostrophes and empty arguments
# across Windows PowerShell 5.1 / PowerShell 7 native argument handling.
$encoded = foreach ($value in @($InstallDirectory, $Checkpoint, [string]$Gpu, $Network, $ProxyUrl)) {
    'b64:' + [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($value))
}
& wsl.exe -d $Distribution --exec bash "$linuxSource/windows-wsl.sh" $Action @encoded
if ($LASTEXITCODE -ne 0) { throw "WSL action $Action failed with exit code $LASTEXITCODE. See the output above." }
