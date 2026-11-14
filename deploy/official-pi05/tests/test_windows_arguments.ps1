# Run with Windows PowerShell 5.1 or PowerShell 7. No real WSL calls.
$ErrorActionPreference = 'Stop'
$entry = Join-Path $PSScriptRoot '..\windows.ps1'
$tokens = $null
$errors = $null
[System.Management.Automation.Language.Parser]::ParseFile((Resolve-Path $entry), [ref]$tokens, [ref]$errors) | Out-Null
if ($errors.Count) { throw ($errors | Out-String) }
$global:HbotTestCalls = @()
function global:wsl.exe {
    $global:HbotTestCalls += ,@($args)
    $global:LASTEXITCODE = 0
    if ($args[0] -eq '--list') { return 'Ubuntu-22.04' }
    if ($args -contains 'wslpath') { return '/mnt/c/checkout with spaces/deploy/official-pi05' }
}
try {
    $checkpoint = "/home/alice/model with 'quotes'/`$literal;echo unsafe"
    & $entry Configure -Checkpoint $checkpoint -InstallDirectory '/home/alice/install with spaces'
    $call = $global:HbotTestCalls[-1]
    if ($call[4] -ne '/mnt/c/checkout with spaces/deploy/official-pi05/windows-wsl.sh') { throw 'Script path split incorrectly' }
    $decoded = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($call[7].Substring(4)))
    if ($decoded -cne $checkpoint) { throw 'Checkpoint argument was modified' }
    if ($call.Count -ne 11) { throw 'Unexpected native argument count' }
    Write-Output 'WINDOWS_ARGUMENT_TESTS_OK'
}
finally {
    Remove-Item Function:\wsl.exe
    Remove-Variable HbotTestCalls -Scope Global
}
