$ErrorActionPreference = 'Stop'
$workspacePath = Split-Path -Parent $PSScriptRoot
$expectedScript = Join-Path $PSScriptRoot 'argon4_forward_server.py'
$escapedScript = [regex]::Escape($expectedScript)
$scriptArgument = '(?i)(?:^|[\s"])' + $escapedScript + '(?:[\s"]|$)'
# A failed legacy duplicate launch could overwrite service.json with a dead
# PID. Resolve only Python processes with this exact absolute script argument.
$matchingProcesses = @(Get-CimInstance Win32_Process | Where-Object {
    ($_.Name -eq 'python.exe' -or $_.Name -eq 'pythonw.exe') -and
    $_.CommandLine -and $_.CommandLine -match $scriptArgument
})
if ($matchingProcesses.Count -eq 0) { Write-Output 'Already stopped.'; exit 0 }
$matchingIds = @($matchingProcesses | ForEach-Object { $_.ProcessId })
# Stop worker children before the venv launcher; never touch unrelated Python.
$orderedProcesses = @($matchingProcesses | Sort-Object @{Expression={if ($matchingIds -contains $_.ParentProcessId) {0} else {1}}})
foreach ($serviceProcess in $orderedProcesses) {
    $current = Get-CimInstance Win32_Process -Filter "ProcessId=$($serviceProcess.ProcessId)" -ErrorAction SilentlyContinue
    if ($current -and $current.CreationDate -eq $serviceProcess.CreationDate -and $current.CommandLine -match $scriptArgument) {
        Stop-Process -Id $current.ProcessId -ErrorAction Stop
    }
}
Write-Output 'Paper observation stopped. Account records retained.'
