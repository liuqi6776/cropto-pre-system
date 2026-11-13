$ErrorActionPreference = 'Stop'
$workspacePath = Split-Path -Parent $PSScriptRoot
$expectedScript = Join-Path $PSScriptRoot 'argon4_forward_server.py'
$escapedScript = [regex]::Escape($expectedScript)
$scriptArgument = '(?i)(?:^|[\s"])' + $escapedScript + '(?:[\s"]|$)'
# A failed legacy duplicate launch could overwrite service.json with a dead
# PID. Resolve only Python processes with this exact absolute script argument.
$matchingProcesses = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
    ($_.Name -eq 'python.exe' -or $_.Name -eq 'pythonw.exe') -and
    $_.CommandLine -and $_.CommandLine -match $scriptArgument
})
if ($matchingProcesses.Count -eq 0) {
    $serviceJsonPath = Join-Path $workspacePath 'paper_runtime\argon4_forward_20261105\service.json'
    if (Test-Path -LiteralPath $serviceJsonPath) {
        try {
            $svc = Get-Content -LiteralPath $serviceJsonPath -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($svc.pid) {
                $p = Get-Process -Id ([int]$svc.pid) -ErrorAction SilentlyContinue
                if ($p -and ($p.ProcessName -eq 'python' -or $p.ProcessName -eq 'pythonw')) {
                    Stop-Process -Id $p.Id -Force -ErrorAction Stop
                    Write-Output "Paper observation stopped (PID=$($p.Id)). Account records retained."
                    exit 0
                }
            }
        } catch {}
    }
    Write-Output 'Already stopped.'
    exit 0
}
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
