$ErrorActionPreference = 'Stop'
$workspacePath = Split-Path -Parent $PSScriptRoot
$statePath = Join-Path $workspacePath 'paper_runtime\argon4_forward_20261105\service.json'
if (-not (Test-Path -LiteralPath $statePath)) { Write-Output 'No service state.'; exit 0 }
$serviceState = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
$serviceProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$($serviceState.pid)" -ErrorAction SilentlyContinue
if (-not $serviceProcess) { Write-Output 'Already stopped.'; exit 0 }
$expectedScript = Join-Path $PSScriptRoot 'argon4_forward_server.py'
if (-not $serviceProcess.CommandLine.Contains($expectedScript)) { throw 'PID identity mismatch; refusing to stop another process.' }
Stop-Process -Id $serviceState.pid
Write-Output 'Paper observation stopped. Account records retained.'
