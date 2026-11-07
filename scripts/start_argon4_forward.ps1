param([switch]$OpenBrowser)
$ErrorActionPreference = 'Stop'
$workspacePath = Split-Path -Parent $PSScriptRoot
$runtimePath = Join-Path $workspacePath 'paper_runtime\argon4_forward_20261105'
$serverPath = Join-Path $PSScriptRoot 'argon4_forward_server.py'
$pythonPath = Join-Path $workspacePath '.venv\Scripts\python.exe'
$paperPolicy = Get-Content -LiteralPath (Join-Path $workspacePath 'config\argon4_forward_policy.json') -Raw -Encoding UTF8 | ConvertFrom-Json
New-Item -ItemType Directory -Path $runtimePath -Force | Out-Null
$existingListener = Get-NetTCPConnection -LocalPort 8897 -State Listen -ErrorAction SilentlyContinue
if ($existingListener) {
    $activeProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$($existingListener[0].OwningProcess)"
    if (-not $activeProcess.CommandLine.Contains($serverPath) -or $existingListener[0].LocalAddress -ne '127.0.0.1') { throw 'Port 8897 is occupied by another application or non-local listener.' }
} else {
    $processArgs = @('-B', '-u', ('"' + $serverPath + '"'), '--port', '8897')
    if ($paperPolicy.paper_trading_active -eq $true) { $processArgs += '--enable-paper' }
    Start-Process -FilePath $pythonPath -ArgumentList $processArgs -WorkingDirectory $workspacePath -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runtimePath 'server.stdout.log') -RedirectStandardError (Join-Path $runtimePath 'server.stderr.log') | Out-Null
}
$ready = $false
for ($attempt = 0; $attempt -lt 10; $attempt++) {
    try {
        $panelStatus = Invoke-RestMethod -Uri 'http://127.0.0.1:8897/api/status' -TimeoutSec 2
        if ($panelStatus.strategy -eq 'ARGON4_SEVEN_STREAM_PRIORITY_WATERFALL' -and $panelStatus.live_enabled -eq $false) { $ready = $true; break }
    } catch { Start-Sleep -Milliseconds 500 }
}
if (-not $ready) { throw 'Panel did not become ready. Check the new runtime server.stderr.log.' }
if ($OpenBrowser) { Start-Process 'http://127.0.0.1:8897/' }
Write-Output "Local paper panel: http://127.0.0.1:8897/ (status: $($panelStatus.service_status); live trading disabled)"
