param([switch]$OpenBrowser)
$ErrorActionPreference = 'Stop'
$workspacePath = Split-Path -Parent $PSScriptRoot
$runtimePath = Join-Path $workspacePath 'paper_runtime\argon4_forward_20261105'
if (Test-Path -LiteralPath (Join-Path $runtimePath 'maintenance_pause.json')) {
    Write-Output 'ARGON4 maintenance in progress; automatic restart is temporarily paused.'
    return
}
$serverPath = Join-Path $PSScriptRoot 'argon4_forward_server.py'
$pythonPath = Join-Path $workspacePath '.venv\Scripts\python.exe'
$paperPolicy = Get-Content -LiteralPath (Join-Path $workspacePath 'config\argon4_forward_policy.json') -Raw -Encoding UTF8 | ConvertFrom-Json
New-Item -ItemType Directory -Path $runtimePath -Force | Out-Null
$panelStatus = $null
try { $panelStatus = Invoke-RestMethod -Uri 'http://127.0.0.1:8897/api/status' -TimeoutSec 3 } catch {}
if ($panelStatus -and ($panelStatus.strategy -ne 'ARGON4_SEVEN_STREAM_PRIORITY_WATERFALL' -or $panelStatus.live_enabled -ne $false)) { throw 'Port 8897 is serving an unexpected application.' }
$existingService = $false
if (-not $panelStatus) {
    $statePath = Join-Path $runtimePath 'service.json'
    if (Test-Path -LiteralPath $statePath) {
        $serviceState = Get-Content -LiteralPath $statePath -Raw -Encoding UTF8 | ConvertFrom-Json
        $serviceProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$($serviceState.pid)" -ErrorAction SilentlyContinue
        $existingService = $serviceProcess -and $serviceProcess.CommandLine -and $serviceProcess.CommandLine.Contains($serverPath)
    }
}
if (-not $panelStatus -and -not $existingService) {
    # A bind probe is reliable even where Get-NetTCPConnection misses a socket.
    # The Python server also binds BEFORE constructing the account, closing the
    # race if two launchers start between this probe and Start-Process.
    $portProbe = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 8897)
    try { $portProbe.Start() } catch { throw 'Port 8897 is occupied; refusing a second paper process.' } finally { $portProbe.Stop() }
    $processArgs = @('-B', '-u', ('"' + $serverPath + '"'), '--port', '8897')
    if ($paperPolicy.paper_trading_active -eq $true) { $processArgs += '--enable-paper' }
    Start-Process -FilePath $pythonPath -ArgumentList $processArgs -WorkingDirectory $workspacePath -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runtimePath 'server.stdout.log') -RedirectStandardError (Join-Path $runtimePath 'server.stderr.log') | Out-Null
}
$ready = $false
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    try {
        $panelStatus = Invoke-RestMethod -Uri 'http://127.0.0.1:8897/api/status' -TimeoutSec 2
        if ($panelStatus.strategy -eq 'ARGON4_SEVEN_STREAM_PRIORITY_WATERFALL' -and $panelStatus.live_enabled -eq $false) { $ready = $true; break }
    } catch { Start-Sleep -Milliseconds 500 }
}
if (-not $ready) { throw 'Panel did not become ready. Check the new runtime server.stderr.log.' }
if ($OpenBrowser) { Start-Process 'http://127.0.0.1:8897/' }
Write-Output "Local paper panel: http://127.0.0.1:8897/ (status: $($panelStatus.service_status); live trading disabled)"
