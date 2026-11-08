param([switch]$OpenBrowser)
$ErrorActionPreference = 'Stop'
$workspacePath = Split-Path -Parent $PSScriptRoot
$runtimePath = Join-Path $workspacePath 'paper_runtime\argon4_forward_20261105'
if (Test-Path -LiteralPath (Join-Path $runtimePath 'maintenance_pause.json')) {
    Write-Output 'ARGON4 maintenance in progress; automatic restart is temporarily paused.'
    return
}
$serverPath = Join-Path $PSScriptRoot 'argon4_forward_server.py'
$pythonPath = if (Test-Path -LiteralPath 'C:\Users\liuqi\anaconda3\envs\min_env\python.exe') {
    'C:\Users\liuqi\anaconda3\envs\min_env\python.exe'
} else {
    Join-Path $workspacePath '.venv\Scripts\python.exe'
}
$sitePackages = Join-Path $workspacePath '.venv\Lib\site-packages'
if (Test-Path -LiteralPath $sitePackages) {
    $env:PYTHONPATH = if ($env:PYTHONPATH) { "$sitePackages;$env:PYTHONPATH" } else { $sitePackages }
}
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
    $stdoutLog = Join-Path $runtimePath 'server.stdout.log'
    $stderrLog = Join-Path $runtimePath 'server.stderr.log'
    $paperFlag = if ($paperPolicy.paper_trading_active -eq $true) { ' --enable-paper' } else { '' }
    $cmdLine = "cmd.exe /c `"`"$pythonPath`" -B -u `"$serverPath`" --port 8897$paperFlag 1>`"$stdoutLog`" 2>`"$stderrLog`"`""
    $startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
    $created = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine = $cmdLine
        CurrentDirectory = $workspacePath
        ProcessStartupInformation = $startup
    }
    if ($created.ReturnValue -ne 0) {
        $processArgs = @('-B', '-u', ('"' + $serverPath + '"'), '--port', '8897')
        if ($paperPolicy.paper_trading_active -eq $true) { $processArgs += '--enable-paper' }
        Start-Process -FilePath $pythonPath -ArgumentList $processArgs -WorkingDirectory $workspacePath -WindowStyle Hidden -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog | Out-Null
    }
}
$ready = $false
for ($attempt = 0; $attempt -lt 60; $attempt++) {
    try {
        $panelStatus = Invoke-RestMethod -Uri 'http://127.0.0.1:8897/api/status' -TimeoutSec 2
        if ($panelStatus.strategy -eq 'ARGON4_SEVEN_STREAM_PRIORITY_WATERFALL' -and $panelStatus.live_enabled -eq $false) { $ready = $true; break }
    } catch { Start-Sleep -Milliseconds 500 }
}
if (-not $ready) { throw 'Panel did not become ready. Check the new runtime server.stderr.log.' }
if ($OpenBrowser) { Start-Process 'http://127.0.0.1:8897/' }
Write-Output "Local paper panel: http://127.0.0.1:8897/ (status: $($panelStatus.service_status); live trading disabled)"
