param(
    [ValidateSet('shadow', 'live')]
    [string]$Mode = '',
    [double]$Scale = 0.0
)
$ErrorActionPreference = 'Stop'
$workspacePath = Split-Path -Parent $PSScriptRoot
$runtimePath = Join-Path $workspacePath 'runtime'
New-Item -ItemType Directory -Path $runtimePath -Force | Out-Null

# Ensure Conda min_env DLLs (Library\bin) are on PATH for Python 3.12 venv
$condaBin = 'C:\Users\liuqi\anaconda3\envs\min_env\Library\bin'
$condaRoot = 'C:\Users\liuqi\anaconda3\envs\min_env'
if (Test-Path -LiteralPath $condaBin) {
    $env:PATH = "$condaRoot;$condaBin;$env:PATH"
}

$pythonPath = Join-Path $workspacePath '.venv\Scripts\python.exe'
$scriptPath = Join-Path $PSScriptRoot 'run_live_executor.py'
$candidateStatePaths = @(
    (Join-Path $runtimePath 'live\live_service.json'),
    (Join-Path $runtimePath 'shadow\live_service.json'),
    (Join-Path $runtimePath 'live_service.json')
)

foreach ($statePath in $candidateStatePaths) {
    if (Test-Path -LiteralPath $statePath) {
        try {
            $svc = Get-Content -LiteralPath $statePath -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($Mode -and $svc.mode -and ($svc.mode -ne $Mode)) {
                continue
            }
            $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$($svc.pid)" -ErrorAction SilentlyContinue
            $isRunning = ($proc -and $proc.CommandLine -and $proc.CommandLine.Contains('run_live_executor.py'))
            if (-not $isRunning) {
                $gp = Get-Process -Id $svc.pid -ErrorAction SilentlyContinue
                if ($gp -and $gp.ProcessName -match '^python') {
                    $isRunning = $true
                }
            }
            if ($isRunning) {
                Write-Output "ARGON4 Live Executor is already running (PID=$($svc.pid), mode=$($svc.mode), scale=$($svc.live_scale_factor))."
                return
            }
        } catch {}
    }
}

$argList = @('-B', '-u', ('"' + $scriptPath + '"'))
if ($Mode) { $argList += @('--mode', $Mode) }
if ($Scale -gt 0) { $argList += @('--scale', "$Scale") }

$stdoutLog = Join-Path $runtimePath 'live_executor.stdout.log'
$stderrLog = Join-Path $runtimePath 'live_executor.stderr.log'

$proc = Start-Process -FilePath $pythonPath -ArgumentList $argList -WorkingDirectory $workspacePath -WindowStyle Hidden -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog -PassThru
$deadline = (Get-Date).AddSeconds(10)
$serviceReady = $false
while ((Get-Date) -lt $deadline) {
    if ($proc.HasExited) {
        break
    }
    foreach ($statePath in $candidateStatePaths) {
        if (Test-Path -LiteralPath $statePath) {
            try {
                $svc = Get-Content -LiteralPath $statePath -Raw -Encoding UTF8 | ConvertFrom-Json
                if ($svc.pid -eq $proc.Id) {
                    $serviceReady = $true
                    break
                }
            } catch {}
        }
    }
    if ($serviceReady) {
        break
    }
    Start-Sleep -Milliseconds 250
}
if ($proc.HasExited) {
    $errText = if (Test-Path -LiteralPath $stderrLog) { Get-Content -LiteralPath $stderrLog -Raw } else { 'Unknown startup error' }
    throw "ARGON4 Live Executor exited during startup:`n$errText"
}
Write-Output "ARGON4 Live Executor started in background (PID=$($proc.Id)). Logs: $stdoutLog / $stderrLog"
