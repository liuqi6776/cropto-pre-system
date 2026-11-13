param(
    [ValidateSet('', 'shadow', 'live')]
    [string]$Mode = ''
)
$ErrorActionPreference = 'Stop'
$workspacePath = Split-Path -Parent $PSScriptRoot

$candidatePaths = @()
if ($Mode -eq 'live') {
    $candidatePaths += (Join-Path $workspacePath 'runtime\live\live_service.json')
    $candidatePaths += (Join-Path $workspacePath 'runtime\live_service.json')
} elseif ($Mode -eq 'shadow') {
    $candidatePaths += (Join-Path $workspacePath 'runtime\shadow\live_service.json')
    $candidatePaths += (Join-Path $workspacePath 'runtime\live_service.json')
} else {
    $candidatePaths += (Join-Path $workspacePath 'runtime\live\live_service.json')
    $candidatePaths += (Join-Path $workspacePath 'runtime\shadow\live_service.json')
    $candidatePaths += (Join-Path $workspacePath 'runtime\live_service.json')
}

$stoppedAny = $false
$foundAnyFile = $false

foreach ($statePath in $candidatePaths) {
    if (Test-Path -LiteralPath $statePath) {
        $foundAnyFile = $true
        try {
            $svc = Get-Content -LiteralPath $statePath -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($Mode -and $svc.mode -and ($svc.mode -ne $Mode)) {
                continue
            }
            $proc = Get-Process -Id $svc.pid -ErrorAction SilentlyContinue
            if ($proc) {
                Stop-Process -Id $svc.pid -Force -ErrorAction SilentlyContinue
                Write-Output "Stopped ARGON4 Live Executor (PID=$($svc.pid), mode=$($svc.mode)) from $statePath."
                $stoppedAny = $true
            }
        } finally {
            Remove-Item -LiteralPath $statePath -Force -ErrorAction SilentlyContinue
        }
    }
}

if (-not $foundAnyFile -and -not $stoppedAny) {
    Write-Output "No active live_service.json found in runtime/live, runtime/shadow, or runtime/."
}
