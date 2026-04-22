param(
    [switch]$ForceRestart
)

$ErrorActionPreference = "Stop"

$projectPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$appPath = Join-Path $projectPath "app.py"

$pythonCandidates = @(
    $env:TRADER_PREMIUM_MAX_PYTHON,
    "C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
) | Where-Object { $_ -and (Test-Path $_) }

if (-not $pythonCandidates) {
    throw "Nao encontrei um Python valido. Defina TRADER_PREMIUM_MAX_PYTHON ou reinstale o runtime local."
}

$pythonPath = @($pythonCandidates)[0]

function Get-ProjectProcesses([string[]]$needles) {
    $projectNeedle = $projectPath.ToLowerInvariant()
    $allNeedles = @($projectNeedle) + ($needles | ForEach-Object { $_.ToLowerInvariant() })
    Get-CimInstance Win32_Process | Where-Object {
        $cmd = [string]$_.CommandLine
        if (-not $cmd) {
            return $false
        }

        $cmdLower = $cmd.ToLowerInvariant()
        foreach ($needle in $allNeedles) {
            if (-not $cmdLower.Contains($needle)) {
                return $false
            }
        }

        return $true
    }
}

function Start-ProjectWindow([string]$title, [string]$commandLine) {
    Start-Process -FilePath "$env:SystemRoot\System32\cmd.exe" `
        -ArgumentList @("/k", "title $title && cd /d `"$projectPath`" && $commandLine") `
        -WorkingDirectory $projectPath | Out-Null
}

$workerProcesses = @(Get-ProjectProcesses -needles @("-m workers.trader_worker"))
$webProcesses = @(Get-ProjectProcesses -needles @("-m streamlit run", "app.py"))

if ($ForceRestart) {
    foreach ($proc in ($workerProcesses + $webProcesses)) {
        try {
            Stop-Process -Id $proc.ProcessId -Force -ErrorAction Stop
        } catch {
        }
    }

    Start-Sleep -Seconds 1
    $workerProcesses = @()
    $webProcesses = @()
}

if (-not $workerProcesses) {
    Start-ProjectWindow -title "Trader Premium Max - Worker" -commandLine "`"$pythonPath`" -m workers.trader_worker"
}

if (-not $webProcesses) {
    Start-ProjectWindow -title "Trader Premium Max - Web" -commandLine "`"$pythonPath`" -m streamlit run `"$appPath`""
}

Write-Host "Launcher concluido."
Write-Host "Projeto: $projectPath"
Write-Host "Python: $pythonPath"
Write-Host "Worker ativo: $([bool](-not -not ($workerProcesses.Count)))"
Write-Host "Web ativo: $([bool](-not -not ($webProcesses.Count)))"
