# Sets up the paper-trader agent as a Windows scheduled task that:
#   - Runs at user logon
#   - Restarts automatically if it crashes
#   - Has no visible window
#   - Persists across reboots
#
# Usage:
#   1. Right-click PowerShell -> Run as Administrator
#   2. cd to this folder
#   3. ./setup_agent_task.ps1
#
# To remove: Unregister-ScheduledTask -TaskName "PaperTraderAgent" -Confirm:$false
#
# NOTE: This file is intentionally ASCII-only. Windows PowerShell 5.1 reads
# .ps1 files as Windows-1252 by default; non-ASCII characters (emoji, smart
# quotes, en/em dashes) silently corrupt and break string parsing.

$ErrorActionPreference = "Stop"

# --- Require admin: Task Scheduler registration needs elevation ---
$IsAdmin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()
).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $IsAdmin) {
    Write-Host ""
    Write-Host "[ERROR] This script must run as Administrator." -ForegroundColor Red
    Write-Host ""
    Write-Host "Close this window, then:"
    Write-Host "  1. Start menu -> search 'PowerShell'"
    Write-Host "  2. Right-click -> Run as administrator"
    Write-Host "  3. cd `"$((Resolve-Path "$PSScriptRoot\..").Path)`""
    Write-Host "  4. .\scripts\setup_agent_task.ps1"
    Write-Host ""
    Write-Host "Or use the Startup-folder alternative (no admin needed)"
    Write-Host "see docs/paper_trader_setup.md for the .bat-shortcut method."
    exit 1
}

$TaskName = "PaperTraderAgent"
$RepoRoot = (Resolve-Path "$PSScriptRoot\..").Path
$PythonExe = (Get-Command pythonw.exe -ErrorAction SilentlyContinue).Path
if (-not $PythonExe) {
    Write-Warning "pythonw.exe not found; falling back to python.exe (will show a console window)."
    $PythonExe = (Get-Command python.exe).Path
}

Write-Host "Project root: $RepoRoot"
Write-Host "Python exe  : $PythonExe"

# Build the task. RunLevel Limited = no UAC prompt.
$Action = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument "-m paper_trader.agent" `
    -WorkingDirectory $RepoRoot

$Trigger = New-ScheduledTaskTrigger -AtLogOn

# Restart on failure, no idle requirement, run on battery, no time limit
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -DontStopOnIdleEnd `
    -ExecutionTimeLimit (New-TimeSpan -Days 0) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 5) `
    -StartWhenAvailable

# Run as the current user, on logon
$Principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Limited

# Remove any existing version first
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue

try {
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $Action `
        -Trigger $Trigger `
        -Settings $Settings `
        -Principal $Principal `
        -Description "Paper-trades HOU/HOD/HNU/HND with logged signals" `
        -ErrorAction Stop | Out-Null
}
catch {
    Write-Host ""
    Write-Host "[ERROR] Registration failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[OK] Scheduled task '$TaskName' registered."
Write-Host ""
Write-Host "Useful commands:"
Write-Host "  Start now : Start-ScheduledTask -TaskName $TaskName"
Write-Host "  Stop      : Stop-ScheduledTask  -TaskName $TaskName"
Write-Host "  Status    : Get-ScheduledTask   -TaskName $TaskName"
Write-Host "  Remove    : Unregister-ScheduledTask -TaskName $TaskName -Confirm:`$false"
Write-Host ""
Write-Host "Starting it now..."
Start-ScheduledTask -TaskName $TaskName
Write-Host "[OK] Started. Check logs/agent.log for activity."
