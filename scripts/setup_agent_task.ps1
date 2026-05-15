# Sets up the paper-trader agent as a Windows scheduled task that:
#   - Runs at user logon
#   - Restarts automatically if it crashes
#   - Has no visible window
#   - Persists across reboots
#
# Usage:
#   1. Right-click PowerShell → Run as Administrator
#   2. cd to this folder
#   3. ./setup_agent_task.ps1
#
# To remove: Unregister-ScheduledTask -TaskName "PaperTraderAgent" -Confirm:$false

$ErrorActionPreference = "Stop"

$TaskName = "PaperTraderAgent"
$RepoRoot = (Resolve-Path "$PSScriptRoot\..").Path
$PythonExe = (Get-Command pythonw.exe -ErrorAction SilentlyContinue).Path
if (-not $PythonExe) {
    Write-Warning "pythonw.exe not found; falling back to python.exe " +
                  "(will show a console window)."
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

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Principal $Principal `
    -Description "Paper-trades HOU/HOD/HNU/HND with logged signals" | Out-Null

Write-Host ""
Write-Host "✅ Scheduled task '$TaskName' registered."
Write-Host ""
Write-Host "Useful commands:"
Write-Host "  Start now : Start-ScheduledTask -TaskName $TaskName"
Write-Host "  Stop      : Stop-ScheduledTask  -TaskName $TaskName"
Write-Host "  Status    : Get-ScheduledTask   -TaskName $TaskName"
Write-Host "  Remove    : Unregister-ScheduledTask -TaskName $TaskName -Confirm:`$false"
Write-Host ""
Write-Host "Starting it now…"
Start-ScheduledTask -TaskName $TaskName
Write-Host "✅ Started. Check logs/agent.log for activity."
