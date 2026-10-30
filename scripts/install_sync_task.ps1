param(
    [string]$Time = '18:30',
    [string]$Python = (Join-Path (Split-Path -Parent $PSScriptRoot) '.venv/Scripts/python.exe')
)
$ErrorActionPreference = 'Stop'
$parsedTime = [datetime]::ParseExact($Time, 'HH:mm', [cultureinfo]::InvariantCulture)
$projectPath = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if (-not (Test-Path -LiteralPath $Python)) { throw 'Run uv sync --locked before installing the scheduled task.' }
$Python = (Resolve-Path -LiteralPath $Python).Path
$windowlessPython = Join-Path (Split-Path -Parent $Python) 'pythonw.exe'
if (Test-Path -LiteralPath $windowlessPython) { $Python = $windowlessPython }
# Explicit +08:00 start boundary: schedule stays in Shanghai time even if the
# machine's timezone changes. StartWhenAvailable handles sleep/missed starts.
$startBoundary = (Get-Date -Format 'yyyy-MM-dd') + 'T' + $parsedTime.ToString('HH:mm') + ':00+08:00'
$action = New-ScheduledTaskAction -Execute $Python -Argument "-m market_research.tasks.sync --once --time $Time" -WorkingDirectory $projectPath
$trigger = New-ScheduledTaskTrigger -Daily -At $parsedTime
$trigger.StartBoundary = $startBoundary
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 12) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$userId = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$principal = New-ScheduledTaskPrincipal -UserId $userId -LogonType Interactive -RunLevel Limited
$taskName = 'MarketResearch-DailySync'
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description 'Synchronize only registered Market Research assets daily.' -Force | Out-Null
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName, State

# Remove this workspace's old registration only after the new one is installed.
$legacyTask = Get-ScheduledTask -TaskName 'MarketLens-DailySync' -ErrorAction SilentlyContinue
if ($legacyTask -and @($legacyTask.Actions | Where-Object { $_.WorkingDirectory -eq $projectPath }).Count -gt 0) {
    Unregister-ScheduledTask -TaskName $legacyTask.TaskName -Confirm:$false
}
