Write-Host "Starting AIPoker Telemetry Dashboard Server..."
Start-Process "http://localhost:8080/dashboard.html"

# Contract (enforced by check_telemetry_contract.py, documented in .agents/AGENTS.md):
# every versions/*/self_play/train.py tees stdout to this single, fixed, repo-root path,
# regardless of which version is training. The watcher trusts that contract instead of
# guessing via a recursive log search.
$repoDir = "c:\REPO\Antigravity\AIPoker"
$targetLog = "$repoDir\active_training.log"

# Start a background job to continuously parse the active training log every 5 seconds
$watcherScript = {
    param($repoDir, $targetLog)
    while ($true) {
        if (Test-Path $targetLog) {
            & "$repoDir\.venv\Scripts\python.exe" "$repoDir\tools\training_monitor\parse_training_log.py" $targetLog | Out-Null
        }
        Start-Sleep -Seconds 5
    }
}
$job = Start-Job -ScriptBlock $watcherScript -ArgumentList $repoDir, $targetLog -Name "TelemetryWatcher"

Write-Host "Background telemetry parser started (watching $targetLog)."
try {
    & "$repoDir\.venv\Scripts\python.exe" -m http.server 8080 -d "$repoDir\tools\training_monitor"
} finally {
    Stop-Job -Job $job
    Remove-Job -Job $job
}
