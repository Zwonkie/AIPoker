Write-Host "Starting AIPoker Telemetry Dashboard Server..."
Start-Process "http://localhost:8080/dashboard.html"

# Start a background job to continuously parse the active training log every 5 seconds
$watcherScript = {
    $repoDir = "c:\REPO\Antigravity\AIPoker"
    while ($true) {
        $targetLog = "$repoDir\active_training.log"
        if (-Not (Test-Path $targetLog)) {
            $latest = Get-ChildItem "$repoDir\training_*.log" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
            if ($latest) { $targetLog = $latest.FullName }
        }
        if (Test-Path $targetLog) {
            & "$repoDir\.venv\Scripts\python.exe" "$repoDir\.agents\skills\monitor-training-session\scripts\parse_training_log.py" $targetLog | Out-Null
        }
        Start-Sleep -Seconds 5
    }
}
$job = Start-Job -ScriptBlock $watcherScript -Name "TelemetryWatcher"

Write-Host "Background telemetry parser started (Watching active_training.log)."
try {
    .\.venv\Scripts\python.exe -m http.server 8080 -d .agents\skills\monitor-training-session\scripts
} finally {
    Stop-Job -Job $job
    Remove-Job -Job $job
}
