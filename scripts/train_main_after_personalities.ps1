while (Get-CimInstance Win32_Process -Filter "CommandLine LIKE '%train_selfplay.py%'") {
    Start-Sleep -Seconds 10
}
Write-Host "========================================="
Write-Host "Personality training finished."
Write-Host "Starting V10 Main Model Training (250k hands)"
Write-Host "========================================="
.venv\Scripts\python.exe tools\self_play\v10\train_selfplay.py --personality main --num_hands 250000 | Tee-Object -FilePath "active_training.log"
