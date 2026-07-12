Write-Host "========================================="
Write-Host "1/3: Training MANIAC Personality"
Write-Host "========================================="
.\.venv\Scripts\python.exe tools\self_play\v11\train_selfplay.py --personality maniac | Tee-Object -FilePath "active_training.log"

Write-Host "========================================="
Write-Host "2/3: Training NIT Personality"
Write-Host "========================================="
.\.venv\Scripts\python.exe tools\self_play\v11\train_selfplay.py --personality nit | Tee-Object -FilePath "active_training.log"

Write-Host "========================================="
Write-Host "3/3: Training STICKY Personality"
Write-Host "========================================="
.\.venv\Scripts\python.exe tools\self_play\v11\train_selfplay.py --personality sticky | Tee-Object -FilePath "active_training.log"

Write-Host "All personality trainings complete!"
