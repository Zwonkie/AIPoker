# V11 Self-Play Training Run
# Driven primarily by training_config.yaml

echo "Starting V11 Training (Main Personality)"
.venv\Scripts\python.exe tools\self_play\v11\train_selfplay.py --personality main --save_name expert_v11_main.pth
