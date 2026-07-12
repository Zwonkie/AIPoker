.venv\Scripts\python.exe tools\self_play\v10\train_selfplay.py --num_hands 50000 --save_name v10_50k_main.pt
Copy-Item core\weights\v10_50k_main.pt core\weights\v10_100k_main.pt
.venv\Scripts\python.exe tools\self_play\v10\train_selfplay.py --num_hands 50000 --resume_path core\weights\v10_50k_main.pt --save_name v10_100k_main.pt --hands_done 50000
