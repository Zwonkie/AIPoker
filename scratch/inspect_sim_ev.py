import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.self_play.train_selfplay import create_opponent_pool, simulate_worker
from multiprocessing import Pool

def main():
    opponent_pool = create_opponent_pool()
    num_hands = 100
    equity_sims = 200
    
    sim = simulate_worker(opponent_pool, 10.0, 10, 400, equity_sims, num_hands)
    
    print("=== INSPECTING SIMULATED DECISION POINTS ===")
    count = 0
    for rec in sim:
        for dp in rec.get_training_samples():
            print(f"Hand {rec.hand_id} | Street {dp['street']} | Equity {dp['equity']:.3f} | Call {dp['call_amount']:.1f} | Pot {dp['pot_size']:.1f} | Action {dp['action']} | Target EV {dp['target_ev']:.2f}")
            count += 1
            if count >= 20:
                break
        if count >= 20:
            break

if __name__ == '__main__':
    main()
