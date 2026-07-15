import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from versions.v14.core.model import PokerEVModelV4
from versions.v14.self_play.simulator import SixMaxSimulator
from versions.v14.core.manifest import MANIFEST
from shared.manifest import load_state_dict as load_ckpt_state

def evaluate_model(model_path: str, num_hands: int = 500, bb_size: float = 10.0, equity_sims: int = 200):
    print("=" * 60)
    print("  V12 GAMEPLAY EVALUATION HARNESS")
    print("=" * 60)
    print(f"  Model Path:   {model_path}")
    print(f"  Target Hands: {num_hands:,}")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model weights not found at {model_path}")
        return
        
    print("Loading model...")
    model = PokerEVModelV4().to(device)
    try:
        model.load_state_dict(load_ckpt_state(model_path, MANIFEST, map_location=device))
        model.eval()
    except Exception as e:
        print(f"FATAL: could not load model {model_path}: {e}")
        return
        
    print("Initializing simulator...")
    sim = SixMaxSimulator(
        bb_size=bb_size, 
        equity_sims=equity_sims, 
        hero_personality='main',
        bootstrap_alpha=0.0  # Pure model, no exploration/heuristics
    )
    sim.hero_model = model
    
    print(f"\nRunning {num_hands} hands against heuristic lineup...")
    
    records = []
    
    for i in range(num_hands):
        if (i + 1) % 50 == 0:
            print(f"Simulating hand {i + 1}/{num_hands}...")
            
        rec = sim.simulate_hand(current_hand=100000 + i) # Large number to bypass phase 1 stack limits
        if rec and rec.decision_points:
            records.append(rec)
            
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    
    total_hands = len(records)
    print(f"Hands completed: {total_hands}")
    
    # Hero is always seat 0 in seat_histories
    hero_stats = sim.seat_histories[0]
    
    profit = hero_stats['profit']
    bb100 = (profit / max(1.0, total_hands)) * 10.0
    
    vpip_ops = hero_stats['vpip_ops']
    vpip_acts = hero_stats['vpip_acts']
    vpip = (vpip_acts / max(1, vpip_ops)) * 100.0
    
    agg_ops = hero_stats['agg_ops']
    agg_acts = hero_stats['agg_acts']
    agg = (agg_acts / max(1, agg_ops)) * 100.0
    
    raises = hero_stats['raises']
    folds = hero_stats['folds']
    all_ins = hero_stats['all_ins']
    
    print(f"Win Rate:        {bb100:+.2f} BB/100")
    print(f"VPIP:            {vpip:.1f}%")
    print(f"AGG:             {agg:.1f}%")
    print(f"Actions taken:   Raises: {raises}, Folds: {folds}, All-ins: {all_ins}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a V12 model's gameplay performance.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint to evaluate")
    parser.add_argument("--hands", type=int, default=500, help="Number of hands to simulate")
    
    args = parser.parse_args()
    evaluate_model(args.model_path, num_hands=args.hands)
