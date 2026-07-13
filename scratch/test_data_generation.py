import sys
import os
sys.path.append(os.getcwd())
from tools.self_play.v11.six_max_simulator import SixMaxSimulator

def trace_mcts():
    print("--- Tracing Nit ---")
    sim_nit = SixMaxSimulator(bb_size=10.0, hero_personality='main')
    # Force the simulator to play a specific hand scenario
    # We will just generate 1 hand and print the target EVs for the decisions
    sim_nit.focus_archetype = 'Nit'
    record_nit = sim_nit.play_hand()
    for dp in record_nit.decision_points:
        if dp['hero_position'] is not None:
            print(f"Street {dp['street']}, Pot: {dp['pot_size']}, Action: {dp['action']}, EVs: {dp['target_evs']}")
            
    print("\n--- Tracing Maniac ---")
    sim_maniac = SixMaxSimulator(bb_size=10.0, hero_personality='main')
    sim_maniac.focus_archetype = 'Maniac'
    record_maniac = sim_maniac.play_hand()
    for dp in record_maniac.decision_points:
        if dp['hero_position'] is not None:
            print(f"Street {dp['street']}, Pot: {dp['pot_size']}, Action: {dp['action']}, EVs: {dp['target_evs']}")

if __name__ == "__main__":
    trace_mcts()
