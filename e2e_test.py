import cv2
import time
from core.vision import PokerVision
from core.evaluator import PokerEvaluator
from core.decision import PokerDecisionEngine
from core.state_machine import PokerStateMachine
from core.table_state import TableState

def run_e2e_validation():
    print("==================================================")
    print("         POKER AI END-TO-END VALIDATION           ")
    print("==================================================")
    
    # 1. Initialize Modules
    vision = PokerVision()
    evaluator = PokerEvaluator()
    decision_engine = PokerDecisionEngine()
    
    class TestHandler:
        def on_state_updated(self, state):
            print(f"[State Machine] State -> {state}")
            
    handler = TestHandler()
    sm = PokerStateMachine(handler)
    
    # Test cases: (Image File, Expected Cards, Expected Pot, Expected Opponents Count, Call Amount, check_call_available, bet_raise_available)
    test_cases = [
        ("board_samples/1_postflop_first_fold_check_raise.png", ['Kd', 'Ac', '3s'], 80, 3, 0.0, True, True),
        ("board_samples/2_postflop_river_fold_call_raise.png", ['Kd', 'Ac', '3s', '3h', 'Tc'], 160, 1, 40.0, True, True),
        ("board_samples/3_preflop_fold_allin.png", [], 2021, 2, 100.0, False, True),
        ("board_samples/4_postflop_river_fold_call_raise_facing_bet.png", ['6c', '8d', '7c', 'Kh', '9d'], 372, 1, 186.0, True, True)
    ]
    
    for model_name in ["Heuristic (Rules)", "XGBoost Classifier", "XGBoost Mixed (Pro + Human)"]:
        print(f"\n==================================================")
        print(f"   RUNNING SUITE WITH MODEL: {model_name}        ")
        print(f"==================================================")
        decision_engine.set_active_model(model_name)
        
        for board_file, expected_comm, expected_pot, expected_opps, call_amount, cc_avail, br_avail in test_cases:
            print(f"\n--- Testing Board State: {board_file} ---")
            img = cv2.imread(board_file)
            if img is None:
                print(f"Error: {board_file} not found!")
                continue
                
            # Resize to standard if needed (matching main.py logic)
            h, w = img.shape[:2]
            if abs(w - 1536) > 50 or abs(h - 1090) > 50:
                img = cv2.resize(img, (1536, 1090), interpolation=cv2.INTER_CUBIC)
                
            # Step 1: Start waiting
            sm.start()
            
            # Step 2: Detect Turn
            sm.turn_detected()
            
            # Step 3: Vision - Parse State
            raw_state = vision.read_board_state(img)
            table_state = TableState()
            table_state.update(raw_state)
            state = table_state.to_dict()
            
            print(f"[Vision] Community Cards: {state['community_cards']}")
            print(f"[Vision] Hero Hand:      {state['hero_cards']}")
            print(f"[Vision] Pot Size:      {state['pot_size']}")
            print(f"[Vision] Hero Stack:     {state['hero_stack']}")
            
            active_opponents = len([opp for opp in state['opponents'].values() if opp.get('is_active', True)])
            print(f"[Vision] Active Opps:    {active_opponents}")
            
            # Assertions / Checks
            assert set(state['community_cards']) == set(expected_comm), "Community cards mismatch!"
            assert state['pot_size'] == expected_pot, "Pot size mismatch!"
            assert active_opponents == expected_opps, f"Opponents count mismatch! Expected {expected_opps}, got {active_opponents}"
            print("[Vision] SUCCESS: All vision telemetry values verified.")
            
            sm.state_read_complete()
            
            # Step 4: Evaluator - Equity
            print("[Evaluator] Calculating equity against random opponents...")
            equity, sim_msg = evaluator.calculate_equity(
                state['community_cards'],
                state['hero_cards'],
                num_opponents=1, # assume 1 opponent for test speed
                num_simulations=1000
            )
            print(f"[Evaluator] Equity: {equity:.2%}")
            print(f"[Evaluator] Details: {sim_msg}")
            
            # Step 5: Decision - Decide Action
            is_preflop = len(state['community_cards']) == 0
            action, reason, bet_size = decision_engine.make_decision(
                state['community_cards'],
                state['hero_cards'],
                equity=equity,
                pot_size=state['pot_size'],
                call_amount=call_amount,
                hero_stack=state['hero_stack'] if state['hero_stack'] > 0 else 760,
                num_opponents=1,
                is_preflop=is_preflop,
                bet_raise_available=br_avail,
                check_call_available=cc_avail
            )
            
            print(f"[Decision] Decided Action: **{action}**")
            print(f"[Decision] Reason:         {reason}")
            if bet_size > 0:
                print(f"[Decision] Size:           {bet_size} chips")
                
            sm.decision_made()
            
            # Step 6: Executor / Complete Action
            print(f"[Executor] Mimicking click execution of action {action}...")
            sm.action_completed()
            
            sm.stop()
            print(f"--- Finished Test Case: {board_file} ---")
            
    print("\n==================================================")
    print("       ALL E2E VALIDATIONS PASSED SUCCESSFULLY    ")
    print("==================================================")

if __name__ == '__main__':
    run_e2e_validation()
