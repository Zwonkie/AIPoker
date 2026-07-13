import torch
import sys
import os

# Add root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from versions.v12.core.contract import ContractV12
from versions.v12.self_play.train import vectorize_hand_samples
from versions.v12.self_play.simulator import HandRecordV4
from core.board_state import BoardState, SeatState, HUDStats

def test_v12_alignment():
    # Construct a synthetic hand history and decision points
    hero_cards = ['Ah', 'Kh']
    opp_profiles = {
        'seat_1': {'vpip': 0.3, 'agg': 0.4, 'style': 'tag'}
    }
    
    # Create HandRecord for train.py
    record = HandRecordV4(hand_id=1, hero_cards=hero_cards, opponents_profiles=opp_profiles)
    
    board1 = []
    board2 = ['2c', '3c', '4c']
    
    # First action preflop
    record.add_decision(
        step=0, street=0, board=board1, hero_position=0, pot_size=30.0, big_blind=10.0,
        call_amount=10.0, hero_stack=1000.0, active_opponents_mask=[1.0, 0.0, 0.0, 0.0, 0.0],
        opponents_stacks=[1000.0, 0.0, 0.0, 0.0, 0.0], action_history=[], equity=0.65,
        action_taken=1, chips_committed_before=10.0, target_evs=[0.0, 1.0, 2.0],
        opp_strength=0.5, opp_bluff_prob=0.1
    )
    
    # Second action flop
    record.add_decision(
        step=1, street=1, board=board2, hero_position=0, pot_size=50.0, big_blind=10.0,
        call_amount=0.0, hero_stack=990.0, active_opponents_mask=[1.0, 0.0, 0.0, 0.0, 0.0],
        opponents_stacks=[990.0, 0.0, 0.0, 0.0, 0.0], action_history=['c'], equity=0.75,
        action_taken=2, chips_committed_before=20.0, target_evs=[0.0, 1.0, 5.0],
        opp_strength=0.3, opp_bluff_prob=0.0
    )
    
    # Generate Training Tensors
    samples = vectorize_hand_samples(record, max_seq_len=20)
    train_h, train_b, train_c, train_a, train_sa, train_mc, train_mask, train_ob, train_os, train_eq, train_w, train_pol = samples[0]
    
    train_h_t = torch.tensor([train_h], dtype=torch.long)
    train_b_t = torch.tensor([train_b], dtype=torch.long)
    train_c_t = torch.tensor([train_c], dtype=torch.float32)
    train_a_t = torch.tensor([train_a], dtype=torch.long)
    
    # Generate Inference Tensors via Contract
    states = []
    
    s1 = BoardState(
        community_cards=board1, hero_cards=hero_cards, pot_size=30.0, hero_stack=1000.0,
        big_blind=10.0, call_amount=10.0, equity=0.65, hero_position=0, street="Preflop"
    )
    s1.seats['seat_1'] = SeatState("Opp 1", stack=1000.0, is_active=True, hud=HUDStats(vpip_color="Yellow", agg_color="Green"))
    
    s2 = BoardState(
        community_cards=board2, hero_cards=hero_cards, pot_size=50.0, hero_stack=990.0,
        big_blind=10.0, call_amount=0.0, equity=0.75, hero_position=0, street="Flop"
    )
    s2.seats['seat_1'] = SeatState("Opp 1", stack=990.0, is_active=True, hud=HUDStats(vpip_color="Yellow", agg_color="Green"))
    
    states = [s1, s2]
    # hero actions: taken AT those states. In vectorizer, 'action_taken' is 1(call) and 2(raise).
    # In ContractV12, act_ints expects vocab indexes: 3=call, 6=raise.
    hero_actions = [3, 6]
    
    bridge = ContractV12(max_seq_len=20)
    inf_h_t, inf_b_t, inf_c_t, inf_a_t = bridge.to_tensors(states, hero_actions=hero_actions)
    
    # Verify Alignment
    assert torch.equal(train_h_t, inf_h_t), "Hole cards differ"
    assert torch.equal(train_b_t, inf_b_t), "Board cards differ"
    
    # The context vector is float, so we use allclose
    if not torch.allclose(train_c_t, inf_c_t, atol=1e-3):
        diff = torch.abs(train_c_t - inf_c_t)
        max_diff = torch.max(diff)
        print(f"Max Context Diff: {max_diff}")
        for i in range(train_c_t.shape[1]):
            for j in range(train_c_t.shape[2]):
                if diff[0, i, j] > 1e-3:
                    print(f"Diff at pos [{i}, {j}]: train={train_c_t[0, i, j]}, inf={inf_c_t[0, i, j]}")
        assert False, "Context vectors differ"
    
    assert torch.equal(train_a_t, inf_a_t), f"Action sequences differ:\nTrain: {train_a_t}\nInf:   {inf_a_t}"
    
    # Mask alignment test is removed since key_padding_mask was intentionally dropped
    
    print("V12 Alignment Test: PASSED")

if __name__ == '__main__':
    test_v12_alignment()
