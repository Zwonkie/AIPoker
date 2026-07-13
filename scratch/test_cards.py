import sys
import os
sys.path.append(os.getcwd())
from core.decision import PokerDecisionEngine
from core.board_state import BoardState, SeatState, HUDStats

def create_mock_board_state(street, hero_cards, community_cards, pot, call, stack, equity, num_opps, opp_profile='Standard'):
    seats = {}
    for i in range(1, num_opps + 1):
        seats[f'seat_{i}'] = SeatState(is_active=True, stack=stack, hud=HUDStats(vpip_color='Green', agg_color='Green'))
    return BoardState(hero_cards=hero_cards, community_cards=community_cards, pot_size=pot, hero_stack=stack, call_amount=call, seats=seats, big_blind=10.0, street=street, equity=equity)

engine = PokerDecisionEngine()
engine.set_active_model('herocules_v11_fuzzyHeuristicsOpp.pth')

bs = create_mock_board_state('Preflop', ['7d', '2c'], [], 15.0, 10.0, 1000.0, 0.15, 1, 'Standard')
action, reason, bet_size, ev_dict = engine.make_decision(bs, use_math_engine=False)
print('72o vs 1 opp:', ev_dict)

bs = create_mock_board_state('Preflop', ['Ah', 'As'], [], 15.0, 10.0, 1000.0, 0.85, 1, 'Standard')
action, reason, bet_size, ev_dict = engine.make_decision(bs, use_math_engine=False)
print('AA vs 1 opp:', ev_dict)

