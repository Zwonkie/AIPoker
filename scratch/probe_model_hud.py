import sys
import os
sys.path.append(os.getcwd())
import torch
from core.models.engine import ModelEngine
from core.bridge.v11.contract_v11 import ContractV8V9

def probe_hud():
    engine = ModelEngine(expert_name="herocules_v11_fuzzyHeuristicsOpp.pth", is_v11=True)
    bridge = ContractV8V9()
    
    # Base scenario: Hero has KK, Flop is T 7 2.
    from core.board_state import BoardState, SeatState, HUDStats
    
    print(f"{'VPIP'.ljust(8)} | {'AGG'.ljust(8)} | {'EV(Fold)'.ljust(10)} | {'EV(Call)'.ljust(10)} | {'EV(Raise)'.ljust(10)}")
    print("-" * 55)
    
    # We will manually construct the tensors so we can inject exact VPIP/AGG values
    bs = BoardState(
        hero_cards=['Ks', 'Kd'],
        community_cards=['Th', '7c', '2d'],
        pot_size=50.0,
        hero_stack=1000.0,
        call_amount=10.0,
        seats={
            'seat_1': SeatState(is_active=True, stack=1000.0, hud=HUDStats(vpip_color='Green', agg_color='Green'))
        },
        big_blind=10.0,
        street="Flop",
        equity=0.80
    )
    
    h_t, b_t, c_t, a_t = bridge.to_tensors(bs)
    
    # Check the context sequence
    for vpip in [0.0, 0.2, 0.5, 0.8, 1.0]:
        for agg in [0.0, 0.5, 1.0]:
            # Modify the context vector for the LAST step
            c_t[0, -1, 13] = vpip
            c_t[0, -1, 14] = agg
            
            with torch.no_grad():
                out = engine.model(h_t.to(engine.device), b_t.to(engine.device), c_t.to(engine.device), a_t.to(engine.device))
                q = out['q_vals'][0, -1, :].cpu().numpy()
                
            print(f"{vpip:<8.2f} | {agg:<8.2f} | {q[0]:<10.2f} | {q[1]:<10.2f} | {q[2]:<10.2f}")

if __name__ == "__main__":
    probe_hud()
