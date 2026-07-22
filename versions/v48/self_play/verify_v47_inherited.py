"""V47 pre-training verification probes -- the SPECS' "verification before training" items that
are unit-testable (C1/C2 have their own scripts: calibrate_raise_sizes.py / instrument_c2.py).

Covers:
  [M9]   chip-identical bucket collapse: EV unification + aliased flags at a clamped node,
         no-op at a deep node, actor-target mass concentration (dataset-time and torch paths).
  [M4]   occupant-true fold models: analytic Fuzzy closed form == empirical mean of the sampled
         path; Tree fold_prob validity; NN batched fold probs validity on real V44 weights.
  [CURR] stack_depth_mix distribution check for the new 2-5bb / 5-8bb bands.
  [VAL-5] checkpoint save -> resume round-trip restores optimizer + scheduler state exactly.
  [P0.4] contract_version hard validation raises on mismatch, opt-in allows.
  [M9-serve] the serve-side mirror exists (v47_engine declares collapse_aliased_allin; the
         decision.py sampler reads it) and the clamping precondition matches train-side sizing.

Run:  .venv/Scripts/python.exe versions/v48/self_play/verify_v47.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import torch

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    ok = bool(cond)
    PASS += ok
    FAIL += (not ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def main():
    random.seed(474747)
    torch.manual_seed(474747)

    from versions.v48.self_play.simulator import SixMaxSimulator, HandRecordV4
    from versions.v48.self_play.opponents import HeuristicOpponent, NNOpponent
    from versions.v48.self_play.tree_opponent import TreeOpponent
    from versions.v48.self_play import train as v47_train
    from versions.v48.self_play.opponent_bots import (FuzzyPlayerArchetype, TAG, NIT,
                                                      sample_raise_fraction,
                                                      raise_size_distribution_for)
    from shared.registry import load_model, get_manifest
    from shared.manifest import save_checkpoint, load_state_dict

    sim = SixMaxSimulator(bb_size=10.0, equity_sims=60, hero_personality='main', bootstrap_alpha=0.0)
    sim.range_aware_equity = False

    import copy
    def opp_entry(seat, cards, stack, bot=None):
        bot = bot or copy.deepcopy(TAG)
        bot.start_new_hand()
        return {'bot': bot, 'agent': HeuristicOpponent('tag', bot), 'stack': stack,
                'cards': cards, 'seat': seat}

    print("== [M9] chip-identical bucket collapse ==")
    # Clamped node: pot 100, to_call 0, hero stack 30 -> every raise bucket resolves to 30 chips.
    opps = [opp_entry(1, ['Qd', 'Jc'], 300.0)]
    evs, _, _ = sim._mc_target_evs_sized(['As', 'Ks'], pot=100.0, to_call=0.0, hero_stack=30.0,
                                         street_idx=0, active_opponents=opps, board_str=[],
                                         raise_fracs=sim.raise_fracs)
    al = sim.last_sized_aliased
    check("clamped node: raise_33/66/pot all flagged aliased", al == [True, True, True, False], f"aliased={al}")
    check("clamped node: aliased EVs EXACTLY equal ALLIN's",
          evs[2] == evs[5] and evs[3] == evs[5] and evs[4] == evs[5],
          f"evs={['%.3f' % e for e in evs]}")

    # Deep node: nothing clamps.
    evs_d, _, _ = sim._mc_target_evs_sized(['As', 'Ks'], pot=100.0, to_call=0.0, hero_stack=1000.0,
                                           street_idx=0, active_opponents=opps, board_str=[],
                                           raise_fracs=sim.raise_fracs)
    check("deep node: no aliasing", sim.last_sized_aliased == [False, False, False, False])
    check("deep node: sizes stay distinct EVs", len({round(e, 6) for e in evs_d[2:]}) > 1,
          f"evs={['%.3f' % e for e in evs_d]}")

    # Actor-target collapse, dataset-time path.
    vals = [0.0, 1.0, 2.5, 2.5, 2.5, 2.5]           # aliased buckets tie ALLIN exactly
    al_full = [False, False, True, True, True, False]
    pol = v47_train.regret_match_policy(vals, aliased=al_full)
    check("regret_match_policy: aliased buckets get 0 mass",
          pol[2] == pol[3] == pol[4] == 0.0, f"pol={['%.3f' % p for p in pol]}")
    check("regret_match_policy: ALLIN concentrates the jam mass",
          pol[5] > 0.5 and abs(sum(pol) - 1.0) < 1e-9, f"p_allin={pol[5]:.3f}")
    pol_no = v47_train.regret_match_policy(vals)
    check("regret_match_policy: WITHOUT mask the same EVs split 4 ways (the bug being fixed)",
          abs(pol_no[5] - pol[5] / 4.0) < 1e-9, f"unmasked p_allin={pol_no[5]:.3f}")

    # Actor-target collapse, torch path.
    q = torch.tensor([[[0.0, 1.0, 2.5, 2.5, 2.5, 2.5]]])
    mask = torch.tensor([[[False, False, True, True, True, False]]])
    out = v47_train.regret_match_policy_torch(q, baseline_mode='fold', aliased_mask=mask)[0, 0]
    check("regret_match_policy_torch: aliased zero + ALLIN concentrated",
          float(out[2]) == 0.0 and float(out[3]) == 0.0 and float(out[4]) == 0.0
          and float(out[5]) > 0.5, f"target={[round(float(x), 3) for x in out]}")

    print("== [M4] occupant-true fold models ==")
    # Analytic Fuzzy closed form == empirical mean of the sampled decision (same distribution).
    bot = copy.deepcopy(NIT)
    bot.start_new_hand()
    for eq, po, ia in ((0.42, 0.40, False), (0.42, 0.40, True), (0.55, 0.33, False), (0.20, 0.50, True)):
        analytic = sim._heuristic_fold_prob(bot, eq, po, 1, ia)
        n = 20000
        emp = sum(sim._ev_target_fold_decision(bot, eq, po, 1, ia) for _ in range(n)) / n
        check(f"analytic == E[sampled] (eq={eq}, po={po}, allin={ia})",
              abs(analytic - emp) < 0.02, f"analytic={analytic:.3f} empirical={emp:.3f}")

    tree = TreeOpponent(0, recording_bot=copy.deepcopy(TAG))
    tp = tree.fold_prob(0.35, 0.45, 1, 200.0, 2)
    check("tree fold_prob valid", tp is not None and 0.0 <= tp <= 1.0, f"p_fold={tp}")

    model = load_model('v48', 'frozen_v47.pth')
    nn_opp = {'agent': NNOpponent('past', model, None, None, recording_bot=copy.deepcopy(TAG)),
              'seat': 2, 'cards': ['Qd', 'Jc'], 'stack': 300.0,
              'model_state_history': [], 'hero_actions_history': []}
    ts = {'board': [], 'street': 0, 'opponents_profiles': {'seat_2': {'vpip': 0.22, 'agg': 0.45}},
          'actor_seat': 0, 'button_seat': 0, 'hand_strength': 0.5, 'effective_field': 1.0,
          'committed': [0.0, 0.0, 10.0, 0.0, 0.0, 0.0], 'raise_count': 0,
          'raised_this_hand': [False] * 6, 'raised_this_street': [False] * 6,
          'folded': [False, True, False, True, True, True],
          'stacks': [300.0, 0.0, 300.0, 0.0, 0.0, 0.0]}
    sizes = [(33.0, 33.0, False), (66.0, 66.0, False), (100.0, 100.0, False), (300.0, 300.0, True)]
    ps = sim._nn_fold_probs_for_sizes(nn_opp, sizes, pot=30.0, table_state_dict=ts,
                                      num_opps_for_query=1, opp_equity=0.45, opp_hand_strength=0.4)
    check("NN batched fold probs: 4 valid probabilities",
          ps is not None and len(ps) == 4 and all(0.0 <= p <= 1.0 for p in ps),
          f"p_fold per size={[round(p, 3) for p in ps or []]}")
    check("NN fold probs: history NOT mutated by hypothetical queries",
          nn_opp['model_state_history'] == [])

    print("== [C1 wiring] archetype size repertoires ==")
    check("style mapping resolves (maniac->LAG, fish->Calling Station)",
          raise_size_distribution_for('maniac') is raise_size_distribution_for('LAG')
          and raise_size_distribution_for('fish') is raise_size_distribution_for('Calling Station'))
    draws = [sample_raise_fraction('maniac') for _ in range(4000)]
    # [V48, Change 1b] Expectations updated to the FITTED tables (population fit, hero
    # excluded): LAG postflop jam weight 0.058; Calling Station DOES jam in the measured
    # population (postflop 0.100, preflop 0.33) -- the old "never jams" was a hand-authored
    # fiction the fit corrected.
    jam_rate = sum(1 for d in draws if d is None) / len(draws)
    check("LAG repertoire samples jams near fitted 5.8%", 0.03 < jam_rate < 0.10, f"jam_rate={jam_rate:.3f}")
    st_draws = [sample_raise_fraction('fish') for _ in range(2000)]
    st_jam = sum(1 for d in st_draws if d is None) / len(st_draws)
    check("station repertoire jams near fitted 10%", 0.05 < st_jam < 0.16, f"jam_rate={st_jam:.3f}")
    pf_draws = [sample_raise_fraction('fish', street_idx=0) for _ in range(2000)]
    pf_jam = sum(1 for d in pf_draws if d is None) / len(pf_draws)
    check("station PREFLOP repertoire jams near fitted 33% (street split live)", 0.26 < pf_jam < 0.41,
          f"jam_rate={pf_jam:.3f}")

    print("== [CURR] stack_depth_mix bands ==")
    sim2 = SixMaxSimulator(bb_size=10.0, equity_sims=60, hero_personality='main', bootstrap_alpha=0.0)
    sim2.stack_depth_mix = [[2, 5, 0.08], [5, 8, 0.07], [5, 14, 0.35], [14, 30, 0.25],
                            [30, 60, 0.17], [10, 100, 0.08]]
    stacks_bb = [sim2._get_starting_stack(50000) / 10.0 for _ in range(5000)]
    frac_sub5 = sum(1 for s in stacks_bb if s < 5) / len(stacks_bb)
    frac_5to8 = sum(1 for s in stacks_bb if 5 <= s <= 8) / len(stacks_bb)
    check("sub-5bb density present (~6-10% incl. band-edge rounding)", 0.03 < frac_sub5 < 0.12,
          f"P(<5bb)={frac_sub5:.3f}")
    check("5-8bb band upweighted", frac_5to8 > 0.12, f"P(5-8bb)={frac_5to8:.3f}")
    check("min draw >= 2bb, max <= 100bb", min(stacks_bb) >= 2.0 and max(stacks_bb) <= 100.0,
          f"range=[{min(stacks_bb):.0f}, {max(stacks_bb):.0f}]bb")

    print("== [VAL-5] resume round-trip ==")
    from versions.v48.core.model import PokerEVModelV4
    from versions.v48.core.manifest import MANIFEST
    m = PokerEVModelV4()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=37)
    # take a couple of real steps so Adam moments are non-trivial
    for _ in range(3):
        loss = sum((p ** 2).sum() for p in m.parameters())
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    tmp = os.path.join(os.path.dirname(__file__), '_verify_roundtrip.pth')
    save_checkpoint(m.state_dict(), tmp, MANIFEST, hands_trained=123,
                    optimizer_state=opt.state_dict(), scheduler_state=sched.state_dict())
    ck = torch.load(tmp, map_location='cpu')
    m2 = PokerEVModelV4(); m2.load_state_dict(load_state_dict(tmp, MANIFEST))
    opt2 = torch.optim.Adam(m2.parameters(), lr=1e-3)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=999)  # deliberately different
    opt2.load_state_dict(ck['optimizer_state'])
    sched2.load_state_dict(ck['scheduler_state'])
    same_moments = all(
        torch.equal(opt.state_dict()['state'][k]['exp_avg'], opt2.state_dict()['state'][k]['exp_avg'])
        for k in opt.state_dict()['state'])
    check("optimizer Adam moments bit-identical after round trip", same_moments)
    check("scheduler position + T_max restored (original schedule wins over the fresh one)",
          sched2.last_epoch == sched.last_epoch and sched2.T_max == 37,
          f"last_epoch={sched2.last_epoch}, T_max={sched2.T_max}")
    os.remove(tmp)

    print("== [P0.4] contract_version hard validation ==")
    v44_manifest = get_manifest('v44')
    mismatch_path = os.path.join('versions', 'v44', 'weights', 'frozen_v43.pth')
    raised = False
    try:
        load_state_dict(mismatch_path, v44_manifest)
    except ValueError:
        raised = True
    check("cv mismatch RAISES by default", raised)
    ok_opt_in = load_state_dict(mismatch_path, v44_manifest, allow_contract_mismatch=True) is not None
    check("cv mismatch loads with explicit opt-in", ok_opt_in)

    print("== [M9-serve] mirror declarations ==")
    from core.models.v47_engine import V47ModelEngine
    check("v47_engine declares collapse_aliased_allin", getattr(V47ModelEngine, 'collapse_aliased_allin', False))
    dsrc = open(os.path.join('core', 'decision.py'), encoding='utf-8').read()
    check("decision.py sampler reads the flag", "collapse_aliased_allin" in dsrc)
    # Precondition parity: serve-side slider sizing clamps every bucket to the stack exactly when
    # train-side sizing does (same min/max recipe) -- spot-check the clamped geometry above.
    from core.board_state import BoardState
    bs = BoardState(community_cards=[], hero_cards=['As', 'Ks'], pot_size=100.0, hero_stack=30.0,
                    street="Preflop", big_blind=10.0, call_amount=0.0, equity=0.6)
    import core.decision as core_decision
    eng = object.__new__(core_decision.PokerDecisionEngine)
    sizes = [core_decision.PokerDecisionEngine._v14_size_to_slider(eng, f, bs)[0]
             for f in (0.33, 0.66, 1.0, None)]
    check("serve-side sizes clamp identically at the clamped node", all(s == 30.0 for s in sizes),
          f"sizes={sizes}")

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
