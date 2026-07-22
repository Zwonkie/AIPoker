import os
import csv
import torch
from core.board_state import BoardState
from core.bridge.data_contract import DataContract
from typing import Tuple

# Must match exactly with the training vocabulary
VOCAB = {'<PAD>': 0, 'B': 1, 'b': 2, 'c': 3, 'k': 4, 'K': 5, 'r': 6, 'f': 7, 'A': 8, 'Q': 9}

def card_to_int(card_str: str) -> int:
    if not card_str or len(card_str) != 2:
        return 52 # PAD
    rank, suit = card_str[0], card_str[1]
    ranks = '23456789TJQKA'
    suits = 'cdhs'
    try:
        r = ranks.index(rank)
        s = suits.index(suit)
        return s * 13 + r
    except ValueError:
        return 52


# ======================================================================= #
#  Money-denominated context feature rescale constants.
# ======================================================================= #
# Named here (not just inlined) so `train.py::vectorize_hand_samples` -- a SEPARATE context-vector
# builder used for the actual gradient-training tensors, which does NOT go through this class --
# can import these same constants instead of re-deriving its own copy. That duplication is exactly
# what let V20's own rescale silently miss vectorize_hand_samples (it kept the old /400,/1000
# scale while this file got the new /100,/250 scale) -- see versions/v20_preflopEq/SPECS.md and
# MANIFEST's docstring.
#
# [V22] Ceilings raised 50/100/50 -> 100/200/100 ([STACK-2] -- training had never gone past 50bb
# effective stacks). SCALE constants unchanged (same per-bb resolution as before, no rescale
# needed): stack/call's old ceiling/scale ratio was exactly 0.5 (50/100), so doubling the ceiling
# to 100/100 now uses the FULL previously-half-wasted [0,1] range. Pot's ratio was already
# sub-1.0 by design (100/250=0.4, finer per-bb resolution than stack/call) -- doubling its ceiling
# to 200/250=0.8 uses more of the range than before without hitting 1.0; not a completeness
# requirement, pot just never needed the same "half-wasted" fix stack/call did. See
# versions/v22/SPECS.md.
STACK_CEIL_BB = 100.0
POT_CEIL_BB = 200.0
CALL_CEIL_BB = 100.0
STACK_SCALE = 100.0
POT_SCALE = 250.0
CALL_SCALE = 100.0

VPIP_MAP = {'Red': 0.45, 'Yellow': 0.30, 'Green': 0.22, 'Blue': 0.10}
AGG_MAP = {'Red': 0.85, 'Yellow': 0.63, 'Green': 0.46, 'Blue': 0.18}


def scaled_stack_bb(stack_chips: float, big_blind: float) -> float:
    """hero/opponent stack, in bb, clamped to the training ceiling and rescaled to ~[0,1].
    Shared by contract.py and vectorize_hand_samples so they can never drift apart again."""
    return min(stack_chips / big_blind, STACK_CEIL_BB) / STACK_SCALE


def scaled_pot_bb(pot_chips: float, big_blind: float) -> float:
    return min(pot_chips / big_blind, POT_CEIL_BB) / POT_SCALE


def scaled_call_bb(call_chips: float, big_blind: float) -> float:
    return min(call_chips / big_blind, CALL_CEIL_BB) / CALL_SCALE


def effective_contested_field(after_vpips, n_front: int = 0) -> float:
    """[V44] E[k | k>=1] -- how many opponents are EXPECTED to actually contest the pot, which is
    the field `equity` is really measured against.

    `compute_range_aware_equity` rolls each still-to-act opponent at their VPIP preflop and SKIPS
    all-fold samples, so its output is showdown strength CONDITIONAL on someone contesting. This
    reproduces that same conditioning in closed form -- no MC, so no extra variance:

        E[k | k>=1] = (n_front + sum(p_after)) / (1 - prod(1 - p_after))

    `after_vpips` are the per-opponent VPIP probabilities of seats still to act (use
    `_COLOR_TO_VPIP`-equivalent values, i.e. the same numbers the equity roll uses -- VPIP_MAP
    below). `n_front` counts opponents already committed this round: they are never rolled (p=1)
    and, being guaranteed in, drive P(k=0) to 0 so no conditioning applies.

    POSTFLOP callers must pass the nominal opponent count as `n_front` with no `after_vpips`,
    because no roll happens postflop -- this then returns exactly that count and the resulting
    feature is byte-identical to V43's.
    """
    ps = [float(p) for p in (after_vpips or [])]
    expected = float(n_front) + sum(ps)
    if n_front > 0:
        return expected                      # someone is guaranteed in; k >= 1 always
    p_none = 1.0
    for p in ps:
        p_none *= (1.0 - p)
    if p_none >= 1.0:                        # no opponents at all
        return 0.0
    return expected / (1.0 - p_none)


def equity_edge_feature(equity: float, num_active: float) -> float:
    """Equity's edge over the field's fair share: 1.0 = exactly average, >1 better, <1 worse.
    `equity * (num_active + 1)`, deliberately NOT the plain `1/(num_active+1)` fair share alone --
    that's a deterministic unary function of the already-present `num_active` feature and would add
    ~nothing on its own; the genuinely hard part for a plain feedforward layer is the equity*N
    interaction, which this precomputes directly (same rationale as explicit cross-product features
    in factorization machines / wide-and-deep). [V20_preflopEq], see that version's SPECS.md.

    [V44] `num_active` is now the EFFECTIVE contested field (a float from
    `effective_contested_field`), not the nominal seat count. Through V43 this normalized by the
    nominal count while `equity` was measured against the rolled/conditional one, so the two halves
    counted different things and the ratio drifted upward with field size instead of measuring hand
    strength -- AKs' edge ran 1.34 -> 3.12 across 1->5 opponents purely from the denominator, and no
    model in the lineage ever learned to use the feature. With the denominators agreed it is flat
    (1.32 / 1.37 / 1.46) and still separates hands (AA 1.70-2.16 ... 72o 0.59-0.62). See
    versions/v48/core/manifest.py."""
    return equity * (num_active + 1)


# ======================================================================= #
#  V20_preflopEq `hand_strength` — preflop O(1) lookup table
# ======================================================================= #
_PREFLOP_HAND_STRENGTH = None  # {'AA': 0.8515, 'AKs': 0.6614, ...} -- 169 canonical hands


def _load_preflop_hand_strength():
    """Loads preflop_equities.csv (repo root; generated by scripts/math/generate_equities.py --
    169 canonical starting hands, each vs 1 random opponent over 10,000 MC sims), cached in
    memory after the first call. See versions/v20_preflopEq/SPECS.md `hand_strength` section --
    this is a strictly lower-variance source than the in-simulator `_get_preflop_ranked()` cache
    (that one scores 1326 raw combos at only 80 sims each and only stores rank order)."""
    global _PREFLOP_HAND_STRENGTH
    if _PREFLOP_HAND_STRENGTH is not None:
        return _PREFLOP_HAND_STRENGTH
    path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'preflop_equities.csv')
    table = {}
    try:
        with open(path, newline='') as f:
            reader = csv.reader(f)
            next(reader, None)  # header row
            for row in reader:
                if len(row) != 2:
                    continue
                hand, eq = row
                table[hand] = float(eq)
    except Exception:
        table = {}
    _PREFLOP_HAND_STRENGTH = table
    return table


def canonical_hand_key(card_a: str, card_b: str) -> str:
    """2 concrete hole cards (e.g. 'Ah', 'Ks') -> one of the 169 canonical hand-class strings
    matching preflop_equities.csv's Hand column ('AKs'). Pairs have no suffix ('AA'). Falls back
    to 'XXo' (worst-case offsuit-shaped key, never a real CSV entry) on malformed input so the
    caller's .get(..., default) always applies cleanly."""
    ranks = '23456789TJQKA'
    if not card_a or not card_b or len(card_a) != 2 or len(card_b) != 2:
        return 'XXo'
    r1, s1 = card_a[0].upper(), card_a[1].lower()
    r2, s2 = card_b[0].upper(), card_b[1].lower()
    if r1 not in ranks or r2 not in ranks:
        return 'XXo'
    if r1 == r2:
        return f"{r1}{r2}"
    hi, lo = (r1, r2) if ranks.index(r1) > ranks.index(r2) else (r2, r1)
    suited = 's' if s1 == s2 else 'o'
    return f"{hi}{lo}{suited}"


def preflop_hand_strength(card_a: str, card_b: str) -> float:
    """O(1) lookup: hero's raw preflop equity vs 1 random opponent, independent of board/field --
    the `hand_strength` context feature's preflop source. Falls back to 0.5 (neutral) for an
    unrecognized hand string rather than raising, since this feeds a live/training tensor path."""
    table = _load_preflop_hand_strength()
    return table.get(canonical_hand_key(card_a, card_b), 0.5)


class ContractV12(DataContract):
    """
    Implements the 54-feature context extraction for Pluribus V12 models (V29).

    44 features inherited from V23-V28 unchanged, plus 10 new appended (not inserted -- every
    existing index 0-43 is stable): per-opponent-seat raise attribution [OPP-2] -- for each of the
    5 opponent slots (same seat order as the existing per-opponent block), `raised_this_hand`
    (ctx[44:49]) and `raised_this_street` (ctx[49:54]), each a 0.0/1.0 flag for whether THAT
    SPECIFIC seat has raised so far this hand / on the current betting street. Distinguishes a
    specific opponent's own in-hand aggression from the static, cross-hand VPIP/AGG HUD colors
    already in the per-opponent block -- previously the model only ever saw "someone raised"
    (via `pot_type`, a hand-level aggregate) with no way to attribute it to a seat. See
    .agents/skills/OFK/references/known-shortcomings-backlog.md [OPP-2] and versions/v29/SPECS.md.
    (PokerEVModelV4 architecture)
    """

    def __init__(self, max_seq_len: int = 20):
        self.max_seq_len = max_seq_len

    def to_tensors(self, states, hero_actions: list = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(states, list):
            states = [states]

        states = states[-self.max_seq_len:]
        # Left-padding: start index is at the end of the array
        start_idx = self.max_seq_len - len(states)

        # 1. Hole Cards (from final state)
        hole_ints = [card_to_int(c) for c in states[-1].hero_cards]
        while len(hole_ints) < 2:
            hole_ints.append(52)

        # 2. Board Cards Sequence (padded to max_seq_len)
        board_seq = [[52]*5 for _ in range(self.max_seq_len)]

        # 3. Context (54 features) Sequence
        context_seq = [[0.0]*54 for _ in range(self.max_seq_len)]

        for i, state in enumerate(states):
            idx = start_idx + i

            b_ints = [card_to_int(c) for c in state.community_cards]
            while len(b_ints) < 5:
                b_ints.append(52)
            board_seq[idx] = b_ints

            pot_odds = state.call_amount / (state.pot_size + state.call_amount) if (state.pot_size + state.call_amount) > 0 else 0.0

            board_len = len(state.community_cards)
            if board_len == 0: street_level = 0.0
            elif board_len == 3: street_level = 1.0
            elif board_len == 4: street_level = 2.0
            else: street_level = 3.0

            # Determine active opponents mask
            active_mask = []
            for j in range(5):
                seat_key = f"seat_{j+1}"
                if seat_key in state.seats:
                    active_mask.append(1.0 if state.seats[seat_key].is_active else 0.0)
                else:
                    active_mask.append(0.0)

            # Dynamically calculate global VPIP/AGG norms from active opponents
            total_active = sum(active_mask)
            if total_active > 0:
                sum_vpip = 0.0
                sum_agg = 0.0
                for j in range(5):
                    if active_mask[j] == 1.0:
                        seat_key = f"seat_{j+1}"
                        opp = state.seats[seat_key]
                        vpip_col = opp.hud.vpip_color
                        agg_col = opp.hud.agg_color
                        sum_vpip += VPIP_MAP.get(vpip_col, 0.3)
                        sum_agg += AGG_MAP.get(agg_col, 0.4)
                opp_vpip_norm = sum_vpip / total_active
                opp_agg_norm = sum_agg / total_active
            else:
                opp_vpip_norm = 0.3
                opp_agg_norm = 0.4

            ctx = [
                float(state.hero_position) / 5.0,
                # V20 rescale (inherited unchanged, see STACK_SCALE/POT_SCALE comment above).
                scaled_stack_bb(state.hero_stack, state.big_blind),
                scaled_pot_bb(state.pot_size, state.big_blind),
                state.equity,
                pot_odds,
                sum(active_mask) / 10.0,
                street_level / 3.0,
                opp_vpip_norm,
                opp_agg_norm,

                # BB Ratios
                scaled_call_bb(state.call_amount, state.big_blind),
            ]

            # Opponents' seats HUD
            opp_committed_bb = [0.0] * 5
            # [V29, OPP-2] Per-seat raise attribution (this hand / current street) -- same
            # is_active-gated lookup pattern as opp_committed_bb above.
            opp_raised_hand = [0.0] * 5
            opp_raised_street = [0.0] * 5
            for j in range(5):
                seat_key = f"seat_{j+1}"
                opp = state.seats.get(seat_key)
                if opp:
                    opp_stack = opp.stack
                    vpip_col = opp.hud.vpip_color
                    agg_col = opp.hud.agg_color
                    opp_committed_bb[j] = scaled_stack_bb(getattr(opp, 'committed', 0.0), state.big_blind)
                    opp_raised_hand[j] = 1.0 if getattr(opp, 'raised_this_hand', False) else 0.0
                    opp_raised_street[j] = 1.0 if getattr(opp, 'raised_this_street', False) else 0.0
                else:
                    opp_stack = 0.0
                    vpip_col = "Yellow"
                    agg_col = "Green"

                opp_pos = (j + 1 + state.hero_position) % 6
                pos_val = float(opp_pos) / 5.0 if active_mask[j] == 1.0 else -1.0

                ctx.append(active_mask[j])
                ctx.append(pos_val)
                ctx.append(scaled_stack_bb(opp_stack, state.big_blind))
                ctx.append(VPIP_MAP.get(vpip_col, 0.3))
                ctx.append(AGG_MAP.get(agg_col, 0.4))

            # [V20_preflopEq] Two appended features (indices 35, 36 -- everything above is
            # untouched V20). `equity_edge`: precomputed from state.equity + the active-opponent
            # count already derived above. `hand_strength`: populated by the CALLER onto the
            # BoardState (mirrors how `state.equity` itself is populated by the caller, not derived
            # here) -- simulator.py computes it once per decision via the preflop CSV lookup or a
            # cheap postflop vs-1-random MC call; defaults to 0.5 (neutral) if the caller never set
            # it (e.g. an older BoardState construction site that hasn't been updated).
            # [V44] Normalize by the EFFECTIVE contested field the caller measured `equity`
            # against, not the nominal seat count -- see equity_edge_feature's own note. Falling
            # back to `total_active` when the caller supplies nothing (0.0) keeps any
            # not-yet-updated construction site on exactly V43's behaviour rather than silently
            # producing a differently-scaled feature.
            _eff = float(getattr(state, 'effective_field', 0.0) or 0.0)
            ctx.append(equity_edge_feature(state.equity, _eff if _eff > 0 else total_active))
            ctx.append(float(getattr(state, 'hand_strength', 0.5)))

            # [V22] Six new appended features (indices 37-42 -- everything above is untouched
            # V21_auxhead). `opp_committed_this_hand_bb` per opponent slot (seat order matching the
            # per-opponent block above): chips that seat has ALREADY put into this hand's pot,
            # scaled the same way as `opp_stack` (same STACK_CEIL_BB/STACK_SCALE) since
            # committed + remaining-stack == starting stack, same domain. 0.0 for an inactive/absent
            # seat, matching `opp_stack`'s own inactive default. `hero_committed_this_hand_bb`:
            # same idea for hero (a single global feature, not per-seat). Both populated by the
            # CALLER (simulator.py's `committed[]` array, already tracked for pot math -- see
            # versions/v22/SPECS.md) -- default 0.0 (inert) for any construction site that hasn't
            # been updated, same pattern as `hand_strength` above.
            ctx.extend(opp_committed_bb)
            ctx.append(scaled_stack_bb(getattr(state, 'hero_committed', 0.0), state.big_blind))

            # [V23] One new appended feature (index 43 -- everything above is untouched V22).
            # `pot_type`: whole-hand raise count so far, bucketed (0=limped/unraised,
            # 1=single-raised, 2=3-bet+) and normalized /2.0. Populated by the CALLER
            # (simulator.py's `raise_count` counter) -- default 0 (inert, "limped") for any
            # construction site that hasn't been updated, same pattern as `hand_strength` above.
            ctx.append(float(getattr(state, 'pot_type', 0)) / 2.0)

            # [V29] Ten new appended features (indices 44-53 -- everything above is untouched
            # V23-V28). Per-opponent-seat raise attribution [OPP-2]: `opp_raised_hand` (ctx[44:49])
            # then `opp_raised_street` (ctx[49:54]), same seat order as the per-opponent block
            # above. Populated by the CALLER (simulator.py's `raised_this_hand`/`raised_this_street`
            # arrays) via SeatState.raised_this_hand/raised_this_street -- default False/0.0 (inert)
            # for any construction site that hasn't been updated, same pattern as `committed` above.
            ctx.extend(opp_raised_hand)
            ctx.extend(opp_raised_street)

            context_seq[idx] = ctx

        # 4. Action Sequence
        act_ints = [0] * self.max_seq_len
        if hero_actions is not None:
            # Transformer shifts actions by 1 internally. act_ints[i] should be the action taken AT states[i]
            for i in range(min(len(hero_actions), len(states))):
                act_ints[start_idx + i] = hero_actions[i]

        # Convert to batch-first tensors [1, ...]
        hole_tensor = torch.tensor([hole_ints], dtype=torch.long)
        board_tensor = torch.tensor([board_seq], dtype=torch.long)
        ctx_tensor = torch.tensor([context_seq], dtype=torch.float32)
        act_tensor = torch.tensor([act_ints], dtype=torch.long)

        return hole_tensor, board_tensor, ctx_tensor, act_tensor
