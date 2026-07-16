"""V19 manifest — clones V18 (versions/v18, the opponent-architecture refactor; never itself
deployed live, only sanity-run to 20k hands), and is a genuine training-content experiment, not
another plumbing pass. Three targeted fixes, all part of the carried-forward backlog V18's
SPECS.md flagged but deliberately left untouched:

1. **[P0] Deep-stack OOD trash-jam fix.** `deep_stack_ood_guard` has FAILED on 5 straight versions
   (V15/V16/v16_foldregret/V17/v17_gauntlet) with the same signature: marginal equity (~0.35-0.55),
   15-40bb stack, facing one modest bet -> ALL-IN argmax. Root cause: `opponent_bots.py`'s
   `decide_preflop` never used its own `pot_odds` argument -- the fold bar was a fixed per-style
   threshold regardless of bet size, unlike `decide_postflop` (fixed for this in V14 [P1b]).
   `_mc_target_evs_sized` samples `bot.decide_preflop(oeq, size_pot_odds)` to build the fold-equity
   term for EVERY raise size including all-in -- with size ignored, a min-raise and an all-in shove
   got the SAME simulated fold rate, systematically inflating the all-in EV target in the equity
   band just above breakeven (exactly where the check fails worst). Fix: mirror the postflop
   `continue_bar = pot_odds + style_shift` pattern into the preflop `facing_bet` branch.
2. **[hero_position] fix.** `_query_model_decide` never set `hero_position` for ANY opponent NN
   query (defaulted to 0 = Button, the loosest position, for every seat regardless of where it
   actually sat relative to the dealer button) -- and the querying seat's own hero-side context
   construction had the same gap. Fixed to thread each seat's real button-relative position
   (`(seat - button_seat) % 6`, the same formula the hero's own query already used) into every
   NN query, hero and opponent alike, so position is a genuine, correctly-labeled training signal
   for everyone at the table, not just the live Hero.
3. **[Past-Self mystery] investigation.** See SPECS.md for findings -- v17_gauntlet's `past` seat
   (true lagged self-play mirror) sat at ~24-25% VPIP for 130k-200k hands while Hero itself was
   flat at ~40-41% over the same stretch; if `past` were genuinely "Hero from <=5,000 hands ago" it
   should have converged onto Hero's own plateau once Hero stopped moving. Unresolved before this
   version; see SPECS.md for whatever this investigation turned up.

Same opponent-architecture refactor (`Opponent`/`HeuristicOpponent`/`NNOpponent`,
`build_opponent_pool`) as V18, carried over unchanged -- this version only touches target
construction and position-feature wiring, not the plumbing layer itself.

Same 6-action contract, same context_dim=35, same weights_dir layout as every version in this line.

See: .agents/skills/OFK/references/versioned-architecture-guardrails.md
     versions/v19/SPECS.md   |   versions/v18/SPECS.md (parent line, plumbing refactor + backlog)
"""
from shared.manifest import VersionManifest

MANIFEST = VersionManifest(
    version_id="v19",
    context_dim=35,                 # input schema UNCHANGED (35-feature context)
    contract_version=3,             # UNCHANGED: same discretized 6-action space
    action_space=("fold", "call", "raise_33", "raise_66", "raise_pot", "allin"),
    model_class="versions.v19.core.model:PokerEVModelV4",
    contract_class="versions.v19.core.contract:ContractV12",
    weights_dir="versions/v19/weights",
    status="active",
    milestone=False,                # v13 remains the kept milestone/fallback
)
