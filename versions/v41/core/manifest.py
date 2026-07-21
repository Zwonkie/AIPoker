"""V41 manifest -- simulation-realism package from the 2026-07-20 Fable full-stack review of V29.

Base: cloned from `versions/v40` (fresh weights, not resumed -- per [VAL-5]). Inherits V40's
[BET-3] package unchanged (betting round no longer ends on a check; CALL variance-penalised and
given multi-street continuation credit; ALLIN veto rescoped) -- see versions/v40/SPECS.md.

Contract is BYTE-IDENTICAL to V29/V40 (context_dim=54, contract_version=8): every change is in the
SIMULATION or in how an opponent-seat QUERY is encoded, never in the tensor schema. Checkpoints
stay contract-compatible and the live bridge needs no new wiring.

Four changes, all review findings, each with a measured before/after (see versions/v41/SPECS.md):

  #11 + #10  Opponent-seat query encoding. [OPP-7]'s V27 remap was DEFEATED AT THE TENSOR
             BOUNDARY: it keyed slots by absolute seat number while `ContractV12.to_tensors` reads
             only `seat_1..seat_5`, so for any non-hero actor the real hero went to a `seat_0` key
             the encoder never reads (and the surviving slots were misaligned). Now keyed by SLOT
             index. Measured: V40 dropped the hero on 128/128 NN-opponent queries, V41 on zero.
             Alongside it, `is_active` and `stack` now come from ground truth (`folded`/`stacks`
             threaded through table_state) instead of `idx < num_opponents` and a `hero_stack`
             placeholder -- the rollout encoder had drifted from both the gradient path and live.

  #8         NN opponents no longer play a degraded self: range-aware equity is no longer gated on
             `current_actor == 0` (the lagged-self mirror trained on it but was served vs-random),
             and the corrupted `call_amount = pot_odds * pot_size` -- which is NOT to_call, a
             pot-sized bet arrived at half size -- is now inverted exactly.

  #9         Asymmetric starting stacks (opponents 0.35x-2.0x log-uniform around hero's curriculum
             depth; hero's own depth untouched so every stack sweep still measures what it did).
             Measured: hero decisions facing a covering opponent 0 -> 199 per 250 hands. Plus the
             two rule bugs that symmetry was hiding: min-raise floor is now `to_call + last
             increment` (was always +1bb, an illegal under-raise after any prior aggression), and
             a short all-in no longer re-opens betting to players who already acted.

  #7         Dead blinds. Pre-folding ran BEFORE blinds were posted, so a pre-folded seat could pay
             a blind and never act -- hero learned that stealing prints money because the blind
             cannot defend. Blind seats are now resolved first and excluded from the pre-fold pool.
             Measured: hands reaching a flop with a dead blind 47.6% -> 0.0%.

Still open and deliberately out of scope: #6 (every opponent raise is exactly 0.75 pot), the last
member of the reviewer's [BET-3] bundle.

See: .agents/skills/OFK/references/fable-review-resolution-log.md | versions/v40/SPECS.md |
versions/v29/SPECS.md
"""
from shared.manifest import VersionManifest

MANIFEST = VersionManifest(
    version_id="v41",
    context_dim=54,                  # unchanged from V29 -- no contract change in this version
    contract_version=8,              # unchanged from V29 -- V29/V40 checkpoints are contract-compatible
    action_space=("fold", "call", "raise_33", "raise_66", "raise_pot", "allin"),
    model_class="versions.v41.core.model:PokerEVModelV4",
    contract_class="versions.v41.core.contract:ContractV12",
    weights_dir="versions/v41/weights",
    status="active",                 # DEPLOYED LIVE 2026-07-21 (core/decision.py active_model_name)
    # MILESTONE (2026-07-21): resolved [BET-3], the multiway-passivity complaint that drove this
    # whole line of work, with the cleanest scorecard any version has produced (22/5/0/0, first
    # ever with zero skips) and the first REAL beats_frozen_predecessor since the V18 refactor.
    # Keep as a known-good rollback point -- do NOT delete versions/v41/weights/expert_main.pth.
    # See versions/v41/MILESTONE.md.
    milestone=True,
)
