"""V21_auxhead manifest — tests whether the bluff/strength/equity aux heads produce rational output
once actually trained. IDENTICAL architecture/tensor schema (context_dim=37, contract_version=5).

Motivation (see versions/v21/SPECS.md item 7): the aux heads have existed in PokerEVModelV4 since
early versions but have always trained at `aux_loss_weight=0.0` -- a forward pass + loss computed
every step, contributing exactly zero gradient. Genuinely inert since V14.

**Phase 1** (2026-07-17, complete): a fast rationality check, WARM-STARTED from V21's own final
100k-hand checkpoint (`frozen_v21.pth`) for a short ~20k-hand continuation with
`aux_loss_weight=0.05`. Findings (full detail in SPECS.md): no lasting destabilization (a mid-run
loss spike traced to the resume mechanism not restoring optimizer/scheduler state, not the aux
heads -- fully resettled by the end); `self_equity` correlated well with its label (r=0.894,
proving the gradient path itself is wired correctly); `opp_strength`/`opp_bluff` showed only weak
correlation. Digging into WHY surfaced a real bug: `opp_bluff_prob` (in
`simulator.py::_mc_target_evs_sized`) was computed as `max_opp_equity < 0.33` -- true whenever ANY
active opponent held weak cards, regardless of whether anyone had actually acted aggressively, so
it fired just as often on a hand where a weak opponent simply folds as one where they genuinely
bluff. Fixed 2026-07-17: now gated on `last_raiser`, reading specifically the last aggressor's own
equity, 0.0 when no opponent is the last raiser (not a bluff scenario by definition).

**Phase 2** (2026-07-17, complete): a FRESH from-scratch 100k-hand run (matching V21's own
`target_hands` exactly, no `--resume_path`) -- both because Phase 1's loss-spike confound (fresh
optimizer/scheduler on resume) makes a from-scratch comparison cleaner, and because the corrected
`opp_bluff_prob` deserves a full training run's worth of exposure rather than a 20k-hand tail
grafted onto a model that never saw the corrected label. `self_equity` improved further
(r=0.942). `opp_bluff` got WORSE (r=0.080 vs Phase 1's 0.132 on the OLD broken label) despite full
training -- not a failure of the label fix: gating on `last_raiser` correctly made the label
sparser (~2% positive rate), and plain MSE trivially minimizes an imbalanced target by predicting
near-zero for everything, exactly what was observed.

Phase 2's own `model_verify --full`: 16 PASS/2 WARN/1 FAIL/0 SKIP, same shape as V21. Notable
deltas: `hand_strength_sweep` 0.237->0.825 (3x+ more responsive -- first real evidence for the
representation-learning hypothesis), `action_diversity` shows `call` winning argmax for the first
time in this lineage. One real dip: `bb100_vs_standard_fields`'s `tight_deep` field below V21's own
range. Full detail in SPECS.md.

**Phase 3** (complete): fixed the bluff-head collapse via `_bluff_pos_weight()` in `train.py` --
per-batch inverse-frequency reweighting (mirrors `BCEWithLogitsLoss`'s `pos_weight`). Warm-started
from Phase 2 (+25k hands). Result: fixed the collapse (pred std 0.02->0.25) but OVERCORRECTED
(pred mean 0.30 vs a true ~2% base rate) -- the full ratio equalizes gradient mass but doesn't
anchor predicted magnitude.

**Phase 4** (complete): dampened via sqrt of the ratio instead of the raw ratio. Best-calibrated
bluff result of any variant (pred mean 0.019 vs label mean 0.020, r=0.115, the best of the three).
But `strength`/`equity` correlations dropped in this and Phase 3 (both short 25k warm-started
continuations from the same base) -- likely the shared `aux_loss_weight` budget across all three
heads, not something specific to either weighting scheme.

**Phase 5** (in progress): same dampened weighting, longer continuation (+50k hands) testing
whether the strength/equity dip is a short-continuation transient (like Phase 1's main-loss spike,
which fully resettled) or a standing tradeoff.

See: versions/v21/SPECS.md item 7 (motivation) | versions/v21_auxhead/SPECS.md (full detail, all
phases' results, the opp_bluff_prob fix, the reweighting fix)
"""
from shared.manifest import VersionManifest

MANIFEST = VersionManifest(
    version_id="v21_auxhead",
    context_dim=37,                 # UNCHANGED from V21 -- same tensor schema
    contract_version=5,              # UNCHANGED -- aux_loss_weight-only change, no contract change
    action_space=("fold", "call", "raise_33", "raise_66", "raise_pot", "allin"),
    model_class="versions.v21_auxhead.core.model:PokerEVModelV4",
    contract_class="versions.v21_auxhead.core.contract:ContractV12",
    weights_dir="versions/v21_auxhead/weights",
    status="active",
    milestone=False,
)
