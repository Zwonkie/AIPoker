# V29 SPECS

Branches from `versions/v28` (fresh weights, not resumed -- per [VAL-5]). SAME opponent pool,
deep-stack curriculum, entry-sizing, `pot_type`, multi-street EV fix, and risk-adjusted target
mechanism as V28 (see `versions/v28/SPECS.md`), except the risk-aversion coefficient itself is
bumped (see below). Two changes this version, both scoped by explicit user direction (2026-07-20:
"for V29 lets do this: 'a critic-consistency filter' and the 'Per-opponent action attribution
[OPP-2]'"), built and trained unsupervised per that same instruction ("I will be gone for a couple
of hours, so I trust in your recommendations and fixes for V29").

## Change 1: [OPP-2] per-opponent-seat raise attribution (context_dim 44->54, contract_version 7->8)

**Motivation**: flagged since the V16 ROADMAP as `[P6]`/`[OPP-2]` in the OFK backlog, never
addressed. Before V29, the model had no way to tell WHICH specific opponent seat was raising during
a hand -- only a hand-level aggregate (`pot_type`: limped/single-raised/3-bet+) and each seat's
STATIC, cross-hand VPIP/AGG HUD color. A seat that's been the sole aggressor all hand and a seat
that's only ever called looked identical if their HUD colors matched.

**Design**: two new per-seat boolean arrays, `raised_this_hand[seat]` (has this seat raised at
least once so far this hand) and `raised_this_street[seat]` (did this seat raise on the CURRENT
betting street), threaded through the exact same plumbing `committed`/`opponents_committed`
already use:
- `versions/v29/self_play/simulator.py`: `raised_this_hand` reset once per hand (alongside
  `committed`/`raise_count`); `raised_this_street` reset once per street (alongside
  `street_committed`). Both seats' branches of the betting loop (hero's own and every opponent
  bot's) set `raised_this_hand[actor]=True` / `raised_this_street[actor]=True` on a raise action.
  Threaded into `table_state` (for `_query_model_decide`, so every NN-backed opponent's OWN query
  sees this too, not just hero's) and into `add_decision` (for hero's own training record).
- `core/board_state.py`: `SeatState` gained matching `raised_this_hand`/`raised_this_street` fields
  (default `False`) -- optional/additive, inert for every earlier version's contract, same
  convention `committed` used before V22.
- `versions/v29/core/contract.py`: appended (not inserted -- every existing index 0-43 stable) as
  ctx[44:49]=`opp_raised_this_hand`, ctx[49:54]=`opp_raised_this_street` (same 5-seat order as the
  existing per-opponent block).
- `versions/v29/self_play/train.py`'s `vectorize_hand_samples` (the SEPARATE gradient-training
  context builder that does NOT go through `ContractV12` -- the exact duplication that let V20's
  own rescale silently drift out of sync) mirrors the same 10 features in the same order.
- `tools/model_verify/scenarios.py`'s `build_ctx` extended for `contract_version>=8` with matching
  optional params, defaulting to all-zero so every existing FAST check (none of which pass them) is
  unaffected.

**Verification before training**: a 5-part pre-training test suite (model forward pass at
context_dim=54; `ContractV12.to_tensors` producing a real 54-length context with the new features
firing correctly for a seat that raised; `vectorize_hand_samples` matching it exactly;
`build_ctx(contract_version=8)` matching both; `regret_match_policy_torch`'s new filter, see Change
2) plus two real-simulated-hand integration smoke tests (one with the default heuristic pool, one
with the REAL config-driven pool including TreeOpponent and lagged-self NN opponents) -- confirmed
the real betting loop actually sets `opponents_raised_this_hand=True` at least once across a small
batch of real hands (not just my synthetic unit-test construction), and every vectorized sample's
context width is 54 with no crashes anywhere in the real opponent-pool code paths.

**Update (2026-07-20, later same day)**: initially shipped functional for TRAINING only (exercised
by every training hand and by `model_verify --full`, which runs entirely against the version's own
simulator, not the live PHPHelp bridge) with LIVE serving deliberately deferred -- `core/
table_state.py` had no per-seat raise/aggression tracking, and building it would mean modifying the
same live game-state code the then-active V28 model depended on, unsupervised. Per explicit user
request ("deploy to live and make sure the live boardstate can provide all informations needed"),
this was completed the same day -- see the OFK backlog's [OPP-2] entry for the full mechanism
(`_generate_timeline_actions`'s new per-seat raise-vs-call classification, preflop BB-seeded bet
level, per-street reset) and two byproduct fixes: (1) `committed`/`hero_committed`/`pot_type`
(V22/V23 features) were ALSO silently always-0/inert in live play for every version since V22 --
`to_board_state()` never set them -- now sourced from real tracked state; (2) neither hero's nor an
opponent's stack correctly registered a genuine all-in (both had a `> 0` monotonic-decay guard with
no way to distinguish a real 0 from a failed OCR read) -- opponents already had a reliable
`state='All-In'` vision signal `table_state.py` wasn't using; hero's OCR path had no equivalent
signal at all, so one was added to `core/vision.py` mirroring the opponent pattern exactly.
Verified via unit tests (raise/call classification, committed tracking, both all-in fixes, and the
all-in-vs-noise regression case) plus a full end-to-end test through
`core.decision.PokerDecisionEngine.make_decision` with V29 active. NOT verified against real screen
captures (no live table this session) -- the vision.py change mirrors already-proven production
logic rather than introducing a new heuristic, but real-table confirmation is still worth doing.

## Change 2: critic-consistency filter (training-loop only, no contract impact)

**Motivation**: `regret_match_policy_torch`'s fold-relative regret matching (V17 round 2's
established, validated baseline) gives ALLIN real actor-target probability mass any time its Q
merely clears the ~0 fold baseline -- it never compares ALLIN against the actual best alternative
action. Hypothesis going in: this divergence (a well-ranked critic, a policy formula that doesn't
fully respect that ranking) could be part of why `deep_stack_ood_guard`/[STACK-1] has failed in
every version since V22's one-time pass.

**Calibration finding (important, changed the plan honestly)**: ran the candidate filter against
the frozen V28 checkpoint's REAL Q-values across `deep_stack_ood_guard`'s own eq x stack grid
(`versions/v29/self_play/calibrate_critic_consistency.py`) BEFORE committing to a design. Result: at eq=0.43
(all 5 stack depths), V28's critic ranks ALLIN (Q=0.16-1.01) clearly below RAISE_POT (Q=1.29-1.55)
while ALLIN still clears the fold baseline -- exactly the spurious-weight case a consistency filter
should fix. But at eq=0.48 and eq=0.55 (the check's OTHER 10 failing cells, e.g. Q_allin=4.16 vs
Q_raise_pot=3.51 at eq=0.55/stack=25bb), ALLIN is the critic's OWN genuine Q-argmax -- there, a
policy-side consistency filter is a correct no-op, because the problem isn't the actor
misrepresenting the critic, it's what the critic itself learned. **This is documented honestly as a
PARTIAL fix, not a full fix for [STACK-1]** -- the eq=0.48/0.55 half needs the critic's own training
target to change, which is why `risk_aversion_coefficient` (V28's own lever) is bumped alongside
this (0.10->0.15) rather than treating the consistency filter as sufficient alone.

**Design**: `regret_match_policy_torch` gained `critic_consistency_margin` (0.0=off): if any OTHER
action's Q beats ALLIN's own Q by more than this margin, ALLIN's regret is zeroed outright,
regardless of whether it still beats the fold baseline. Applied ONLY to ALLIN (index -1, always the
last action in this codebase's fixed 6-action contract) -- NOT a general all-pairs dominance rule.
An all-pairs version was tested against the same calibration data first and rejected: even at a
lenient margin it collapsed legitimate raise-size mixing (call/raise_33/raise_66/raise_pot's own
intentional spread, e.g. at eq=0.35/stack=40bb) down to a single surviving action in 19-23 of 25
grid cells -- a much bigger unintended hit to `action_diversity`/[BET-2]'s sizing mix than the
targeted ALLIN fix was meant to cause. Calibrated margin: 0.15 (bb units, same scale as the critic's
own Q outputs) -- chosen to match the risk-aversion coefficient's own scale, not independently
re-swept via the full grid-search procedure (time-boxed this pass); revisit if `model_verify --full`
shows it under- or over-firing.

**Verification before training**: direct unit tests against synthetic Q-vectors shaped like the
three real regimes found during calibration -- (a) ALLIN clearly dominated but beats fold (the
eq=0.43 shape): filter correctly zeroes it; (b) ALLIN genuinely the best action (the eq=0.55 shape):
filter correctly leaves it untouched (confirmed no false positives); (c) the pre-existing degenerate
all-below-fold fallback: confirmed unaffected by the new filter, still resolves to 100% FOLD.

## Also changed: `risk_aversion_coefficient` bumped 0.10 -> 0.15

Not independently re-calibrated via the full 40-trial-averaged procedure V28 itself used (time-
boxed this pass) -- a moderate, same-direction step informed directly by the calibration finding
above (V28's own value wasn't enough pushback at eq=0.48/0.55). Re-evaluate via `model_verify
--full` below and iterate again (either a further bump, or a proper re-calibration pass) if
`deep_stack_ood_guard` still fails at those specific cells.

## Results

Trained 100,006 hands from scratch (fresh weights, no `--resume_path`). Hero finished the run at
+42.9 BB/100 vs the field (VPIP 43.5%/AGG 64.7%). `model_verify --full`: **21 PASS, 2 WARN, 0 FAIL,
1 SKIP** -- the cleanest scorecard of any version this whole diagnostic lineage (V21_auxhead through
V28 all carried at least 1 FAIL).

**`deep_stack_ood_guard` [STACK-1]: PASS.** This is the headline result -- the FIRST time this
check has cleared since V22's one-time pass (V23, V24, V24_extreme, V25, V26, V27, V28 all FAILed
it, at gradually improving-but-never-clearing confidence). No marginal-equity/deep-stack/single-
modest-bet cell jams all-in anymore.

**`allin_vs_nextbest_qgap` [BET-1]: PASS, and qualitatively different from every prior "PASS/WARN"
result in this lineage.** WORST-cell gap (allin's Q minus the next-best action's Q, as a fraction
of pot) is NEGATIVE at every single stack depth swept (15bb=-1.00, 20bb=-1.16, 25bb=-1.35,
30bb=-1.56, 40bb=-1.97) and every archetype (NIT=-0.43, TAG=-0.43, LAG=-0.44,
CALLING_STATION=-0.47). V28's OWN worst cells were still POSITIVE (all-in still winning some
cells) -- V29 doesn't just shrink the gap further, it flips the sign everywhere tested. Given the
honest partial-coverage finding documented above (the critic-consistency filter alone doesn't
explain the eq=0.48/0.55 cells), this result is most likely driven primarily by the
`risk_aversion_coefficient` bump (0.10->0.15) doing more work than expected, with the
critic-consistency filter contributing at the eq~0.43 band as designed -- the two mechanisms
weren't tested in isolation this pass, so their individual contributions aren't separately
attributable. `action_diversity` stayed healthy ({'fold': 11, 'call': 5, 'raise_pot': 5} argmax
distribution across the sweep) -- confirms the critic-consistency filter's ALLIN-only scope (not
an all-pairs rule) didn't collapse legitimate raise-size mixing, the exact side effect that
motivated scoping it that narrowly.

**Every other previously-clean check stayed clean**: `vpip_adapts_to_style` PASS (short delta
+6.0pts, deep +5.6pts), `beats_offformula_stress` PASS (+14.1/+30.1 BB/100), `bb100_vs_standard_fields`
PASS across all 4 fields (+20.7 to +72.5 BB/100), `position_sweep` PASS (spread 0.133, no
regression), `stack_full_sweep` PASS (coherent fold->call progression, no all-allin collapse).

**`beats_frozen_predecessor`: SKIP, not FAIL -- a structural evaluation gap, not a model problem.**
Copying V28's `expert_main.pth` in as `frozen_v28.pth` and loading it into V29's architecture fails
fast-loud: V28 is context_dim=44 (contract_version=7), V29 is context_dim=54 (contract_version=8)
-- an actually-different input layer, not just different weights, so the checkpoint literally can't
load into this version's model class. This is the FIRST version-over-version contract change since
V22->V23 to trip this (every version V23-V28 kept the 44-feature contract). The existing
`beats_frozen_predecessor` mechanism (load old weights into the new architecture) fundamentally
doesn't support an architecture change -- a genuine head-to-head vs V28 would need a different
harness (V29-as-hero vs a frozen V28 loaded as an OPPONENT via its own architecture, not weights
loaded into V29's), which wasn't built this pass. `bb100_vs_standard_fields`'s absolute numbers
(+20.7 to +72.5 BB/100) are in the same range as V28's own equivalent numbers, informally
suggesting no regression, but this is NOT a substitute for a real head-to-head and shouldn't be
read as one.

**Unchanged WARNs (both pre-existing, not new)**: `free_check_low_fold` (covered by decision.py's
live free-check mask, tracked not gated, same as every prior version) and
`allin_exploits_opponent_foldiness` [OPP-8] (spread 0.008, essentially flat -- consistent with
V25/V26's own ~0.011, not a regression, still open).

**Status**: **DEPLOYED LIVE (2026-07-20)** per explicit user request, on the strength of this
result (first-ever clean sweep of both [STACK-1] and [BET-1] together). `core/decision.py`'s
`active_model_name` is `'Herocules (v29)'`. [OPP-2]'s live-serving limitation was ALSO closed the
same day (see the "Update" note above and the OFK backlog's [OPP-2] entry) -- the 10 new features
are now real live, not inert placeholders, alongside two byproduct fixes (`committed`/
`hero_committed`/`pot_type` were silently inert since V22/V23; hero/opponent all-in stack tracking).
V25/V26/V27/V28 remain in the registry as rollback options.

See: `versions/v28/SPECS.md` (risk-adjusted target, BET-1 mechanism) | `versions/v27/SPECS.md`
([VAL-3]/[OPP-7]) | `.agents/skills/OFK/references/known-shortcomings-backlog.md` ([BET-1], [OPP-2])
