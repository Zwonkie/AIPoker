# V12 Design Proposal — Root-Cause Fixes & Structural Improvements

**Date Recorded**: 2026-07-13
**Status**: 🟡 PROPOSED (planning; not yet implemented)
**Related Files**:
- [train_selfplay.py](file:///c:/REPO/Antigravity/AIPoker/tools/self_play/v11/train_selfplay.py)
- [six_max_simulator.py](file:///c:/REPO/Antigravity/AIPoker/tools/self_play/v11/six_max_simulator.py)
- [contract_v11.py](file:///c:/REPO/Antigravity/AIPoker/core/bridge/v11/contract_v11.py)
- [engine.py](file:///c:/REPO/Antigravity/AIPoker/core/models/engine.py)
- [V11/issues-and-fixes.md](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/V11/issues-and-fixes.md)

## Context

V11 shipped and trained, but a single debugging session surfaced a long chain of defects: per-seat stat attribution collapse, the NN never being queried (a swallowed `KeyError`), a loose-collapse ratchet, a raise-everything hallucination, a Windows file-lock crash, and three mutually inconsistent inference conventions. Most were patched, but they share a small number of **root causes**. V12 should target those roots rather than accrue more band-aids.

### Recurring root causes (the "why V11 kept breaking")
1. **No single inference contract** — training, data-gen, and the live engine tokenize/query the model differently.
2. **A fragile learning objective** — regressing Q toward realized MC returns of the *taken* action, which has off-policy gaps and fat tails, then `argmax`-ing raw Q.
3. **Silent failure everywhere** — `try/except: pass` and silent random-weight fallback hid disabling bugs for entire runs.
4. **No checkpoint/contract versioning** — a 31→35 feature change (159→163 dims) silently loaded stale weights as garbage.

---

## P0 — Unify the inference contract (single source of truth)

**Problem.** Three conventions coexist today:
| Path | Padding | Mask | Read index |
|:---|:---|:---|:---|
| Training vectorizer (`vectorize_hand_samples`) | left (`start_idx = max-len`) | none | end |
| Simulator data-gen (`_query_model_decide`) | right (`start_idx = 0`) | yes | `[seq_len-1]` |
| `engine.py` (**live production bot**) | (contract) | **none** | `[-1]` |

A model trained under one convention and served under another is silently degraded, and the live bot is on the least-consistent path (no `key_padding_mask`).

**Proposal.** Exactly **one** `to_tensors()` and **one** forward-call signature, imported by training, simulator, `engine.py`, and the live bridge. Decide one padding direction + masking rule and enforce it everywhere.

**Implementation notes.**
- Make `ContractV12.to_tensors` the sole tokenizer; delete the ad-hoc mask/index construction in the simulator and engine.
- Add an **alignment test**: for a fixed `BoardState`, assert training and inference produce byte-identical `(hole, board, ctx, act)` tensors and identical `q_vals`. Run it in CI/pre-commit.
- This also closes the parked `engine.py` / live-bot inference-mask item.

---

## P0 — Replace the MC-return regression objective

**Problem.** Both V11 mode collapses trace to the objective:
- *Loose-collapse ratchet*: fold anchored at 0 vs a positive-tailed "enter" return, with taken-action-only learning → VPIP ratcheted to ~50%.
- *Raise-everything hallucination*: Call/Raise heads never saw states where those actions were −EV → extrapolated positive EV to all hands, including 0-equity air.

The counterfactual-target patch (all-action MC targets, weight 0.5) helps, but `argmax` over uncalibrated regressed Q-values is inherently brittle.

**Proposal (best-first).**
1. **Regret-matching / MCCFR-style targets.** Poker is CFR's home turf; the repo already references **Pluribus** (CFR-based). Training toward counterfactual regret is far more stable and naturally avoids single-action degeneracy.
2. **Actor-critic split.** Output an action *distribution* (policy head, trained by policy gradient / regret matching) plus a *value* baseline — instead of raw Q-values you `argmax`. Decouples "how good is this state" from "which action," removing the argmax-of-one-uncalibrated-head failure.
3. **Minimum bar if staying with Q-learning:** proper **TD targets + target network** (`Q = r + γ·V(s')`) rather than pure MC, keeping the all-action counterfactuals.

**Payoff.** The tightness-prior, ±40 BB clip, and `COUNTERFACTUAL_WEIGHT` hacks become largely unnecessary once the objective is sound.

---

## P1 — Fail-loud checkpoint & contract versioning

**Problem.** The 31→35 feature contract change made every old checkpoint 159-dim; loading one into the 163-dim model silently fell back to random weights and "output garbage" (the diagnostic tool did exactly this).

**Proposal.** Save checkpoints as a dict: `{model_state_dict, context_dim, contract_version, hands_trained, git_sha, timestamp}`. On load, **assert** `context_dim`/`contract_version` match and **raise a clear error** on mismatch. Never silently init random weights on a load failure.

---

## P1 — Stop swallowing exceptions

**Problem.** `try/except: pass` around `_query_model_decide` hid a `KeyError` that disabled the NN for entire training runs (every decision fell back to the heuristic chart, unnoticed).

**Proposal.** Replace blanket `except: pass` in inference/load paths with **log-and-raise** (or at minimum a counted, surfaced warning). A model-query failure during self-play should be loud, not silent.

---

## P1 — Evaluate on gameplay, not just single-state probes

**Problem.** The health suite's single-state EV probes disagreed with in-distribution behavior (model folded air ~95% in play but "raised everything" on canonical probes). Grading solely on probes is misleading.

**Proposal.**
- Primary metric: **BB/100 vs a fixed benchmark lineup over N hands** + an **approximate exploitability / best-response** number.
- Keep the EV-probe suite (`model-testing-suite.md`) as a secondary guardrail, but run it through the **unified** inference path (P0), and clean the stale-checkpoint zoo the diagnostic loads.

---

## P2 — Structural / skill-ceiling improvements

- **Bet sizing in the action space.** V11's fixed 0.75-pot raise caps the model's ceiling and distorts EV targets. Add a **sizing head** (or a small discrete sizing set). Likely the single biggest *skill* upgrade.
- **Real opponent league (PBT).** Fix personality-checkpoint versioning so a diverse **NN** league can actually run (V11 is all-heuristic today), and maintain a *population* of historical snapshots (AlphaStar-style) rather than a single lagged past-self.
- **Learned opponent embeddings.** Replace hand-mapped VPIP/AGG color midpoints with a learned opponent representation.

---

## Known issues carried into V12 (from the V11 endgame)

- **The counterfactual-target fix over-corrected toward loose/passive.** After adding the all-action MC counterfactual (fold=0, taken=realized, untaken=MC EV @ 0.5), the fresh retrain drifted to **VPIP ~69% by 70k hands** — i.e. it swung from "raise everything" to "call/enter everything." The Fold baseline pins fold≈0 while `ev_call = equity·pot − cost` is positive for many marginal hands, so `argmax` favors Call broadly. **This is strong evidence that patching targets is not enough and the objective itself must change** (P0 objective redesign). Interim levers if staying with the current loss: raise `COUNTERFACTUAL_WEIGHT`, apply the preflop tightness prior to the *counterfactual* call/raise targets (not just the taken action), or subtract a per-decision cost/rake so marginal `ev_call` lands below 0.
- **`argmax(Q)` with any uncalibrated head is inherently unstable.** Two opposite collapses (raise-everything, then call-everything) from small target changes confirm the actor-critic / regret-matching move (P0) is the real fix, not more target shaping.

## First structural step is now underway

Per [versioned-architecture-guardrails.md](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/versioned-architecture-guardrails.md) §9, **V12 is being stood up as the first fully-conforming version slice** under `versions/v12/` (self-contained `core/` + `self_play/` + `weights/`, a manifest, self-describing checkpoints, and a shared registry). This delivers P1's "versioned checkpoints + fail-loud loading" as a side effect and gives every subsequent V12 change a clean, isolated home. v8–v11 remain frozen in the legacy layout.

## Suggested sequencing

1. **Scaffold `versions/v12/` + manifest + registry + self-describing weights** (structural; in progress). Establishes the clean slice and kills the 159-vs-163 class.
2. **Inference-contract unification** (P0, cheap, fixes a live bug + the parked engine item). Ship first with the alignment test.
3. **Objective redesign** (P0, high-leverage) — prototype regret-matching or actor-critic on the v12 simulator; this is the main V12 bet and the real fix for the raise-/call-everything oscillation.
4. **Gameplay-based eval harness** (P1) — needed to objectively grade #3.
5. **De-swallowed exceptions** (P1) alongside the above.
6. **Bet sizing + NN league** (P2) — once the objective and contract are solid.

> [!NOTE]
> V11 remains the working baseline while V12 is built. Grade V12 candidates on **BB/100 vs a fixed benchmark lineup** (not the misleading single-state probes alone) plus the model-testing-suite guardrails.
