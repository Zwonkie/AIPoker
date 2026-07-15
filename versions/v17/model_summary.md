# Model Architecture Summary & Gap Analysis (as of V15/V16 lineage, written 2026-07-15)

## What it is

`PokerEVModelV4` (`versions/v15/core/model.py`, unchanged in v16) is a hybrid, not a
textbook implementation of any single named method:

- **Backbone**: a causal-masked Transformer encoder over a sequence of (state, prev-action)
  tokens per hand — card embeddings + 35-dim context features projected to `d_model`,
  positional embedding, standard `nn.TransformerEncoder`.
- **Dual heads**: a Critic (`head` → `q_vals`, per-action counterfactual EV) and an Actor
  (`head_policy` → `policy_logits`, a target action distribution).
- **Equity-primary structure**: both heads are `base(equity, pot_odds, pot, call, street) +
  residual(transformer output)`, zero-init on the residual so the model starts as a pure
  equity player and learns card/board detail as a correction.
- **Critic target**: Monte Carlo rollout EV per action (`_mc_target_evs_sized` /
  `_mc_target_evs`), sampled against a fixed heuristic bot pool (fish/tag/nit/station) +
  one frozen past-self snapshot.
- **Actor target**: one-step **regret matching** over the critic's action values
  (`regret_match_policy`, `versions/v15/self_play/train.py:121`) — proportional to positive
  regret vs. the mean action value, sharpened by a temperature — distilled into the actor via
  cross-entropy.

## How it diverges from the named methods it resembles

### vs. Decision Transformer (Chen et al. 2021)
The DT paper's defining mechanism is **return-conditioning**: a target return-to-go fed as an
input token, so the model can be "prompted" toward a desired outcome. This codebase has no
return-to-go token anywhere (confirmed: no `return_to_go`/`rtg` in the code). The Transformer
here is used purely as a sequence feature extractor for an actor-critic — despite the
docstring calling it a "Decision Transformer," it is not one in the technical sense. Not
necessarily a gap (episodic bb/100 optimization doesn't need return-conditioning), but worth
knowing the name is aspirational, not literal.

### vs. Deep CFR / true CFR
The largest structural gaps live here:
- **No iteration + averaging.** Real CFR only converges via the *average* strategy across
  many iterations — a single iteration does not converge to Nash. This model computes
  **one-step** regret matching per decision point and distills it directly into the actor.
  There is no reservoir buffer, no separate average-strategy network, no iteration loop.
- **Values are population EVs, not equilibrium-relative regrets.** True CFR computes
  counterfactual values via full/sampled game-tree traversal weighted by opponent reach
  probability against the *current* strategy. Here they're plain Monte Carlo rollouts against
  a fixed heuristic bot pool. The "regret" signal measures exploitation of those specific
  bots, not distance from Nash equilibrium.
- **No exploitability metric, ever.** There is no best-response solver checking how
  exploitable the trained policy is — only bb/100 vs. the bot pool used in training. A model
  can look strong against the 4 trained archetypes and still be trivially exploitable by an
  untrained style.
- **No real-time search.** Pluribus's actual edge over a pure blueprint network is
  depth-limited search *at decision time* to refine the blueprint for the exact spot. This
  model's live decision is a single forward pass — zero search. Likely the single largest
  capability gap versus the SOTA poker-AI lineage the code references (docstring literally
  says "for Pluribus V4").

### vs. standard actor-critic (A2C/PPO)
- Critic target is a **single Monte Carlo rollout to terminal**, not bootstrapped/TD — higher
  variance, no cross-street value propagation.
- Actor loss is **cross-entropy toward a precomputed target distribution**, not a
  policy-gradient/advantage-weighted update — structurally closer to behavior cloning onto a
  computed target than classic policy gradient. No PPO clipping, no trust region, no GAE.

### Self-play population
Fixed heuristic archetypes (fish/tag/nit/station) + one frozen past snapshot — not a growing
league of past selves, and no curriculum where the population itself gets harder over time.

## Cross-reference to tracked V16 gaps

The V16 roadmap items are concrete instances of these structural gaps, not independent bugs:
- **P4** (no opponent-aware preflop entry range) ↔ no opponent-conditioned policy input.
- **P6** (no opponent-action attribution — `act` tensor is hero-only) ↔ CFR/DeepStack/Pluribus
  all require full public action history, not just the hero's own actions.
- **P5** (bet-size blindness — history tokens size-blind, `to_call` normalization crushes
  small raises) ↔ no bucketed bet-size abstraction the way classical poker-AI game trees use.

## Priority takeaway

If prioritizing "missing features" by leverage: the highest-value gap relative to genuine
game-theoretic soundness is the **absence of any exploitability check** — everything else in
this pipeline optimizes win-rate against a hand-picked population, with no signal on how safe
that policy is against an opponent style never simulated.
