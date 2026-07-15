# V16_vsNN — exploratory side-branch, NOT a production candidate

**Purpose (2026-07-15):** a learning exercise, not a deploy candidate. Warm-started from
`versions/v16/weights/expert_main.pth` (110,057 hands, the P4-fixed preflop CALL/FOLD equity
basis — see `versions/v16/SPECS.md`), continued for 75,000 MORE hands (110,057 → 185,057) under a
deliberately different training population: a 3-seat table (hero + 2 opponents) with **NO
heuristic bots at all** — both opponents are neural nets:

- `past_model` = a true self-play lagged mirror (`freeze_past_self: false`) — "itself".
- `nit_model` = repurposed (was unused/dormant — see `versions/v16/self_play/train.py`'s
  `nit_model_path` plumbing, already fully wired through the worker pool, just never called with a
  real path) to load a STATIC copy of `versions/v15/weights/expert_main.pth` — "V15".

**Why:** raised directly by the "is the hero overfitting to the training pool's specific
heuristic-formula shape" discussion (`versions/v16/SPECS.md` P7 + the `beats_offformula_stress`
model_verify check). Training against genuinely skilled NNs instead of `FuzzyPlayerArchetype`
heuristics is the most direct way to observe whether/how behavior changes when the opponent
population can't be reverse-engineered from a known closed-form threshold formula. Purely
observational — see how it behaves, not intended to replace the v16 main line.

Continuing the hand-count numbering (110,057 rather than resetting to 0) deliberately skips the
bootstrap-anchor phase (`hands_done < 30000`) — the hero is already competent, so re-introducing
heuristic-anchor influence would work against the point of the experiment.
