# Versioned Architecture Guardrails (CANONICAL)

**Date Recorded**: 2026-07-13
**Status**: 🟢 AUTHORITATIVE — this is the single source of truth for how model versions, data contracts, simulators, and live play are structured and isolated. It supersedes and expands [versioning-standards.md](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/versioning-standards.md) and builds on the 4-layer model in [workspace-architecture-overview.md](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/workspace-architecture-overview.md).

> **For AI agents:** Read this before touching any model, contract, simulator, training loop, weight file, or live-play code. The **Golden Rules** (§5) are binding. If a task appears to require breaking one, stop and flag it instead.

---

## 0. CURRENT CANONICAL FOUNDATION (added 2026-07-14)

- **`versions/v13/`** is the CURRENT best foundation — it inherits every v12_validated fix and
  adds **range-aware equity** (opponent adaptation): it dominates v12_validated on every field
  (Loose −0.9→**+29.4**, Tight −48.3→**−15.0**, Stations +2.7→**+7.5**). See
  [`versions/v13/VALIDATED_FINDINGS.md`](file:///c:/REPO/Antigravity/AIPoker/versions/v13/VALIDATED_FINDINGS.md).
  **Start new versions from v13.** Note the train/serve rule: range-aware equity must be computed
  the same way in training, eval, AND any live-play bridge, or the model silently mismatches.
- **`versions/v12_validated/`** remains the documented, frozen baseline for the *training-loop and
  architecture* fixes (target clip, counterfactual policy target, equity-primary architecture,
  realization discount, postflop-data field).

Both are test-verified and **must not be changed except through the testing workflow** documented in
[`versions/v12_validated/VALIDATED_FINDINGS.md`](file:///c:/REPO/Antigravity/AIPoker/versions/v12_validated/VALIDATED_FINDINGS.md) §4.

- **Start new versions from it:** `cp -r versions/v12_validated versions/vN` (per §6 checklist).
- **Locked fixes (see FINDINGS §1):** target clip = 40 (critic stability); counterfactual policy
  target (no fold-equity ratchet); equity-primary architecture (equity base + bottlenecked 16-dim
  card residual — cures postflop-blindness); realization discount `policy_tightness_bb`; postflop
  data via a loose calling-station field + exploration/bootstrap.
- **Validators (run before/after any change):** `overfit_sanity.py` (loop wiring),
  `inspect_policy_vs_target.py` (behavior), `eval_pure_policy.py` (the ONLY honest winrate —
  training-time BB/100 & VPIP are masked by the exploration anchor and must not be trusted).
- **Known open limitation → [`versions/v13/SPECS.md`](file:///c:/REPO/Antigravity/AIPoker/versions/v13/SPECS.md):** no opponent adaptation (loses to pure-nit fields); fix is range-aware equity.
- `versions/v12d/` is the DEPRECATED scratch copy that produced these findings — do not build on it.

---

## 1. Why this exists

The repo already partitions into 4 layers and clones some files per version, yet versions still got mixed and caused real failures this cycle:
- **159-vs-163 silent mismatch**: a stale contract's weights loaded into the new architecture, silently fell back to random init, and "output garbage" — because weights are flat and loading is not fail-loud.
- **Three inference conventions**: training, the simulator, and the live engine tokenized/queried the model differently — because there is no single enforced contract per version.
- **`is_v11` branching in shared runtime**: `engine.py`/`decision.py` hard-code version knowledge, so every new version edits shared files and risks the old ones.

The fix is not "more discipline" — it is **structure that makes mixing impossible**: a per-version vertical slice, a manifest that is the only dispatch point, namespaced self-describing weights, and a runtime that never knows a version number.

---

## 2. The mental model: a version is a frozen vertical slice

Each model version owns everything that can change its numeric behavior. The rest of the system is **version-agnostic** and dispatches to exactly one version via a manifest.

```
                       ┌─────────────────────────── VERSION-AGNOSTIC (shared, stable) ───────────────────────────┐
  Live:  Vision/OCR ──▶│ BoardState (neutral DTO) ──▶ runtime.registry[active] ──▶ engine(manifest) ──▶ action    │
  Train: Simulator  ──▶│ BoardState (neutral DTO) ──▶      "                 ──▶      "                            │
                       └───────────────────────────────────────────┬───────────────────────────────────────────┘
                                                                    │ manifest points at ONE version's slice:
                       ┌──────────────────────── VERSIONED SLICE (frozen once training starts) ───────────────────┐
                       │  contract.py  (BoardState → tensors, the ONLY tokenizer)                                  │
                       │  model.py     (architecture; state_dict is coupled to this file)                          │
                       │  simulator.py (self-play; produces BoardStates natively, like live vision)                │
                       │  train.py + config.yaml (loop, curriculum, hyperparams)                                   │
                       │  manifest.py  (id, context_dim, contract_version, class paths, weights_dir, status)       │
                       └──────────────────────────────────────────────────────────────────────────────────────────┘
```

**Key boundary — the data contract is the version seam.** `BoardState` is neutral (no tensors/padding/model concepts). The **Contract** is the single thing that turns a `BoardState` into model inputs, and **both live play and the training simulator go through the same version Contract.** That is what keeps train-time and live-time inference identical. Answer to "should we use data contracts in between?" — **yes, the Contract is the mandatory, single interface between the neutral state and every model.**

---

## 3. Target directory layout — one self-contained folder per version

Everything version-specific lives under `versions/vN/`, so a new version is a single `cp -r`. The version-**agnostic** shared layer lives outside any version folder (renamed `shared/` so nothing named `core` is ambiguous).

```
versions/                     # ONE self-contained folder per version — nothing shared lives here
  v11/
    core/                     # the version "brain"
      __init__.py
      manifest.py             # THE source of truth for v11 (see §4)
      model.py                # architecture     (from core/models/v11/poker_transformer_v11.py)
      contract.py             # data contract    (from core/bridge/v11/contract_v11.py)
    self_play/                # training sandbox for this version
      simulator.py            # (from tools/self_play/v11/six_max_simulator.py)
      train.py                # (from tools/self_play/v11/train_selfplay.py)
      config.yaml             # hyperparams, curriculum, feature spec
    weights/                  # THIS version's checkpoints, each self-describing (see §4b)
      expert_main.pth
      past_checkpoint.pth
    __init__.py
  v12/ ...                    # a copy of v11/, then modified

shared/                       # VERSION-AGNOSTIC ONLY — never contains a version number, never branches on one
  board_state.py              # NEUTRAL DTO (no tensors/padding/model concepts)
  poker/                      # evaluator, card utils — truly stable primitives
  vision/                     # OCR, table_state (live capture)
  runtime/
    registry.py               # discovers versions/*/core/manifest.py; maps version_id -> manifest
    engine.py                 # loads model+contract FROM a manifest; runs the forward pass
  live_bridge.py              # live path: BoardState buffer -> runtime[active_version] -> action
config.yaml                   # global: active_version = "v11"
```

Rules baked into this layout:
- **`versions/vN/` is the whole slice** — model, contract, manifest, simulator, train, config, and weights, together. Copying the folder copies the version.
- **`shared/` never imports a version and never branches on one.** It only ever loads a version through its manifest (§4a).
- **`versions/vN/self_play` imports only `versions/vN/core`** — never another version, never `shared/`'s version logic (it may use `shared/board_state.py` and `shared/poker`).
- **Weights live with their version** (`versions/vN/weights/`) so a version is fully reproducible from one folder. The new-version checklist (§6) clears them so a copy never inherits stale weights.

> The current repo still uses the split layout (`core/models/vX/`, `core/bridge/vX/`, `tools/self_play/vX/`, flat `core/weights/`). Treat that as the slice today and follow the same rules; the physical consolidation into `versions/vN/` happens at **V12 bring-up** (§9) — never by mutating a running or shipped version.

---

## 4. The two enforcement mechanisms

### 4a. The Version Manifest (single dispatch point)
Every version ships a `manifest.py`. The runtime loads a version **only** through it. Adding a version = add a manifest; adding a version must touch **zero** runtime files.

```python
# versions/v11/core/manifest.py
MANIFEST = VersionManifest(
    version_id       = "v11",
    context_dim      = 35,                 # feature-vector width; MUST match the model's state_proj input
    contract_version = 2,                  # bump whenever the tensor schema changes (padding, features, tokens)
    action_space     = ["fold", "call", "raise"],
    model_class      = "versions.v11.core.model:PokerEVModelV11",
    contract_class   = "versions.v11.core.contract:ContractV11",
    weights_dir      = "versions/v11/weights",
    status           = "active",           # active | frozen | deprecated
)
```

### 4b. Self-describing checkpoints + fail-loud loading (kills 159-vs-163)
Never save a bare `state_dict`. Never load one silently.

```python
# saving
torch.save({
    "state_dict":       model.state_dict(),
    "version_id":       MANIFEST.version_id,
    "context_dim":      MANIFEST.context_dim,
    "contract_version": MANIFEST.contract_version,
    "hands_trained":    hands_done,
    "git_sha":          <sha>,
    "saved_at":         <iso8601>,
}, path)

# loading (in runtime.engine)
ckpt = torch.load(path)
assert ckpt["context_dim"] == MANIFEST.context_dim, \
    f"Checkpoint {path} is contract v{ckpt['contract_version']} (dim {ckpt['context_dim']}), " \
    f"but {MANIFEST.version_id} expects dim {MANIFEST.context_dim}. Refusing to load."
model.load_state_dict(ckpt["state_dict"])   # strict=True
```
A mismatch must **raise**, never warn-and-continue, and never fall back to random weights.

---

## 5. Golden Rules (binding — for humans and AI agents)

1. **A version is frozen once its training starts or it ships.** Never edit `vN`'s model/contract/simulator/train/manifest to change behavior. To change anything, make `vN+1`.
2. **To start a new version, copy — never mutate.** Clone the previous slice to `vN+1`, then modify only the copy. (Checklist in §6.)
3. **No cross-version imports.** `v11` code must never import `v10` (or vice-versa). Each slice is self-contained.
4. **One contract per version, used by BOTH training and live.** The Contract is the only `BoardState → tensors` path. Do not build ad-hoc tensorization in a simulator, engine, diagnostic, or live bridge. Do not add legacy feature-toggles to an old contract — make a new one.
5. **The runtime never branches on a version number.** No `if is_v11:` in `engine.py`/`decision.py`/eval tools. Dispatch through the manifest registry only.
6. **Weights live with their version and are self-describing; loading is fail-loud.** `versions/<version>/weights/…`, metadata embedded, assert on load. Never silently init random weights on load failure.
7. **`BoardState` is the neutral, version-agnostic DTO.** No tensors, padding, masks, or model concepts inside it. Changing its fields is a breaking change → bump `contract_version` (and therefore a new version).
8. **Live play is config-selected dispatch, not hardcoding.** The live path picks the version from `active_version` in config. Switching the deployed model = change one config value; touch no version code.
9. **Evaluation uses the same contract/engine as training.** No separate inference path for the health suite or diagnostics (this caused the misleading raise-everything readings). Route everything through `runtime.engine` + the version Contract.
10. **Deprecate, don't delete or repurpose.** Old versions and their weights stay for reproducibility. Mark `status="deprecated"`; never overwrite `expert_vN_*.pth` or reuse a version id.

---

## 6. "Start a new version" checklist (streamlined)

```
1. cp -r versions/vN  versions/vN+1          # copies the WHOLE slice: core/, self_play/, weights/
2. rm -f versions/vN+1/weights/*             # start fresh — never inherit vN's weights
3. Edit versions/vN+1/core/manifest.py:
     - version_id       -> "vN+1"
     - weights_dir      -> "versions/vN+1/weights"
     - contract_version -> bump ONLY if the tensor schema changed
     - status           -> "active"
4. (If auto-discovery isn't used) register vN+1 in shared/runtime/registry.py
5. Make ALL changes in versions/vN+1/ only. Never touch versions/vN/.
6. Rename the model class if desired (e.g. PokerEVModelV{N+1}) — the state_dict is coupled to the class,
   so a distinct class name is an extra guard against accidental cross-loading.
7. When ready to deploy: set active_version = "vN+1" in the global config.
```
One folder in, one folder out — no second copy from a different root, no cross-root path edits.
An AI agent asked to "improve the model" or "fix behavior X" should default to **step 1–2 (branch to the next version)**, not editing a shipped version in place — unless the task is explicitly a hotfix to an unreleased, in-progress version.

---

## 7. Live-play separation (explicit)

- **Live Vision Sandbox** (`PHPHelp.py` + `shared/vision/`): stateless; emits one `BoardState` per turn. No ML, no history.
- **Live Bridge** (`shared/live_bridge.py`): stateful hand-history buffer; assembles the `BoardState` sequence and calls `shared/runtime/engine.py` with the **active version's** Contract. This is the ONLY place live inference happens.
- **Training Simulator** (`versions/vX/self_play/simulator.py`): produces `BoardState`s natively (behaves like live vision) and calls the **same** version Contract. It must not duplicate contract logic.
- Live and training therefore share exactly one tokenizer per version. If they ever diverge, that is a bug (add the §8 alignment test).

---

## 8. Required guardrail test (prevents silent divergence)

Add a CI/pre-commit check per active version: build a fixed `BoardState`, run it through the version Contract + `runtime.engine`, and assert the tensors/`q_vals` are **identical** whether produced via the training path or the live-bridge path. This is the mechanical guarantee that "trains fine, plays badly" cannot happen again.

---

## 9. Migration from the current split layout → `versions/` (incremental, non-breaking)

Today the slice is scattered: `core/models/vX/` (model), `core/bridge/vX/` (contract), `tools/self_play/vX/` (sim+train), flat `core/weights/`. Consolidate at **V12 bring-up**, with older versions left untouched (or moved verbatim, never mutated). Do it in low-risk order — the early steps deliver the biggest wins without the full move:

1. **Add `manifest.py` + self-describing save/load + the load-time assert** for v11 (describe the existing files in place). *Biggest immediate win — kills the 159-vs-163 silent mismatch.*
2. **Move the `is_v11` dispatch** out of `engine.py`/`decision.py` into a `shared/runtime/registry.py` driven by manifests.
3. **Stand up the `versions/` tree**: create `versions/v12/` fresh in the new shape (`core/`, `self_play/`, `weights/`) as the **first fully-conforming slice**. Prove the layout on v12.
4. **Backfill v11** by moving its files into `versions/v11/{core,self_play,weights}` and updating imports — do this only when v11 is idle (no training run in flight), never while a run is active.
5. **Rename the shared layer** (`core/` → `shared/`) last, since it touches the most imports; or leave `core/` as the shared root if a rename is too costly — the key invariant is that the shared layer holds **no** version-specific code.
6. Legacy v8–v10 can be deleted.

> Do NOT move files for a version while its training run is active — imports and the running process reference the current paths. Consolidate during a bring-up window.

---

## 10. Cross-references
- 4-layer architecture & sandbox roles: `workspace-architecture-overview.md`
- Original cloning SOP (now a subset of this doc): `versioning-standards.md`
- Why the objective/inference need rework: `V12/design-proposal.md`
- Concrete failures this prevents: `V11/issues-and-fixes.md` (159-vs-163, three inference conventions, swallowed KeyError)
