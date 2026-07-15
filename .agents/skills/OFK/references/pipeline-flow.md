# Pipeline Flow — Simulation/Training vs Live Play

**Date Recorded**: 2026-07-15
**Related Files**: [decision.py](file:///c:/REPO/Antigravity/AIPoker/core/decision.py) · [action_executor.py](file:///c:/REPO/Antigravity/AIPoker/core/action_executor.py) · [PHPHelp.py](file:///c:/REPO/Antigravity/AIPoker/PHPHelp.py) · [simulator.py](file:///c:/REPO/Antigravity/AIPoker/versions/v15/self_play/simulator.py) · [train.py](file:///c:/REPO/Antigravity/AIPoker/versions/v15/self_play/train.py) · [contract.py](file:///c:/REPO/Antigravity/AIPoker/versions/v13/core/contract.py)

## Context
Two Mermaid flow diagrams that map every condition/logic/data tweak in (1) the self-play sim/training
pipeline and (2) the live-play path, colour-coded by scope (🌐 global · 🎲 board · 👤 player-state).
Boxes are ID'd so notes/commits can reference them (e.g. "changed B1", "LD4 sizing"). Complements
[simulation_architecture.md](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/simulation_architecture.md)
and [decision-pipeline-tracing-and-gui-overrides.md](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/decision-pipeline-tracing-and-gui-overrides.md).

## Guidelines
**MAINTAIN THIS** whenever a condition/logic/data tweak is added or changed in the sim/train pipeline
(`versions/<v>/self_play/*`) or the live path (`core/*`, `PHPHelp.py`). Current: **V15 live**
(`Herocules (v15 DoN)`); v14/v13 fallbacks. Open tweaks tracked in `versions/v16/SPECS.md`.

---

## 1. Simulation & Training  (`versions/v15/self_play/`)

```mermaid
flowchart TD
    subgraph GLOBAL["🌐 GLOBAL — run/config (config.yaml)"]
        G1["G1 target_hands 200k · batch/epochs"]
        G2["G2 bootstrap α: 1.0 → 0 by 30k hands"]
        G3["G3 exploration ε=5% · samples all 6 incl ALL-IN"]
        G4["G4 rollout temp=1.0 (policy_temperature)"]
        G5["G5 range_aware_equity = ON"]
        G6["G6 counterfactual target · tightness 2.0bb · clip 40bb"]
        G7["G7 action space F/C/r33/r66/rPot/AI"]
    end
    subgraph BOARD["🎲 BOARD — per hand"]
        B1["B1 stack_depth_mix (DoN 5-14 / 14-30 / 30-50bb)"]
        B2["B2 opponent pool sample per seat: fish/tag/nit/past"]
        B3["B3 betting loop — pot / to_call evolve"]
        B4["B4 MC per-size EV targets (_mc_target_evs_sized)"]
    end
    subgraph PLAYER["👤 PLAYER-STATE — per seat"]
        P1["P1 opponent size-aware fold: bar = pot_odds + (FtP-0.5)·0.30"]
        P2["P2 frozen_v14 pinned in 'past' seat (freeze_past_self)"]
        P3["P3 range-aware equity vs each active opp color range"]
        P4["P4 HUD vpip/agg colour + stack"]
    end
    subgraph HERO["♠ HERO decision (in sim)"]
        H1["H1 _hero_decide: ε-rand / model / heuristic anchor"]
        H2["H2 _query_model_decide: softmax(logits/temp) + fold-when-free mask + sample"]
        H3["H3 _raise_size_for_fraction: bucket → chips (min-raise floor, stack cap)"]
    end
    subgraph TRAIN["🧠 TRAIN"]
        T1["T1 vectorize → 6-wide targets"]
        T2["T2 critic Q-loss + actor regret-matching loss"]
        T3["T3 past_self ckpt (skipped when frozen)"]
        W["weights/expert_main.pth"]
    end

    GLOBAL --> B1
    G7 --> B4
    B1 --> B3
    B2 --> PLAYER
    PLAYER --> B3
    P3 --> H2
    B3 --> H1 --> H2 --> H3 --> B3
    B3 --> B4 --> T1 --> T2 --> W
    T2 -.->|retrain source| T3

    classDef g fill:#12314f,stroke:#4a9eff,color:#eaf2ff;
    classDef b fill:#33265a,stroke:#a06fff,color:#f2ecff;
    classDef p fill:#123f31,stroke:#2eb85c,color:#e7fff2;
    classDef h fill:#4a3410,stroke:#e0a020,color:#fff6e0;
    classDef t fill:#3a1220,stroke:#e05070,color:#ffe6ee;
    class G1,G2,G3,G4,G5,G6,G7 g;
    class B1,B2,B3,B4 b;
    class P1,P2,P3,P4 p;
    class H1,H2,H3 h;
    class T1,T2,T3,W t;
```

**Notes.** 🌐 **G1-G7** are fixed for the whole run (config.yaml): action space, exploration, the
counterfactual/tightness/clip target recipe, range-aware equity. 🎲 **B1-B4** are re-rolled every
hand — B1 samples the DoN depth, B2 fills each opponent seat, B3 runs the betting, B4 scores *every*
size counterfactually. 👤 **P1-P4** are per opponent seat: P1 is the size-aware fold response (bigger
bet → more folds), P2 pins the frozen-V14 expert, P3 feeds hero equity vs that seat's colour range.
♠ Hero acts via **H1→H2→H3** (mask + sample + size); **T1-T3** turn the hand into 6-wide targets and
weights.

---

## 2. Live Play  (`core/decision.py`, `core/action_executor.py`, `PHPHelp.py`)

```mermaid
flowchart TD
    subgraph LGLOBAL["🌐 GLOBAL — decision engine"]
        LG1["LG1 active model v15 (v14/v13 fallback)"]
        LG2["LG2 LIVE_POLICY_TEMPERATURE = 0.5"]
        LG3["LG3 action space + V14_RAISE_FRAC"]
    end
    subgraph LBOARD["🎲 BOARD — per turn (vision)"]
        LB1["LB1 vision OCR → table_state"]
        LB2["LB2 board_state: pot, call_amount (OCR call btn), bb, cards, street"]
        LB3["LB3 range-aware equity vs active opp colours"]
        LB4["LB4 hand_history_buffer (per-turn snapshots)"]
    end
    subgraph LPLAYER["👤 PLAYER-STATE — per seat (vision)"]
        LP1["LP1 HUD vpip/agg colour, stack, is_active"]
    end
    subgraph LDEC["♠ DECISION"]
        LD1["LD1 bridge → hole/board/ctx/act tensors"]
        LD2["LD2 predict_ev → 6-way policy"]
        LD3["LD3 sharpen temp0.5 + fold-when-free mask + sample"]
        LD4["LD4 raise bucket → _v14_size_to_slider → RAISE_SLIDER_x"]
        LD5["LD5 math-engine BYPASS for sized models"]
    end
    subgraph LEXE["🖱 EXECUTE + LOG"]
        LE1["LE1 action_executor: slider drag + button click"]
        LE2["LE2 telemetry → history/{board_id}/turns.jsonl (to_call from tensor)"]
    end

    LB1 --> LB2 --> LP1 --> LB3 --> LB4 --> LD1
    LGLOBAL --> LD2
    LD1 --> LD2 --> LD3 --> LD4 --> LE1 --> LE2
    LD3 -.-> LD5

    classDef g fill:#12314f,stroke:#4a9eff,color:#eaf2ff;
    classDef b fill:#33265a,stroke:#a06fff,color:#f2ecff;
    classDef p fill:#123f31,stroke:#2eb85c,color:#e7fff2;
    classDef d fill:#4a3410,stroke:#e0a020,color:#fff6e0;
    classDef e fill:#3a1220,stroke:#e05070,color:#ffe6ee;
    class LG1,LG2,LG3 g;
    class LB1,LB2,LB3,LB4 b;
    class LP1 p;
    class LD1,LD2,LD3,LD4,LD5 d;
    class LE1,LE2 e;
```

**Notes.** 🌐 **LG1-LG3** are engine constants: which model is active and the serve temperature/action
space (must mirror the training recipe — G4/G7). 🎲 **LB1-LB4** rebuild each turn from vision: LB2's
`call_amount` is OCR'd off the Call button, LB3 recomputes range-aware equity (mirrors P3), LB4 keeps
the per-turn snapshot sequence the model reads. 👤 **LP1** is the per-seat HUD/stack read that feeds
LB3. ♠ **LD1-LD5** mirror the sim's H2/H3 exactly (mask + sample + slider sizing; math engine bypassed)
so train≡serve; **LE1** drags the slider then clicks, **LE2** logs the turn for review/F12.

---

### Train ≡ Serve invariants (must stay paired across both diagrams)
- Sampling temperature: **G4** (rollout 1.0) ↔ **LD3/LG2** (serve 0.5) — eval must match serve temp.
- Fold-when-free mask: **H2** ↔ **LD3**.  · Raise sizing: **H3** (`_raise_size_for_fraction`) ↔ **LD4** (`_v14_size_to_slider`).
- Range-aware equity: **P3** ↔ **LB3** (same `compute_range_aware_equity`).  · Action space: **G7** ↔ **LG3**.
