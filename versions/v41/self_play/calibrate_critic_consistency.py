"""Calibrate CRITIC_CONSISTENCY_MARGIN for V29's critic-consistency filter, against the ALREADY
TRAINED V28 checkpoint (V29 hadn't trained yet at the time this was run -- this samples the SAME
failure mode V28 inherited, since the underlying Q-value miscalibration is what motivates the
filter in the first place).

Sweeps the exact eq x stack grid check_deep_stack_ood_guard/allin_vs_nextbest_qgap use, extracts
RAW Q-values (bb-scale, same units regret_match_policy_torch's action_values operate in), and for
each candidate margin reports: (a) whether the known-worst cell's ALLIN gets zeroed, (b) how many
of the 5x5=25 grid cells would have >1 surviving action (mixing preserved) vs collapse to a single
action.

Run:  .venv/Scripts/python.exe -m versions.v41.self_play.calibrate_critic_consistency

See versions/v29/SPECS.md for the calibration finding this produced (the ALLIN-only filter design
was chosen specifically because an earlier all-pairs version, tested against this same grid,
collapsed legitimate raise-size mixing in 19-23/25 cells even at a lenient margin).
"""
import torch
from tools.model_verify.scenarios import build_ctx, run_policy
from shared.registry import get_manifest, load_model

manifest = get_manifest("v28")
model = load_model("v28", "expert_main.pth", device="cpu")
action_keys = manifest.action_space
allin_i = action_keys.index("allin")

grid = []
for eq in (0.35, 0.40, 0.43, 0.48, 0.55):
    for stack in (15, 20, 25, 30, 40):
        ctx = build_ctx(equity=eq, stack_bb=stack, pot_bb=2.5, call_bb=1.0, num_active_opp=1,
                        contract_version=manifest.contract_version)
        _, q = run_policy(model, ctx, action_keys, device="cpu")
        grid.append((eq, stack, q))

print(f"{'eq':>5} {'stack':>6} " + " ".join(f"{k:>9}" for k in action_keys))
for eq, stack, q in grid:
    print(f"{eq:>5} {stack:>6} " + " ".join(f"{q[k]:>9.3f}" for k in action_keys))

print()
for margin in (0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30):
    n_survive_1 = 0
    n_allin_survives_when_dominated = 0
    n_allin_was_argmax = 0
    for eq, stack, q in grid:
        vals = torch.tensor([q[k] for k in action_keys])
        best = vals.max()
        dominated = (best - vals) > margin
        n_surv = int((~dominated).sum().item())
        if n_surv <= 1:
            n_survive_1 += 1
        allin_dominated_by_gt = (best.item() - q[action_keys[allin_i]]) > 1e-6 and q[action_keys[allin_i]] != best.item()
        if allin_dominated_by_gt and not dominated[allin_i]:
            n_allin_survives_when_dominated += 1
        if q[action_keys[allin_i]] == best.item():
            n_allin_was_argmax += 1
    print(f"margin={margin:.2f}: cells collapsed-to-1-action={n_survive_1}/25, "
          f"allin-dominated-but-still-survives={n_allin_survives_when_dominated}, "
          f"allin-was-argmax={n_allin_was_argmax}/25")
