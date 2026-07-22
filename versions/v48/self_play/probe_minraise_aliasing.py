"""[V47 post-training, scope-boundary measurement] Min-raise-floor bucket aliasing on the
TRAINED model.

The V47 run collapsed only the ALL-IN chip-identity aliasing (Change 2 / M9). The fable
resolution log (Tier 6, T-M9) measured a SECOND aliasing family this run deliberately did NOT
collapse (user decision 2026-07-22, recorded in SPECS): at short stacks the legal min-raise
floor makes raise_0/raise_1 (and sometimes raise_2) chip-identical WITHOUT being all-in --
40.7% of sized decision points overall, 56% preflop, in the V43-era data. Those buckets still
train separate MC targets and each carries its own regret (triple-counting in the
regret-matching normalization), and serve does NOT merge them.

This probe answers, on the trained v47 expert_main (frozen V44 as the incumbent control):
  1. PREVALENCE on the V47 curriculum -- what fraction of preflop raise decisions have >=2
     chip-identical non-allin buckets, using the sim's own `_raise_size_for_fraction`
     (curriculum-weighted over stack_depth_mix, and by stack band).
  2. TARGET CONSISTENCY -- at aliased cells, the intra-group Q-value spread (buckets that ARE
     the same physical action should have learned ~the same Q; spread = residual MC-noise /
     triple-count distortion), against the inter-group spread as the yardstick.
  3. ENTRY-MASS EFFECT -- combined policy mass on the aliased group and total raise mass,
     V44 vs V47 at identical cells (the mechanism by which triple-counted regret could inflate
     preflop entry -- the Change-0 sequencing rule's trigger).

Run:  .venv/Scripts/python.exe versions/v48/self_play/probe_minraise_aliasing.py
"""
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from shared.registry import load_model                      # noqa: E402
from tools.model_verify.scenarios import build_ctx, run_policy   # noqa: E402
from versions.v48.self_play.simulator import SixMaxSimulator     # noqa: E402

ACTION_KEYS = ['fold', 'call', 'raise_0', 'raise_1', 'raise_2', 'raise_3']
RAISE_KEYS = ACTION_KEYS[2:]
STACK_MIX = [[2, 5, 0.08], [5, 8, 0.07], [5, 14, 0.35], [14, 30, 0.25],
             [30, 60, 0.17], [10, 100, 0.08]]
# (label, pot_bb, to_call_bb) -- the preflop geometries where the floor binds hardest.
GEOMS = [("open (blinds only)", 1.5, 1.0),
         ("vs 2.5x open", 4.0, 2.5),
         ("vs 3.5x open", 5.0, 3.5),
         ("BB vs limp", 2.0, 0.0)]


def bucket_sizes(sim, pot, to_call, stack):
    return [sim._raise_size_for_fraction(f, pot, to_call, stack, min_increment=1.0)
            for f in sim.raise_fracs]


def classify_cell(sizes, stack):
    """-> (kind, groups) where kind in {'clean','allin_alias','minraise_alias'} and groups maps
    a chip size to the list of bucket indices resolving to it. minraise_alias = >=2 buckets
    share a chip size that is NOT the shove (the family this run did not collapse)."""
    groups = defaultdict(list)
    for j, s in enumerate(sizes):
        groups[round(s, 6)].append(j)
    minraise = any(len(js) >= 2 and sz < stack - 1e-9 for sz, js in groups.items())
    allin = any(len(js) >= 2 and sz >= stack - 1e-9 for sz, js in groups.items())
    kind = 'minraise_alias' if minraise else ('allin_alias' if allin else 'clean')
    return kind, dict(groups)


def curriculum_prevalence(sim, n=20000, seed=4747):
    rng = random.Random(seed)
    bands, weights = [m[:2] for m in STACK_MIX], [m[2] for m in STACK_MIX]
    counts = defaultdict(int)
    by_band = defaultdict(lambda: defaultdict(int))
    for _ in range(n):
        lo, hi = rng.choices(bands, weights=weights)[0]
        stack = rng.uniform(lo, hi)
        label, pot, to_call = rng.choice(GEOMS)
        kind, _ = classify_cell(bucket_sizes(sim, pot, to_call, stack), stack)
        counts[kind] += 1
        band_key = f"{lo}-{hi}bb"
        by_band[band_key][kind] += 1
        by_band[band_key]['n'] += 1
    return counts, by_band


def probe_models(sim, models):
    """Model behavior at a fixed grid; returns rows of per-cell measurements per model."""
    stacks = [2.5, 3.5, 5, 7, 9, 12, 15, 20, 30, 50]
    equities = [0.30, 0.45, 0.60]
    rows = []
    for stack in stacks:
        for label, pot, to_call in GEOMS:
            sizes = bucket_sizes(sim, pot, to_call, stack)
            kind, groups = classify_cell(sizes, stack)
            # the aliased (non-shove) group's bucket indices, if any
            al_js = next((js for sz, js in groups.items()
                          if len(js) >= 2 and sz < stack - 1e-9), None)
            for eq in equities:
                ctx = build_ctx(eq, stack, pot, to_call, position=2, street=0,
                                num_active_opp=2, contract_version=9)
                for name, model in models.items():
                    pol, q = run_policy(model, ctx, ACTION_KEYS)
                    raise_mass = sum(pol[k] for k in RAISE_KEYS)
                    row = {'model': name, 'kind': kind, 'stack': stack, 'geom': label,
                           'eq': eq, 'raise_mass': raise_mass, 'call': pol['call'],
                           'fold': pol['fold']}
                    if al_js:
                        grp = [pol[ACTION_KEYS[2 + j]] for j in al_js]
                        qg = [q[ACTION_KEYS[2 + j]] for j in al_js]
                        q_all = [q[k] for k in ACTION_KEYS]
                        row['alias_mass'] = sum(grp)
                        row['alias_n'] = len(al_js)
                        row['q_intra'] = max(qg) - min(qg)
                        row['q_inter'] = max(q_all) - min(q_all)
                    rows.append(row)
    return rows


def summarize(rows):
    def agg(pred, field):
        vals = [r[field] for r in rows if pred(r) and field in r]
        return (sum(vals) / len(vals), len(vals)) if vals else (float('nan'), 0)

    print("\n" + "=" * 78)
    print("Model behavior at ALIASED vs CLEAN preflop cells (grid x eq {0.30,0.45,0.60})")
    print("=" * 78)
    for name in sorted({r['model'] for r in rows}):
        for kind in ('minraise_alias', 'clean'):
            rm, n = agg(lambda r: r['model'] == name and r['kind'] == kind, 'raise_mass')
            fm, _ = agg(lambda r: r['model'] == name and r['kind'] == kind, 'fold')
            cm, _ = agg(lambda r: r['model'] == name and r['kind'] == kind, 'call')
            print(f"  {name:<12} {kind:<15} n={n:>3}  raise_mass {rm:.3f} | call {cm:.3f} | fold {fm:.3f}")
        am, n = agg(lambda r: r['model'] == name and r.get('alias_n'), 'alias_mass')
        qi, _ = agg(lambda r: r['model'] == name and r.get('alias_n'), 'q_intra')
        qe, _ = agg(lambda r: r['model'] == name and r.get('alias_n'), 'q_inter')
        ratio = qi / qe if qe and qe == qe and qe > 0 else float('nan')
        print(f"  {name:<12} aliased-group   n={n:>3}  combined mass {am:.3f} | "
              f"intra-group Q spread {qi:.4f} vs inter-action {qe:.4f} (ratio {ratio:.2f})")


def main():
    sim = SixMaxSimulator(bb_size=1.0, equity_sims=10, hero_personality='main',
                          bootstrap_alpha=0.0)
    counts, by_band = curriculum_prevalence(sim)
    total = sum(v for k, v in counts.items())
    print("=" * 78)
    print("PREVALENCE -- curriculum-weighted preflop raise decisions (20k draws)")
    print("=" * 78)
    for kind in ('minraise_alias', 'allin_alias', 'clean'):
        print(f"  {kind:<15} {counts.get(kind, 0) / total:.1%}")
    print("-" * 78)
    for band in sorted(by_band, key=lambda b: float(b.split('-')[0])):
        c = by_band[band]
        n = c['n']
        print(f"  {band:<10} minraise {c.get('minraise_alias', 0) / n:.1%} | "
              f"allin {c.get('allin_alias', 0) / n:.1%} | clean {c.get('clean', 0) / n:.1%}  (n={n})")

    models = {'v48': load_model('v48', 'expert_main.pth'),
              'v44_frozen': load_model('v48', 'frozen_v47.pth')}
    rows = probe_models(sim, models)
    summarize(rows)

    print("\nReading guide: 'minraise_alias' cells are where the un-collapsed aliasing lives.")
    print("  - If v47's raise_mass at aliased cells inflated vs v44_frozen (and vs its own")
    print("    clean cells), the triple-counted regret is leaking into entry frequency ->")
    print("    supports Change 0 as a V47.1 prerequisite (the agreed sequencing rule).")
    print("  - intra-group Q spread ~ inter-action spread would mean the net never learned the")
    print("    aliased buckets are one physical action (target inconsistency is being memorized).")


if __name__ == '__main__':
    main()
