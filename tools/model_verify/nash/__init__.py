"""VAL-1 external ground-truth axis: heads-up Nash push/fold spot-checks.

Fully self-contained plug-in for model_verify. The only integration points into the
rest of the tool are (a) one import + one FAST_CHECKS entry + one CHECK_DOCS entry in
tools/model_verify/checks.py, and (b) this folder. Nothing in versions/*, the simulator,
or train.py is touched.

- nash_chart.json         : static bundled data (curated unambiguous Nash cells + baked-in equities)
- precompute_equities.py  : one-time OFFLINE author-time script that (re)builds nash_chart.json
                            using core.evaluator only. Never imported at model_verify runtime.
- pushfold_check.py       : the runtime FAST check -- pure lookup + run_policy, zero equity deps.
"""
