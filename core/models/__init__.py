# [live2 cleanup 2026-07-24] The legacy PHPHelp-era engines (base/heuristic/engine) moved to
# attic/phphelp_stack/ -- nothing imported them since the v46 legacy sweep. Version engines
# (v48_engine.py, v50_engine.py, ...) are imported directly by core/decision.py; this package
# init deliberately exports nothing.
