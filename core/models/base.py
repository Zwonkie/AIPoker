class PokerModelInterface:
    """[v46_legacySweep] What a LIVE-registrable engine actually provides.

    The old ABC here mandated a 15-argument `predict_action(...)` -- the pre-boundary interface
    where the CALLER interpreted the table for the model. That inverted with V45_liveHandover
    (LiveObservation -> version-owned adapter) and no sized engine ever implemented it; it was
    kept only by the legacy heuristic engine, which still defines it and still subclasses this
    for compatibility.

    A registrable engine today declares (see core/models/v44_engine.py for the pattern):
      predict_ev(hole, board, ctx, act) -> {action_key: prob}   # the 6-way actor policy
      loaded: bool          # False == weights failed to load; decision.py refuses to serve it
      is_sized = True       # sized 6-action actor policy (the only supported kind)
      display_tag: str      # HUD/reason-line label
      has_aux: bool         # whether last_aux carries a trained opponent-read
      make_bridge()         # its OWN tensor contract instance
      live_features()       # its OWN live feature implementations (equity/hand_strength/...)
      make_live_adapter(decision_engine)  # OPTIONAL custom adapter (BaseLiveAdapter subclass)

    Nothing is enforced abstractly on purpose: `decision.py` fail-louds at registration/decision
    time on exactly the declarations it needs, which is the check that actually matters.
    """
    pass
