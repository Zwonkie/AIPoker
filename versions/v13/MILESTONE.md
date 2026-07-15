# V13 — MILESTONE (kept reference / fallback)

**Tagged milestone 2026-07-14.** V13 is the first version that plays **reasonably well live** — the
sound foundation. Keep it as a known-good reference to roll back to.

- Manifest: `versions/v13/core/manifest.py` has `milestone=True` (schema: `shared/manifest.py`).
- **Do NOT delete `versions/v13/weights/expert_main.pth`** — it is the fallback checkpoint.
- What it is: equity-primary architecture + range-aware equity (opponent adaptation), deployed live
  as "Herocules (v13 Range-Aware)". Full detail in [VALIDATED_FINDINGS.md](VALIDATED_FINDINGS.md).
- Known limitations carried into V14 (see [../v14/SPECS.md](../v14/SPECS.md)): preflop policy
  flattening (§6), flat-100BB training vs short-stack live play, no opponent-action/outcome logging.

**To also mark it in git (optional, recommended):**
```
git tag -a v13-milestone -m "V13: first live-viable foundation (range-aware equity)"
```

V14 is a NEW folder (`versions/v14/`) copied from here — this folder stays frozen as the reference.
