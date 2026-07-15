* **Zero-Trust Engineering Mandate:** When debugging model behavior or simulation logic, assume all existing code, vectorization contracts, and mathematical equations are flawed until proven otherwise. Cross-reference the exact shapes, indices, and padding logic between Training (data loaders) and Inference (bridges/contracts). A model is only as smart as the exact tensor footprint it receives; any misalignment between the two phases is fatal. 

* **Versioning:** read .agents/skills/OFK/references/versioned-architecture-guardrails.md first. 

* **When starting a new training session** remember to read `.agents\skills\monitor-training-session` (the monitoring SOP) and start the dashboard via `tools\training_monitor\start_dashboard.ps1` (standalone tool, not skill-owned).

* **Training Telemetry Output:** All training scripts, regardless of model version, MUST output or tee their telemetry and logs to the single fixed path `active_training.log` in the repository root directory. The dashboard (`tools\training_monitor\`) only ever watches that one path — it does not search for logs — so redirecting `sys.stdout` to this exact file is mandatory for every version's `train.py`. Run `tools\training_monitor\check_telemetry_contract.py` after scaffolding a new version to verify it still complies.
