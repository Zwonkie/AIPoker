* **Zero-Trust Engineering Mandate:** When debugging model behavior or simulation logic, assume all existing code, vectorization contracts, and mathematical equations are flawed until proven otherwise. Cross-reference the exact shapes, indices, and padding logic between Training (data loaders) and Inference (bridges/contracts). A model is only as smart as the exact tensor footprint it receives; any misalignment between the two phases is fatal. 

* **Versioning:** read .agents/skills/OFK/references/versioned-architecture-guardrails.md first. 

* **When starting a new training session* remember to read and start \.agents\skills\monitor-training-session

* **Training Telemetry Output:** All training scripts, regardless of model version, MUST output or tee their telemetry and logs to `active_training.log` in the repository root directory. The dashboard specifically parses this file, so redirecting `sys.stdout` to this file is mandatory for all training scripts.
