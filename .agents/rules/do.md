---
trigger: always_on
---

#At the end of a task or chat session, the agent should evaluate if a new memory should be written or updated using ofk-memory-manager skill.

#When creating new scripts / debugging files - write a small coding comment at the top, so a human can understand the purpose and if its a "single use" script. Put then into a "temp" folder, to make it easier to cleanup at some stage if needed.

#Whenever simulation logic or environment conditions are modified outside of the simulation state, the agent MUST update the source-of-truth document at '.agents\skills\OFK\references\simulation_architecture.md' to reflect the new state.