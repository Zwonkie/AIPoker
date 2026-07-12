# Windows PyTorch Multiprocessing Deadlock Remediation

**Date Recorded**: 2026-07-12
**Related Files**: [train_selfplay.py](file:///c:/REPO/Antigravity/AIPoker/tools/self_play/v9/train_selfplay.py)

## Context
During the V9 multi-personality league self-play, we encountered an issue where simulation training was hanging with 0% CPU/GPU utilization right after completing the first batch. We isolated this to PyTorch CUDA context initialization when spawning new workers via `multiprocessing.Pool` inside the simulation loop. On Windows, Python's `multiprocessing` must use the `spawn` start method. Creating a new `Pool` in every loop iteration forced all 8 workers to repeatedly re-initialize CUDA memory for evaluations, leading to memory exhaustion and deadlocks.

## Resolution / Guidelines
- **Pool Scope**: Do not create or destroy `multiprocessing.Pool` inside recurrent training loops if the spawned workers must allocate GPU contexts.
- **Worker Reuse**: The `Pool` must be created exactly once before the `while` loop starts. Use `pool.starmap` inside the loop, allowing the same set of alive worker processes to take the new parameters and repeatedly process batches without re-spawning.
- **Graceful Cleanup**: Always use a `try...finally` block to guarantee `pool.close()` and `pool.join()` are called when the parent process exits.
- **Dynamic File Reading**: Instead of sending massive model weights over IPC, we have workers load active models dynamically from a temporary `.pth` checkpoint generated at the top of every parent loop. This synergizes perfectly with persistent worker processes without requiring shared memory hacking.
