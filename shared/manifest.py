"""Version-agnostic manifest + self-describing checkpoint I/O.

This module knows NOTHING about any specific model version. A version declares a
`VersionManifest` (in `versions/<id>/core/manifest.py`); the runtime loads a version
only through it. Checkpoints are saved with embedded metadata and loaded fail-loud, so
a contract/architecture mismatch (e.g. the old 159-vs-163 dim bug) raises instead of
silently loading garbage.

See: .agents/skills/OFK/references/versioned-architecture-guardrails.md
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Tuple

import torch


@dataclass(frozen=True)
class VersionManifest:
    version_id: str                       # e.g. "v12"
    context_dim: int                      # feature-vector width; MUST match model state_proj input
    contract_version: int                 # bump whenever the tensor schema changes
    action_space: Tuple[str, ...]         # e.g. ("fold", "call", "raise")
    model_class: str                      # "versions.v12.core.model:PokerEVModelV4"
    contract_class: str                   # "versions.v12.core.contract:ContractV8V9"
    weights_dir: str                      # "versions/v12/weights"
    status: str = "active"                # active | frozen | deprecated


def _robust_save(obj, path: str, retries: int = 6, delay: float = 0.4) -> bool:
    """Atomic + retrying save (Windows file-lock safe): temp file then os.replace."""
    tmp = f"{path}.tmp.{os.getpid()}"
    last_err = None
    for attempt in range(retries):
        try:
            torch.save(obj, tmp)
            os.replace(tmp, path)
            return True
        except Exception as e:  # transient FS/lock hiccup
            last_err = e
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
            time.sleep(delay * (attempt + 1))
    print(f"WARNING: could not write {path} after {retries} attempts: {last_err}")
    return False


def save_checkpoint(state_dict, path: str, manifest: VersionManifest,
                    hands_trained: int = 0, **extra) -> bool:
    """Save a self-describing checkpoint (state_dict + version/contract metadata)."""
    payload = {
        "state_dict": state_dict,
        "version_id": manifest.version_id,
        "context_dim": manifest.context_dim,
        "contract_version": manifest.contract_version,
        "hands_trained": hands_trained,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    payload.update(extra)
    return _robust_save(payload, path)


def load_state_dict(path: str, manifest: VersionManifest, map_location="cpu"):
    """Load a checkpoint FAIL-LOUD: raise on a missing-metadata or contract mismatch.

    Returns the bare state_dict. Never silently falls back to random weights.
    """
    ckpt = torch.load(path, map_location=map_location)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(
            f"{path} is not a self-describing checkpoint (no 'state_dict'/metadata). "
            f"Refusing to load into {manifest.version_id}. Re-save it via save_checkpoint()."
        )
    ck_dim = ckpt.get("context_dim")
    if ck_dim != manifest.context_dim:
        raise ValueError(
            f"Checkpoint {path} is context_dim={ck_dim} (contract v{ckpt.get('contract_version')}), "
            f"but {manifest.version_id} expects context_dim={manifest.context_dim}. Refusing to load."
        )
    return ckpt["state_dict"]
