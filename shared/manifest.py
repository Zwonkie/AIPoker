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
    milestone: bool = False               # True = a kept reference/fallback version (a known-good
                                          # checkpoint to roll back to); never delete its weights


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


def load_state_dict(path: str, manifest: VersionManifest, map_location="cpu",
                    allow_contract_mismatch: bool = False):
    """Load a checkpoint FAIL-LOUD: raise on a missing-metadata or contract mismatch.

    Returns the bare state_dict. Never silently falls back to random weights.

    [V47 P0.4] contract_version is validated as hard as context_dim: two contracts can share a
    width while a slot's MEANING differs (V43 vs V44 are both 54-wide but ctx[35] normalizes by
    nominal vs effective field) -- width-checking alone cannot catch that. Seating a frozen
    cross-contract ancestor as an OPPONENT is the one deliberate exception; those call sites pass
    `allow_contract_mismatch=True` and the mismatch is still printed, never silent.
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
    ck_cv = ckpt.get("contract_version")
    if ck_cv != manifest.contract_version:
        msg = (f"Checkpoint {path} is contract_version={ck_cv} (saved by "
               f"{ckpt.get('version_id')!r}), but {manifest.version_id} expects "
               f"contract_version={manifest.contract_version} -- same width does not mean same "
               f"feature semantics.")
        if not allow_contract_mismatch:
            raise ValueError(
                msg + " Refusing to load. If this is a DELIBERATE cross-contract frozen-opponent "
                      "seating, pass allow_contract_mismatch=True at the call site.")
        print(f"NOTICE: {msg} Loading anyway (allow_contract_mismatch=True -- frozen-opponent seating).")
    return ckpt["state_dict"]
