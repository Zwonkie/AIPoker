"""Version-agnostic registry: the single dispatch point from a version_id to its slice.

Discovers `versions/<id>/core/manifest.py:MANIFEST` and can instantiate that version's
model + contract and load its weights fail-loud. Nothing here branches on a version
number; adding a version = adding a manifest, and touches zero code in this file.

See: .agents/skills/OFK/references/versioned-architecture-guardrails.md
"""
from __future__ import annotations

import importlib
import os
from typing import Dict

from shared.manifest import VersionManifest, load_state_dict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VERSIONS_DIR = os.path.join(_REPO_ROOT, "versions")


def discover_manifests() -> Dict[str, VersionManifest]:
    """Import every versions/<id>/core/manifest.py and collect its MANIFEST."""
    manifests: Dict[str, VersionManifest] = {}
    if not os.path.isdir(_VERSIONS_DIR):
        return manifests
    for entry in sorted(os.listdir(_VERSIONS_DIR)):
        mod_path = f"versions.{entry}.core.manifest"
        try:
            mod = importlib.import_module(mod_path)
        except ModuleNotFoundError:
            continue
        manifest = getattr(mod, "MANIFEST", None)
        if isinstance(manifest, VersionManifest):
            manifests[manifest.version_id] = manifest
    return manifests


def get_manifest(version_id: str) -> VersionManifest:
    manifests = discover_manifests()
    if version_id not in manifests:
        raise KeyError(f"No manifest for version '{version_id}'. Known: {sorted(manifests)}")
    return manifests[version_id]


def _import_attr(dotted: str):
    """Resolve 'pkg.mod:Attr' -> the attribute object."""
    module_path, attr = dotted.split(":")
    return getattr(importlib.import_module(module_path), attr)


def build_model(version_id: str):
    """Instantiate the version's model class (uninitialized weights)."""
    return _import_attr(get_manifest(version_id).model_class)()


def build_contract(version_id: str):
    """Instantiate the version's data contract."""
    return _import_attr(get_manifest(version_id).contract_class)()


def load_model(version_id: str, weights_filename: str, device: str = "cpu",
               allow_contract_mismatch: bool = False):
    """Build the version's model and load a self-describing checkpoint fail-loud.

    `allow_contract_mismatch` is ONLY for deliberately seating a frozen cross-contract ancestor
    as an opponent (see shared/manifest.py::load_state_dict [V47 P0.4])."""
    manifest = get_manifest(version_id)
    model = _import_attr(manifest.model_class)().to(device)
    weights_path = os.path.join(_REPO_ROOT, manifest.weights_dir, weights_filename)
    model.load_state_dict(load_state_dict(weights_path, manifest, map_location=device,
                                          allow_contract_mismatch=allow_contract_mismatch))
    model.eval()
    return model
