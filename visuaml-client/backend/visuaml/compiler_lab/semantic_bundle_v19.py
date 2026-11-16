"""Closed-world, independently reloadable semantic decoder bundles.

No arbitrary Python modules, pickle, eval, network retrieval, or unsandboxed
model import. An artifact is an exact canonical SemanticProgram, a bounded
v14 checkpoint archive and a source-bound SHA-256 manifest. Digests establish
integrity against modification, NOT signatures/authenticated provenance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .semantic_decoder_v18 import (
    CompiledDecoderPlan,
    lower_decoder_semantic_program,
    make_decoder_semantic_program,
)
from .stacked_decoder_v14 import (
    checkpoint_digest,
    load_canonical_checkpoint,
    write_canonical_checkpoint,
)

SCHEMA = "visuaml.semantic-decoder-bundle.v19"
_MAX_JSON_BYTES = 1024 * 1024
_FILES = ("semantic.json", "checkpoint.json", "weights.npz")


class BundleV19Error(ValueError):
    """Invalid or tampered with closed-world decoder artifact."""


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _encode(document: Any) -> bytes:
    return (json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise BundleV19Error("Duplicate JSON field")
        result[name] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_JSON_BYTES:
        raise BundleV19Error("Missing, linked, or oversized JSON artifact")
    try:
        document = json.loads(path.read_bytes(), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise BundleV19Error("Invalid JSON artifact") from exc
    if not isinstance(document, dict):
        raise BundleV19Error("Expected JSON object")
    return document


def write_decoder_bundle(
    plan: CompiledDecoderPlan, directory: str | Path, *, provenance: str
) -> Path:
    """Write canonical graph + numerical checkpoint, then their integrity manifest."""
    compiled = lower_decoder_semantic_program(plan.semantic, plan.weights, batch=plan.batch)
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    if any((root / name).exists() or (root / name).is_symlink() for name in (*_FILES, "bundle.json")):
        raise BundleV19Error("Bundle destination must not contain existing artifact files")
    write_canonical_checkpoint(compiled.weights, root, provenance=provenance)
    (root / "semantic.json").write_bytes(_encode(compiled.semantic.as_dict()))
    members = {name: _sha((root / name).read_bytes()) for name in _FILES}
    manifest = {
        "schema": SCHEMA,
        "batch": compiled.batch,
        "semantic_sha256": compiled.digest,
        "checkpoint_sha256": checkpoint_digest(compiled.weights),
        "files_sha256": members,
        "backend": "NumPy CPU interpreted canonical decoder program",
    }
    (root / "bundle.json").write_bytes(_encode({
        **manifest, "manifest_sha256": _sha(_encode(manifest))
    }))
    return root


def load_decoder_bundle(directory: str | Path) -> CompiledDecoderPlan:
    """Verify everything before admitting the reconstructed canonical program."""
    root = Path(directory)
    try:
        document = _read_json(root / "bundle.json")
        expected = {
            "schema", "batch", "semantic_sha256", "checkpoint_sha256",
            "files_sha256", "backend", "manifest_sha256",
        }
        if set(document) != expected or document["schema"] != SCHEMA:
            raise BundleV19Error("Unknown or incomplete bundle schema")
        unsigned = {k: v for k, v in document.items() if k != "manifest_sha256"}
        if document["manifest_sha256"] != _sha(_encode(unsigned)):
            raise BundleV19Error("Bundle manifest digest mismatch")
        sources = document["files_sha256"]
        if not isinstance(sources, dict) or set(sources) != set(_FILES):
            raise BundleV19Error("Unexpected bundle members")
        for name in _FILES:
            target = root / name
            if target.is_symlink() or not target.is_file():
                raise BundleV19Error("Missing or linked artifact member")
            if target.stat().st_size > (16 * 1024 * 1024 if name == "weights.npz" else _MAX_JSON_BYTES):
                raise BundleV19Error("Bundle member exceeds byte budget")
            if sources[name] != _sha(target.read_bytes()):
                raise BundleV19Error("Bundle member SHA-256 mismatch")
        serialized = _read_json(root / "semantic.json")
        weights = load_canonical_checkpoint(root)
        batch = document["batch"]
        if type(batch) is not int or not 1 <= batch <= 8:
            raise BundleV19Error("Invalid bundle batch contract")
        compiled = lower_decoder_semantic_program(
            make_decoder_semantic_program(weights, batch=batch), weights, batch=batch
        )
        if (
            compiled.digest != document["semantic_sha256"]
            or compiled.checkpoint_sha256 != document["checkpoint_sha256"]
            or _encode(compiled.semantic.as_dict()) != _encode(serialized)
        ):
            raise BundleV19Error("Serialized semantic graph disagrees with checkpoint or compiler")
        return compiled
    except (OSError, KeyError, TypeError, AttributeError, OverflowError, RuntimeError) as exc:
        raise BundleV19Error("Decoder bundle could not be validated") from exc
