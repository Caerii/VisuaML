"""Closed-world v20 real Llama semantic bundle with exact-file integrity checks.

No untrusted Python loading, pickle, network access, or external code execution.
Bundle SHA-256 gives bytes identity, NOT authenticated publisher identity.
Only the v19's bounded, safe Llama checkpoint and a compiler-canonical
SemanticProgram may be reconstructed from a bundle.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .hf_llama_checkpoint_v19 import load_hf_llama_local
from .llama_semantic_v20 import (
    LlamaSemanticPlan,
    lower_llama_semantic_program,
    make_llama_semantic_program,
)

SCHEMA = "visuaml.llama-semantic-bundle.v20"
FILES = ("config.json", "model.safetensors", "semantic.json")
_MAX_JSON = 2 * 1024 * 1024
_MAX_WEIGHTS = 128 * 1024 * 1024


class LlamaBundleError(ValueError):
    """Malformed, tampered, or inadmissible Llama semantic bundle."""


def _bytes(document: Any) -> bytes:
    return (
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read(path: Path, maximum: int) -> bytes:
    if (
        path.is_symlink() or not path.is_file()
        or not 0 < path.stat().st_size <= maximum
    ):
        raise LlamaBundleError("Unsafe, missing, or oversized bundle file")
    return path.read_bytes()


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict:
    seen = {}
    for name, value in pairs:
        if name in seen:
            raise LlamaBundleError("Duplicate JSON key")
        seen[name] = value
    return seen


def _parse(raw: bytes) -> dict:
    try:
        result = json.loads(raw, object_pairs_hook=_unique_keys)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise LlamaBundleError("Invalid JSON document") from exc
    if not isinstance(result, dict):
        raise LlamaBundleError("Bundle JSON must be an object")
    return result


def _source_valid(label: str) -> bool:
    return (
        isinstance(label, str)
        and bool(re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}/[A-Za-z0-9][A-Za-z0-9_.-]{0,99}",
            label,
        ))
    )


def write_llama_bundle(
    plan: LlamaSemanticPlan, source: str | Path, destination: str | Path,
    *, declared_source: str,
) -> Path:
    """Publish a validated, immutable-input checkpoint and its canonical IR."""
    if not _source_valid(declared_source):
        raise LlamaBundleError("Specify a bounded user-declared source repo")
    canonical = lower_llama_semantic_program(
        plan.program, plan.weights, batch=plan.batch
    )
    src = Path(source)
    dest = Path(destination)
    if dest.is_symlink():
        raise LlamaBundleError("Destination cannot be a symlink")
    dest.mkdir(parents=True, exist_ok=True)
    if any((dest / name).exists() or (dest / name).is_symlink()
           for name in (*FILES, "bundle.json")):
        raise LlamaBundleError("Bundle destination must be empty of artifact files")
    for filename, maximum in (
        ("config.json", _MAX_JSON),
        ("model.safetensors", _MAX_WEIGHTS),
    ):
        raw = _read(src / filename, maximum)
        expected = (
            canonical.weights.config_sha256 if filename == "config.json"
            else canonical.weights.checkpoint_sha256
        )
        if _sha(raw) != expected:
            raise LlamaBundleError("Source bytes differ from compiler checkpoint")
        (dest / filename).write_bytes(raw)
    (dest / "semantic.json").write_bytes(_bytes(canonical.program.as_dict()))
    hashes = {
        name: _sha(_read(dest / name, _MAX_WEIGHTS if name.endswith(".safetensors") else _MAX_JSON))
        for name in FILES
    }
    manifest = {
        "schema": SCHEMA,
        "declared_source": declared_source,
        "declared_revision": canonical.weights.declared_revision,
        "batch": canonical.batch,
        "checkpoint_sha256": canonical.weights.checkpoint_sha256,
        "config_sha256": canonical.weights.config_sha256,
        "semantic_sha256": canonical.semantic_sha256,
        "file_sha256": hashes,
        "claim": "bounded, non-authenticated, checkpoint-bound real Llama semantic graph",
    }
    (dest / "bundle.json").write_bytes(_bytes({
        **manifest, "manifest_sha256": _sha(_bytes(manifest)),
    }))
    return dest


def load_llama_bundle(directory: str | Path) -> LlamaSemanticPlan:
    """Verify stored bytes, regenerate the program, then lower it to an executor."""
    root = Path(directory)
    try:
        manifest_raw = _read(root / "bundle.json", _MAX_JSON)
        metadata = _parse(manifest_raw)
        if _bytes(metadata) != manifest_raw:
            raise LlamaBundleError("Bundle manifest must use canonical JSON bytes")
        required = {
            "schema", "declared_source", "declared_revision", "batch",
            "checkpoint_sha256", "config_sha256", "semantic_sha256",
            "file_sha256", "claim", "manifest_sha256",
        }
        if set(metadata) != required or metadata["schema"] != SCHEMA:
            raise LlamaBundleError("Unknown bundle schema or fields")
        if not _source_valid(metadata["declared_source"]):
            raise LlamaBundleError("Invalid declared model source")
        unsigned = {key: val for key, val in metadata.items() if key != "manifest_sha256"}
        if metadata["manifest_sha256"] != _sha(_bytes(unsigned)):
            raise LlamaBundleError("Manifest checksum mismatch")
        hashes = metadata["file_sha256"]
        if not isinstance(hashes, dict) or set(hashes) != set(FILES):
            raise LlamaBundleError("Unexpected bundle member set")
        for name in FILES:
            raw = _read(root / name, _MAX_WEIGHTS if name.endswith(".safetensors") else _MAX_JSON)
            if hashes[name] != _sha(raw):
                raise LlamaBundleError("Artifact file content digest mismatch")
        batch = metadata["batch"]
        if type(batch) is not int or not 1 <= batch <= 2:
            raise LlamaBundleError("Unsupported batch contract")
        checkpoint = load_hf_llama_local(
            root,
            declared_revision=metadata["declared_revision"],
            expected_weights_sha256=metadata["checkpoint_sha256"],
        )
        if checkpoint.config_sha256 != metadata["config_sha256"]:
            raise LlamaBundleError("Configuration fingerprint mismatch")
        program = make_llama_semantic_program(checkpoint, batch=batch)
        plan = lower_llama_semantic_program(program, checkpoint, batch=batch)
        if (
            plan.semantic_sha256 != metadata["semantic_sha256"]
            or _bytes(plan.program.as_dict()) != _bytes(
                _parse(_read(root / "semantic.json", _MAX_JSON))
            )
        ):
            raise LlamaBundleError("Stored semantic graph is not compiler-canonical")
        return plan
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError) as exc:
        if isinstance(exc, LlamaBundleError):
            raise
        raise LlamaBundleError("Could not load admitted Llama bundle") from exc
