"""Integrity-check source-bound v32 native codegen evidence JSON.

This detects accidental/tampered source, binary/receipt mismatches and
inconsistent fixture claims; SHA-256 alone is NOT an author signature,
independent re-execution, or proof of all-input semantic equivalence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


class EvidenceError(ValueError):
    """Invalid or incompatible research evidence."""


def _digest(value: str) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def _source_receipt(record: dict, root: Path) -> None:
    entries = record.get("source_sha256")
    if not isinstance(entries, dict) or not entries:
        raise EvidenceError("Missing source manifest")
    for name, expected in entries.items():
        if (
            not isinstance(name, str)
            or not name.startswith(("examples/", "visuaml/"))
            or ".." in Path(name).parts
            or not _digest(expected)
        ):
            raise EvidenceError("Invalid source path or digest")
        file = root / name
        if not file.is_file():
            raise EvidenceError("Source file absent from checkout")
        observed = hashlib.sha256(file.read_bytes()).hexdigest()
        if observed != expected:
            raise EvidenceError("Source hash disagrees with checkout")


def _fixture(fixture: dict, *, warm: bool) -> None:
    compiler = fixture.get("compiler")
    if not isinstance(compiler, dict):
        raise EvidenceError("Missing generated compiler receipt")
    source = fixture.get("generated_cpp")
    if not isinstance(source, str) or len(source) > 100_000:
        raise EvidenceError("Generated C++ source missing or over budget")
    if hashlib.sha256(source.encode()).hexdigest() != compiler.get(
        "generated_cpp_sha256"
    ):
        raise EvidenceError("Generated C++ content hash mismatch")
    for key in (
        "semantic_sha256", "source_program_sha256",
        "generated_cpp_sha256", "native_binary_sha256", "output_sha256",
    ):
        if not _digest(compiler.get(key)):
            raise EvidenceError(f"Invalid {key}")
    if (
        compiler.get("schema") != "visuaml.semantic-native-cpu.v32"
        or not compiler.get("op_sequence")
        or not compiler.get("compiler_identity")
        or compiler.get("retained_bytes_upper_bound", 0) > 8 * 1024 * 1024
        or "-fno-fast-math" not in compiler.get("compiler_flags", [])
    ):
        raise EvidenceError("Unadmitted native compiler receipt")
    for key in ("max_abs_error_numpy", "max_abs_error_torch"):
        value = fixture.get(key)
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or value < 0 or value > 2e-5
        ):
            raise EvidenceError("Numerical witness exceeded declared tolerance")
    hashes = compiler.get("input_sha256")
    if not isinstance(hashes, dict) or not hashes or any(
        not _digest(value) for value in hashes.values()
    ):
        raise EvidenceError("Input tensor fingerprints missing")
    if warm:
        timings = fixture.get("warm_single_run_ms")
        if not isinstance(timings, dict):
            raise EvidenceError("Warm execution samples missing")
        raw = timings.get("raw_samples")
        if not isinstance(raw, list) or len(raw) < 2 or any(
            type(x) not in (int, float) or not math.isfinite(x) or x <= 0
            for x in raw
        ):
            raise EvidenceError("Invalid warm execution samples")
    else:
        for name, expected in fixture.get("weight_sha256", {}).items():
            if hashes.get(name) != expected:
                raise EvidenceError("Bound FX weight digest disagrees with execution")


def verify(declared: dict, fx: dict, *, expected_commit: str, root: Path) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", expected_commit):
        raise EvidenceError("Expected a canonical Git commit SHA")
    if (
        declared.get("schema") != "visuaml.semantic-native-fixtures.v32"
        or fx.get("schema") != "visuaml.trusted-fx-generated-native.v32"
        or declared.get("source_commit") != expected_commit
        or fx.get("source_commit") != expected_commit
    ):
        raise EvidenceError("Evidence schema/source revision mismatch")
    _source_receipt(declared, root)
    _source_receipt(fx, root)
    declared_cases = {x.get("family"): x for x in declared.get("fixtures", [])}
    fx_cases = {x.get("family"): x for x in fx.get("fixtures", [])}
    if set(declared_cases) != {"residual-mlp", "convolutional-residual"} or set(
        fx_cases
    ) != {"trusted_fx_mlp", "trusted_fx_conv"}:
        raise EvidenceError("Expected both distinct graph families and FX imports")
    for case in declared_cases.values():
        _fixture(case, warm=True)
    for case in fx_cases.values():
        _fixture(case, warm=False)
    for name, imported in (
        ("residual-mlp", "trusted_fx_mlp"),
        ("convolutional-residual", "trusted_fx_conv"),
    ):
        left = declared_cases[name]["compiler"]
        right = fx_cases[imported]["compiler"]
        if (
            left["generated_cpp_sha256"] != right["generated_cpp_sha256"]
            or left["native_binary_sha256"] != right["native_binary_sha256"]
        ):
            raise EvidenceError(
                "Graph-generated binary unexpectedly depends on model name/FX origin"
            )
    return {
        "schema": "visuaml.native-semantic-evidence-integrity.v32",
        "source_commit": expected_commit,
        "declared_families": sorted(declared_cases),
        "trusted_fx_families": sorted(fx_cases),
        "source_and_cpp_hashes_verified": True,
        "numerical_witness_tolerance": 2e-5,
        "limitations": (
            "Hash consistency and finite witness checks only; not a signature, "
            "full independent re-execution, proof, or pretrained certification."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declared", type=Path, required=True)
    parser.add_argument("--fx", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    summary = verify(
        json.loads(args.declared.read_text(encoding="utf-8")),
        json.loads(args.fx.read_text(encoding="utf-8")),
        expected_commit=args.source_commit, root=root,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
