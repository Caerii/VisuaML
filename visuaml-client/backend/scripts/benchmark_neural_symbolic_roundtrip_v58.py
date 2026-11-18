"""Offline v58 blinded neural-symbolic round-trip benchmark.

This is a narrow, reproducible bridge between the original v49/v50 research
and the v58 typed microstep machine. It never contacts GitHub or runs CI.

The synthesizer sees oracle I/O only. Source programs are exclusively owned
by the wrapper oracle and independent Z3 verifier; proof artifacts certify
typed source-vs-candidate i32 equivalence, NOT arbitrary neural weights.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path

import numpy as np

from visuaml.compiler_lab.neural_symbolic_roundtrip_v58 import (
    SCHEMA,
    source_workloads,
    verify_roundtrip,
)


def stable(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()


def git_identity(root: Path) -> dict[str, str]:
    def git(*args: str) -> str:
        return subprocess.check_output(
            ("git", "-C", str(root), *args),
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    if git("status", "--porcelain", "--untracked-files=normal"):
        raise RuntimeError("Benchmark evidence requires a clean Git worktree")
    return {
        "commit": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "branch": git("branch", "--show-current"),
    }


def produce(root: Path, *, holdout: bool, timeout_ms: int) -> dict:
    import z3  # noqa: F401: require Z3 before producing benchmark artifacts
    import wasmtime  # noqa: F401: require independent Wasmtime

    identity = git_identity(root)
    cases = []
    for case in source_workloads():
        if case.split == "heldout" and not holdout:
            continue
        cases.append(verify_roundtrip(case, timeout_ms=timeout_ms))
    source_hashes = {entry["original_wasm_sha256"] for entry in cases}
    if len(source_hashes) != len(cases):
        raise AssertionError("Round-trip benchmark duplicated source programs")
    if not all(entry["all_i32_typed_source_equivalence"] for entry in cases):
        raise AssertionError("SMT failure cannot be recorded as successful")
    if not all(entry["finite_results"] for entry in cases):
        raise AssertionError("Missing independent execution replay")
    return {
        "schema": SCHEMA,
        "producer": "offline-source-bound-v58-roundtrip",
        "source": identity,
        "versions": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "numpy": np.__version__,
            "wasmtime": importlib.metadata.version("wasmtime"),
            "z3_solver": importlib.metadata.version("z3-solver"),
        },
        "policy": {
            "holdout_enabled": holdout,
            "solver_timeout_ms_per_candidate": timeout_ms,
            "synthesis_sees_source": False,
            "synthesis_sees_weights": False,
            "smts_prove_typed_source_vs_symbolic_candidate": True,
            "prove_all_inputs_through_constructed_neural_weights": False,
            "arbitrary_learned_weight_decompilation": False,
            "historical_source_uniqueness": False,
            "must_check_all_finite_v58_traces": True,
            "github_actions_used": False,
            "source_and_candidate_wasm_validated_by_wasmtime": True,
        },
        "inventory": {
            "source_programs": len(cases),
            "historical_programs": sum(
                case["split"] == "historical" for case in cases
            ),
            "heldout_programs": sum(
                case["split"] == "heldout" for case in cases
            ),
            "finite_wasm_replays": sum(
                case["finite_replay_inputs"] for case in cases
            ),
            "recovered_v58_instruction_transitions": sum(
                case["finite_recovered_instruction_transitions"]
                for case in cases
            ),
            "cegis_total_refinements": sum(
                case["cegis_refinements"] for case in cases
            ),
        },
        "results": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holdout", action="store_true",
                        help="Add eight source programs held out from v50")
    parser.add_argument("--timeout-ms", type=int, default=15000)
    args = parser.parse_args()
    if not 1 <= args.timeout_ms <= 60000:
        parser.error("Z3 timeout must be between 1 and 60000ms")
    target = args.output.resolve()
    if target.exists():
        parser.error(f"Refusing to overwrite original evidence: {target}")
    root = Path(__file__).resolve().parents[3]
    # Reproducibility is checked before any local file is written.
    report = produce(
        root, holdout=args.holdout, timeout_ms=args.timeout_ms,
    )
    if target == root or root in target.parents:
        parser.error("Evidence output must be outside the Git checkout")
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = stable(report) + b"\n"
    with target.open("xb") as handle:
        handle.write(raw)
    print(json.dumps({
        "artifact": str(target),
        "artifact_sha256": hashlib.sha256(raw).hexdigest(),
        "source_commit": report["source"]["commit"],
        "inventory": report["inventory"],
        "all_i32_proof_scope": "typed source vs accepted symbolic candidate",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
