"""Offline proof-gated representation-cost benchmark (original vs recovered).

This evaluates two symbolic representations of *the same typed i32 function*
after independently replaying a valid v50->v58 source-bound SMT receipt.
It does NOT specialize the neural VM's weights, and it does not imply
hardware acceleration, mechanism identification or a universal compiler.

Costs are paired within one engine and source/input, not pooled across
different engines. Compilation/setup is excluded from warm execution;
the reported break-even estimate is a LOWER BOUND ignoring build/compile
and deployment costs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

import wasmtime

from scripts.verify_neural_roundtrip_evidence_v58 import verify
from visuaml.compiler_lab.neural_microstep_executor_v58 import (
    execute_microstep_control,
)
from visuaml.compiler_lab.neural_symbolic_roundtrip_v58 import as_control
from visuaml.compiler_lab.wasm_state_vm_v40 import decode_module

SCHEMA = "visuaml.proof-gated-representation-cost.v58"
TIMING_INPUTS = (-17, 0, 17)


def stable_bytes(data: object) -> bytes:
    return json.dumps(
        data, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _measure(call, expect: int) -> int:
    start = time.perf_counter_ns()
    actual = call()
    elapsed = time.perf_counter_ns() - start
    actual = (
        actual if type(actual) is int
        else getattr(actual, "result_i32", None)
    )
    if actual != expect:
        raise AssertionError("A timed representation returned a wrong result")
    return elapsed


def compare_pair(
    original, recovered, expected: int, *,
    warmups: int = 2, repeats: int = 7,
) -> dict:
    if not 0 <= warmups <= 1000 or not 1 <= repeats <= 10000:
        raise ValueError("Invalid representation timing policy")
    for _ in range(warmups):
        if original() != expected or recovered() != expected:
            raise AssertionError("Representation warmup was not semantically valid")
    left = []
    right = []
    for i in range(repeats):
        # Alternating AB/BA pairing reduces—but cannot eliminate—
        # clock drift, cache and scheduler order effects.
        if i % 2 == 0:
            left.append(_measure(original, expected))
            right.append(_measure(recovered, expected))
        else:
            right.append(_measure(recovered, expected))
            left.append(_measure(original, expected))
    a = statistics.median(left)
    b = statistics.median(right)
    return {
        "samples_per_representation": repeats,
        "original_median_ns": a,
        "recovered_median_ns": b,
        "original_sample_ns": left,
        "recovered_sample_ns": right,
        "speedup_original_over_recovered": a / b if b > 0 else None,
        "warmup_excluded": True,
        "compilation_excluded": True,
    }


def run(path: Path, root: Path, *, repeats: int, warmups: int) -> dict:
    # This replays Z3 proofs and all original/recovered finite execution
    # traces FIRST. Failed receipts never enter the timing study.
    provenance = verify(path, root)
    archive = json.loads(path.read_bytes())
    engine = wasmtime.Engine()
    cases = []
    for entry in archive["results"]:
        original_bytes = bytes.fromhex(entry["original_wasm_hex"])
        recovered_bytes = bytes.fromhex(entry["recovered_wasm_hex"])
        original_program = as_control(decode_module(original_bytes).program)
        recovered_program = as_control(decode_module(recovered_bytes).program)
        original_module = wasmtime.Module(engine, original_bytes)
        recovered_module = wasmtime.Module(engine, recovered_bytes)
        store = wasmtime.Store(engine)
        original_wasm = wasmtime.Instance(
            store, original_module, []
        ).exports(store)["run"]
        recovered_wasm = wasmtime.Instance(
            store, recovered_module, []
        ).exports(store)["run"]
        results = []
        for value in TIMING_INPUTS:
            expected = next(
                row["result_i32"] for row in entry["finite_results"]
                if row["input_i32"] == value
            )
            for engine_name, first, second in (
                (
                    "wasmtime",
                    lambda v=value: original_wasm(store, v),
                    lambda v=value: recovered_wasm(store, v),
                ),
                (
                    "v58_microstep",
                    lambda v=value: execute_microstep_control(
                        original_program, v, fuel=400,
                    ).result_i32,
                    lambda v=value: execute_microstep_control(
                        recovered_program, v, fuel=400,
                    ).result_i32,
                ),
            ):
                measured = compare_pair(
                    first, second, expected,
                    repeats=repeats, warmups=warmups,
                )
                delta = (
                    measured["original_median_ns"]
                    - measured["recovered_median_ns"]
                )
                # Lower bound: original recovery wall duration already
                # includes CEGIS and SMT but omits both compilation costs.
                break_even = (
                    math.ceil(
                        entry["cegis_recovery_and_smt_wall_ns"] / delta
                    ) if delta > 0 else None
                )
                results.append({
                    "engine": engine_name,
                    "input_i32": value,
                    "lower_bound_amortization_calls_if_faster": break_even,
                    **measured,
                })
        cases.append({
            "name": entry["name"],
            "split": entry["split"],
            "original_wasm_sha256": entry["original_wasm_sha256"],
            "recovered_wasm_sha256": entry["recovered_wasm_sha256"],
            "original_wasm_bytes": entry["original_wasm_bytes"],
            "recovered_wasm_bytes": entry["recovered_wasm_bytes"],
            "recovery_and_smt_wall_ns": entry[
                "cegis_recovery_and_smt_wall_ns"
            ],
            "results": results,
        })
    return {
        "schema": SCHEMA,
        "proof_archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_commit": provenance["source_commit"],
        "source_tree": provenance["source_tree"],
        "proof_replayed_before_timing": True,
        "neural_weights_specialized": False,
        "only_symbolic_source_representation_changed": True,
        "policy": {
            "timed_inputs": list(TIMING_INPUTS),
            "repeats": repeats,
            "warmups": warmups,
            "within_engine_paired_timing": True,
            "AB_BA_order_alternation": True,
            "setup_compilation_excluded": True,
            "break_even_omits_compile_and_deployment": True,
            "neural_hardware_benchmark": False,
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error("Refusing to overwrite a prior representation-cost run")
    root = Path(__file__).resolve().parents[3]
    data = run(
        args.input, root,
        repeats=args.repeats, warmups=args.warmups,
    )
    if output == root or root in output.parents:
        parser.error("Evidence output must be outside checkout")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as file:
        file.write(stable_bytes(data) + b"\n")
    print(json.dumps({
        "artifact": str(output),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "programs": len(data["cases"]),
        "pairs": sum(len(case["results"]) for case in data["cases"]),
        "scope": "original and recovered symbolic WASM with unchanged interpreter weights",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
