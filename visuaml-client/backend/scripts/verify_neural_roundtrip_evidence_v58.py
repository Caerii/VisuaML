"""Independently replay and verify a v58 neural-symbolic round-trip receipt.

This verifier consumes an archive rather than trusting its success flags.
It recomputes original/recovered WASM hashes, the candidate canonical binary,
all-i32 typed-source Z3 equivalence, and the recorded finite V58 neural
microstep traces versus independent Wasmtime.

It does not prove all i32 inputs through constructed neural weights, nor
does it replay every historical CEGIS oracle query or identify an
arbitrary learned network. No network or GitHub Actions use.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import wasmtime

from visuaml.compiler_lab.bit_neural_vm_v49 import (
    construct_bit_weights,
    run_bit_vm,
)
from visuaml.compiler_lab.blinded_behavioral_synthesis_v50 import (
    Candidate,
    emit_canonical_wasm,
    verify_against_typed_source,
)
from visuaml.compiler_lab.neural_microstep_executor_v58 import (
    execute_microstep_control,
)
from visuaml.compiler_lab.neural_symbolic_roundtrip_v58 import (
    INPUTS, SCHEMA, as_control,
)
from visuaml.compiler_lab.wasm_control_flow_v51 import (
    execute_control,
)
from visuaml.compiler_lab.wasm_state_vm_v40 import decode_module


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _trace_digest(trace) -> str:
    return _digest(json.dumps(
        trace, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode())


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ("git", "-C", str(root), *args),
        text=True, stderr=subprocess.DEVNULL,
    ).strip()


def verify(path: Path, root: Path, *, timeout_ms: int = 15000) -> dict:
    if not 1 <= timeout_ms <= 60000:
        raise ValueError("Invalid finite SMT timeout")
    archive = json.loads(path.read_bytes())
    if archive.get("schema") != SCHEMA:
        raise ValueError("Unknown neural-symbolic round-trip schema")
    if archive.get("policy", {}).get("prove_all_inputs_through_constructed_neural_weights"):
        raise ValueError("Evidence overclaims constructed neural gate equivalence")
    if archive.get("policy", {}).get("arbitrary_learned_weight_decompilation"):
        raise ValueError("Evidence overclaims learned model decompilation")
    source = archive.get("source", {})
    if (_git(root, "rev-parse", "HEAD") != source.get("commit")
            or _git(root, "rev-parse", "HEAD^{tree}") != source.get("tree")):
        raise ValueError("Evidence was not produced by this exact source tree")
    engine = wasmtime.Engine()
    fixed = construct_bit_weights()
    inventory = archive.get("results")
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("Empty/malformed round-trip archive")
    identifiers = set()
    comparisons = 0
    for item in inventory:
        raw_original = bytes.fromhex(item["original_wasm_hex"])
        raw_recovered = bytes.fromhex(item["recovered_wasm_hex"])
        original_sha = _digest(raw_original)
        recovered_sha = _digest(raw_recovered)
        if (original_sha != item["original_wasm_sha256"]
                or recovered_sha != item["recovered_wasm_sha256"]):
            raise ValueError("Archived original/recovered WASM bytes tampered")
        if original_sha in identifiers:
            raise ValueError("Duplicate source program identity")
        identifiers.add(original_sha)
        source_program = decode_module(raw_original).program
        recovered_program = decode_module(raw_recovered).program
        candidate = Candidate(**item["candidate"])
        expected_program, expected_binary = emit_canonical_wasm(candidate)
        if (expected_binary != raw_recovered
                or expected_program != recovered_program
                or candidate.digest() != item["candidate_sha256"]):
            raise ValueError("Recovered source bytes not bound to SMT hypothesis")
        result = verify_against_typed_source(
            source_program, candidate, timeout_ms=timeout_ms,
        )
        if (result["solver_outcome"] != "unsat"
                or result["smt_formula_sha256"] != item["smt_formula_sha256"]
                or not result["all_i32_typed_source_equivalent"]):
            raise ValueError("Independent replay did not reproduce proof")
        if (not item["all_i32_typed_source_equivalence"]
                or item["all_i32_neural_weights_equivalence"]
                or item["unique_original_source_recovered"]
                or item["arbitrary_learned_neural_model_decompiled"]):
            raise ValueError("Incorrect semantic proof claim in artifact")
        if (item["constructed_neural_weights_sha256"] != fixed.digest()
                or item["finite_replay_inputs"] != len(INPUTS)):
            raise ValueError("Neural fixed-weight identity/input policy changed")
        controls = as_control(recovered_program)
        wasmtime.Module.validate(engine, raw_original)
        wasmtime.Module.validate(engine, raw_recovered)
        if len(item["finite_results"]) != len(INPUTS):
            raise ValueError("Finite replay inventory incomplete")
        observed_inputs = [row["input_i32"] for row in item["finite_results"]]
        if observed_inputs != list(INPUTS):
            raise ValueError("Original finite replay ordering/inputs changed")
        steps = 0
        for row in item["finite_results"]:
            x = row["input_i32"]
            store = wasmtime.Store(engine)
            original = wasmtime.Instance(
                store, wasmtime.Module(engine, raw_original), []
            ).exports(store)["run"](store, x)
            recovered = wasmtime.Instance(
                store, wasmtime.Module(engine, raw_recovered), []
            ).exports(store)["run"](store, x)
            neural = run_bit_vm(
                source_program, x, fixed=fixed,
            )["result"]
            candidate_vm = execute_microstep_control(
                controls, x, fuel=400,
            )
            independent = execute_control(
                controls, x, fuel=400,
            )
            if not (
                original == recovered == neural
                == candidate_vm.result_i32 == independent.value
                == row["result_i32"]
            ):
                raise AssertionError("Stored behavior does not replay")
            if (candidate_vm.steps != row["recovered_v58_steps"]
                    or _trace_digest(candidate_vm.trace) !=
                    row["recovered_v58_trace_sha256"]):
                raise AssertionError("V58 full state trace changed")
            for new, old in zip(
                candidate_vm.trace, independent.trace, strict=True,
            ):
                for key in (
                    "pc", "next_pc", "opcode", "stack_i32", "locals_i32",
                    "control_labels", "halted",
                ):
                    if new[key] != old[key]:
                        raise AssertionError("Source-reference trace mismatch")
            comparisons += 1
            steps += candidate_vm.steps
        if steps != item["finite_recovered_instruction_transitions"]:
            raise AssertionError("Transition inventory was altered")
    expected = archive["inventory"]
    if (expected["source_programs"] != len(inventory)
            or expected["finite_wasm_replays"] != comparisons):
        raise ValueError("Archive inventory counts do not agree with replay")
    return {
        "source_commit": source["commit"],
        "source_tree": source["tree"],
        "artifact_sha256": _digest(path.read_bytes()),
        "original_programs_checked": len(inventory),
        "finite_execution_replays": comparisons,
        "typed_source_vs_candidate_all_i32_z3_rechecked": True,
        "neural_weight_all_i32_proved": False,
        "original_source_unique_recovered": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--timeout-ms", type=int, default=15000)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    result = verify(args.input, root, timeout_ms=args.timeout_ms)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
