"""v51.1 real WASM Euclid GCD control-flow replay and bounded-fuel evidence.

Independent Wasmtime validates/executes the original canonical source bytes.
The constructed v49 bit-gated i32 arithmetic is used inside a host-routed
v51 loop/if/branch machine. A separate mathematical oracle supplies expected
gcd results. Negative inputs intentionally return sentinel zero.

Large positive inputs may be mathematically terminating but exceed declared
fuel: this is a resource exhaustion observation, NOT proof of nontermination.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import wasmtime

from visuaml.compiler_lab.wasm_control_flow_v51 import (
    ControlFuelExhausted,
    ControlOperation,
    ControlProgram,
    decode_control,
    encode_control,
    execute_control,
)

SCHEMA = "visuaml.real-wasm-gcd-control-evidence.v51.1"


def _op(name: str, imm: int | None = None) -> ControlOperation:
    return ControlOperation(name, imm)


def gcd_program() -> ControlProgram:
    o = _op
    return ControlProgram(1, (
        o("local.get", 0), o("i32.eqz"),
        o("if"), o("i32.const", 12), o("return"), o("end"),
        o("local.get", 0), o("i32.const", 0), o("i32.lt_s"),
        o("if"), o("i32.const", 0), o("return"), o("end"),
        o("i32.const", 12), o("local.set", 1),
        o("block"), o("loop"),
        o("local.get", 0), o("local.get", 1),
        o("i32.sub"), o("i32.eqz"), o("br_if", 1),
        o("local.get", 0), o("local.get", 1), o("i32.gt_s"),
        o("if"),
        o("local.get", 0), o("local.get", 1),
        o("i32.sub"), o("local.set", 0),
        o("else"),
        o("local.get", 1), o("local.get", 0),
        o("i32.sub"), o("local.set", 1),
        o("end"), o("br", 0),
        o("end"), o("end"),
        o("local.get", 0), o("end"),
    ))


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def run() -> dict:
    program = gcd_program()
    raw = encode_control(program)
    if decode_control(raw) != program:
        raise AssertionError("GCD program canonical WASM decode failed")
    engine = wasmtime.Engine()
    wasmtime.Module.validate(engine, raw)
    module = wasmtime.Module(engine, raw)
    store = wasmtime.Store(engine)
    function = wasmtime.Instance(store, module, []).exports(store)["run"]
    samples = (-123, -1, *range(25))
    witnesses = []
    selected_traces = []
    for x in samples:
        vm = execute_control(program, x, fuel=1500)
        actual_wasm = function(store, x)
        mathematical = math.gcd(x, 12) if x >= 0 else 0
        if vm.value != actual_wasm or vm.value != mathematical:
            raise AssertionError(f"GCD reference divergence on {x}")
        witnesses.append({
            "input_i32": x,
            "gcd_reference_i32": mathematical,
            "wasmtime_i32": actual_wasm,
            "host_control_bit_gates_i32": vm.value,
            "instruction_transitions": vm.transitions,
            "branch_back_edges": sum(
                step["next_pc"] <= step["pc"]
                for step in vm.trace
                if step["opcode"] == "br"
            ),
            "passed": True,
        })
        if x in (0, 1, 12, 17):
            selected_traces.append({
                "input_i32": x,
                "complete_machine_trace": list(vm.trace),
            })
    # A mathematically terminating value, but too many subtraction steps
    # under this fixed budget. Never call this nonterminating WASM behavior.
    too_large = (1 << 31) - 1
    try:
        execute_control(program, too_large, fuel=60)
    except ControlFuelExhausted:
        resource_exhaustion = True
    else:
        resource_exhaustion = False
    if not resource_exhaustion:
        raise AssertionError("Explicit GCD resource budget did not stop execution")
    source_dir = Path(__file__).resolve().parents[1]
    files = (
        "scripts/probe_wasm_gcd_v51_1.py",
        "visuaml/compiler_lab/wasm_control_flow_v51.py",
        "visuaml/compiler_lab/bit_neural_vm_v49.py",
    )
    return {
        "schema": SCHEMA,
        "source_commit": os.environ.get("VISUAML_SOURCE_COMMIT"),
        "source_file_sha256": {
            name: _sha((source_dir/name).read_bytes()) for name in files
        },
        "gcd_12_source_wasm_sha256": _sha(raw),
        "gcd_12_source_wasm_hex": raw.hex(),
        "program_sha256": program.digest(),
        "original_binary_wasmtime_validated": True,
        "independent_mathematical_reference_used": True,
        "source_wasm_real_runtime_executed": True,
        "source_program_local_count": program.local_count,
        "source_instruction_count": len(program.operations),
        "sample_count": len(samples),
        "witnesses": witnesses,
        "sampled_complete_control_traces": selected_traces,
        "total_executed_control_transitions": sum(
            x["instruction_transitions"] for x in witnesses
        ),
        "fuel_exhausted_on_mathematically_terminating_large_input": resource_exhaustion,
        "fuel_exhaustion_means_nontermination": False,
        "neural_routing_of_branch_targets": False,
        "full_wasm_supported": False,
        "general_gcd_all_i32_semantics_proved": False,
        "claims_arbitrary_neural_decompilation": False,
        "limitations": (
            "One original WASM gcd(x,12) repeated-subtraction algorithm. "
            "Negative input uses documented sentinel zero. Host routing "
            "handles control frames/PC, v49 constructed gates compute i32 "
            "arithmetic. Wasmtime and mathematical GCD verify finite inputs; "
            "large positive value is resource-limited, not semantically "
            "nonterminating. No memory/calls or unbounded proof."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "GCD_cases": report["sample_count"],
        "control_steps": report["total_executed_control_transitions"],
        "original_wasm_independently_validated": True,
        "large_input_fuel_limited_not_nonterminating": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
