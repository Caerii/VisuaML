"""v52 original WASM + v51 + fixed neural PC routing differential evidence.

Six different real WASM source modules are independently validated in
Wasmtime. The constructed routing weights are IDENTICAL for all programs;
program-specific lexical jump targets live in separate token tables.

The executor's NEXT instruction is selected from the actual gated one-hot PC
output. Host instruction dispatch, lexical target preprocessing, runtime
labels, and numerical stack/local effects remain separate from PC routing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path

import numpy as np
import wasmtime
from scripts.probe_wasm_control_flow_v51 import fixtures
from scripts.probe_wasm_gcd_v51_1 import gcd_program
from visuaml.compiler_lab.bit_neural_vm_v49 import construct_bit_weights
from visuaml.compiler_lab.neural_control_routing_v52 import (
    compile_route_tokens,
    construct_router_weights,
    execute_routed_control,
    validate_route_tokens,
)
from visuaml.compiler_lab.wasm_control_flow_v51 import (
    ControlFuelExhausted,
    ControlOperation,
    ControlProgram,
    encode_control,
    execute_control,
)

SCHEMA = "visuaml.original-wasm-neural-pc-router.v52"
GCD_CASES = (-123, -1, *range(25))


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _programs() -> dict:
    programs = dict(fixtures())
    programs["gcd_12"] = (gcd_program(), GCD_CASES)
    return programs


def _trap_reference(engine: wasmtime.Engine) -> dict:
    # A program deliberately containing an infinite real WASM loop.
    infinite = ControlProgram(0, (
        ControlOperation("loop"), ControlOperation("br", 0),
        ControlOperation("end"), ControlOperation("i32.const", 1),
        ControlOperation("end"),
    ))
    never = encode_control(infinite)
    wasmtime.Module.validate(engine, never)
    try:
        execute_routed_control(infinite, 0, fuel=60)
    except ControlFuelExhausted:
        resource_exhaustion = True
    else:
        resource_exhaustion = False
    if not resource_exhaustion:
        raise AssertionError("Constructed PC gate failed to respect explicit fuel")
    return {
        "wasm_sha256": _sha(never),
        "original_wasm_hex": never.hex(),
        "bounded_resource_exhaustion_observed": True,
        "resource_exhaustion_is_language_trap": False,
        "all_input_termination_proved": False,
    }


def run() -> dict:
    engine = wasmtime.Engine()
    fixed = construct_router_weights()
    arithmetic = construct_bit_weights()
    weight_sha = fixed.digest()
    bit_sha = arithmetic.digest()
    programs = []
    witnesses = []
    total_steps = 0
    branch_routes = 0
    conditional_routes = 0
    source_path = Path(__file__).resolve().parents[1]
    for name, (source, inputs) in _programs().items():
        raw = encode_control(source)
        wasmtime.Module.validate(engine, raw)
        module = wasmtime.Module(engine, raw)
        store = wasmtime.Store(engine)
        fn = wasmtime.Instance(store, module, []).exports(store)["run"]
        route_tokens = compile_route_tokens(source)
        validate_route_tokens(route_tokens, source)
        if (fixed.digest() != weight_sha or arithmetic.digest() != bit_sha):
            raise AssertionError("Neural routing weights changed by program")
        program_info = {
            "name": name,
            "source_program_sha256": source.digest(),
            "source_wasm_sha256": _sha(raw),
            "canonical_original_wasm_hex": raw.hex(),
            "program_route_tokens_sha256": route_tokens.digest(),
            "shared_fixed_router_sha256": weight_sha,
            "shared_i32_arithmetic_weights_sha256": bit_sha,
            "reviewed_lexical_route_rows": [
                {
                    "pc": idx,
                    "default_pc": int(np.argmax(route_tokens.sequential[idx])),
                    "alternate_pc": int(np.argmax(route_tokens.alternate[idx])),
                    "guard_kind": route_tokens.guard_kind[idx].tolist(),
                }
                for idx in range(len(source.operations))
            ],
            "input_count": len(inputs),
        }
        programs.append(program_info)
        for argument in inputs:
            machine = execute_routed_control(
                source, argument, fuel=1500,
                tokens=route_tokens, fixed=fixed, bit_fixed=arithmetic,
            )
            independent = execute_control(source, argument, fuel=1500,
                                          fixed=arithmetic)
            actual_wasm = fn(store, argument)
            if not (machine.result_i32 == independent.value == actual_wasm):
                raise AssertionError(
                    f"Neural route, host semantics, original WASM diverge: {name} {argument}"
                )
            if machine.steps != independent.transitions:
                raise AssertionError("PC gate changed number of executed instructions")
            full_trace = []
            for gated, original in zip(
                machine.trace, independent.trace, strict=True
            ):
                for key in (
                    "step", "pc", "opcode", "next_pc",
                    "stack_i32", "locals_i32", "control_labels", "halted",
                ):
                    if gated[key] != original[key]:
                        raise AssertionError(
                            f"Actual routed PC altered source semantics: {name} {argument} {key}"
                        )
                # Counterfactual calculation explicitly records guard outputs
                # from the ACTUAL constructed one-hot routing forward pass.
                router = gated["neural_route"]
                if router["branch_guard_gate"]:
                    branch_routes += 1
                if source.operations[gated["pc"]].opcode in ("if", "br_if"):
                    conditional_routes += 1
                full_trace.append({
                    "step": gated["step"],
                    "pc": gated["pc"],
                    "opcode": gated["opcode"],
                    "next_pc": gated["next_pc"],
                    "route": gated["neural_route"],
                    "stack_i32": gated["stack_i32"],
                    "locals_i32": gated["locals_i32"],
                    "control_labels": gated["control_labels"],
                    "halted": gated["halted"],
                })
            total_steps += machine.steps
            canonical = json.dumps(full_trace, sort_keys=True,
                                   separators=(",", ":")).encode()
            observation = {
                "program": name,
                "argument_i32": argument,
                "original_wasmtime_result": actual_wasm,
                "v51_host_pc_result": independent.value,
                "v52_actual_neural_pc_result": machine.result_i32,
                "executed_instruction_count": machine.steps,
                "complete_neural_routing_trace_sha256": _sha(canonical),
                "passed": True,
            }
            # Keep exact complete state/PC traces for representative
            # inputs, and deterministic full-trace digests for every case.
            if argument in (-1, 0, 1, 5, 12, 17):
                observation["complete_machine_trace"] = full_trace
            witnesses.append(observation)

    if len({x["shared_fixed_router_sha256"] for x in programs}) != 1:
        raise AssertionError("Interpreter's fixed PC gates were specialized")
    if len({x["program_route_tokens_sha256"] for x in programs}) != len(programs):
        raise AssertionError("Different control programs share same route tokens")
    sources = (
        "visuaml/compiler_lab/neural_control_routing_v52.py",
        "visuaml/compiler_lab/wasm_control_flow_v51.py",
        "visuaml/compiler_lab/bit_neural_vm_v49.py",
        "scripts/probe_neural_control_router_v52.py",
    )
    return {
        "schema": SCHEMA,
        "source_commit": os.environ.get("VISUAML_SOURCE_COMMIT"),
        "source_file_sha256": {
            path: _sha((source_path / path).read_bytes()) for path in sources
        },
        "program_count": len(programs),
        "witness_count": len(witnesses),
        "total_per_instruction_comparisons": total_steps,
        "taken_jump_routing_events": branch_routes,
        "conditional_instruction_events": conditional_routes,
        "fixed_router_weights_sha256": weight_sha,
        "fixed_i32_arithmetic_weights_sha256": bit_sha,
        "actual_onehot_pc_controls_next_instruction": True,
        "same_analytic_router_weights_for_all_programs": True,
        "branch_targets_compiled_from_source_into_program_data": True,
        "runtime_lexical_control_frames_still_host_managed": True,
        "runtime_wasm_opcode_dispatch_still_host_managed": True,
        "standard_transformer_softmax_executed": False,
        "neural_program_counter_selected_by_matrix_gates": True,
        "complete_neural_wasm_interpreter_constructed": False,
        "arbitrary_pretrained_decompilation_implemented": False,
        "all_input_semantics_formally_proved": False,
        "wasmtime_original_source_validated": True,
        "separate_v51_host_interpreter_compared_every_step": True,
        "reviewed_programs": programs,
        "witnesses": witnesses,
        "bounded_nontermination_reference": _trap_reference(engine),
        "wasmtime_version": version("wasmtime"),
        "numpy_version": np.__version__,
        "limitations": (
            "Actual constructed, fixed-weight one-hot bilinear selection of "
            "next PCs from lexical target table data for six reviewed WASM "
            "programs. Runtime control frame/stack effects and opcode dispatch "
            "remain host logic. Original WASM is validated and executed by "
            "independent Wasmtime and every executed state compared to v51. "
            "No neural label-stack compiler, general WASM typing, memory, "
            "calls, standard transformer layers, arbitrary decompilation "
            "or all-input equivalence theorem."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        result, indent=2, sort_keys=True, allow_nan=False
    ) + "\n", encoding="utf-8")
    print(json.dumps({
        "programs": result["program_count"],
        "witnesses": result["witness_count"],
        "instruction_steps": result["total_per_instruction_comparisons"],
        "fixed_gate_digest": result["fixed_router_weights_sha256"],
        "branch_routes": result["taken_jump_routing_events"],
        "host_label_stack": result["runtime_lexical_control_frames_still_host_managed"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
