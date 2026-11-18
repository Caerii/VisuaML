"""Source-bound v51 real WASM control-flow and bit-gate differential experiment.

Recorded source modules are ACTUAL canonical WASM bytes accepted by Wasmtime.
Only arithmetic is the constructed v49 bit-gated circuit. The branch/label
stack, program-counter updates and execution fuel remain host-interpreted;
these are not yet neural attention transitions.
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
from visuaml.compiler_lab.wasm_control_flow_v51 import (
    ControlFuelExhausted,
    ControlOperation,
    ControlProgram,
    decode_control,
    encode_control,
    execute_control,
)

SCHEMA = "visuaml.structured-control-differential.v51"


def _op(name: str, immediate: int | None = None) -> ControlOperation:
    return ControlOperation(name, immediate)


def fixtures() -> dict[str, tuple[ControlProgram, tuple[int, ...]]]:
    def p(*ops, locals_: int = 0) -> ControlProgram:
        return ControlProgram(locals_, tuple(
            _op(name, immediate) for name, immediate in ops
        ))

    def i(name, value=None):
        return (name, value)

    return {
        "countdown_sum": (
            p(
                i("block"), i("loop"),
                i("local.get", 0), i("i32.eqz"), i("br_if", 1),
                i("local.get", 1), i("local.get", 0), i("i32.add"),
                i("local.set", 1),
                i("local.get", 0), i("i32.const", 1), i("i32.sub"),
                i("local.set", 0), i("br", 0),
                i("end"), i("end"), i("local.get", 1), i("end"),
                locals_=1,
            ),
            tuple(range(13)),
        ),
        "if_else_negative": (
            p(
                i("local.get", 0), i("i32.const", 0),
                i("i32.lt_s"), i("if"),
                i("i32.const", 11), i("local.set", 1), i("else"),
                i("i32.const", 22), i("local.set", 1),
                i("end"), i("local.get", 1), i("end"),
                locals_=1,
            ),
            (-(1 << 31), -100, -1, 0, 1, 100, (1 << 31)-1),
        ),
        "branch_out_of_block": (
            p(
                i("block"), i("local.get", 0), i("i32.const", 0),
                i("i32.gt_s"), i("br_if", 0),
                i("i32.const", 7), i("local.set", 1),
                i("end"), i("local.get", 1), i("end"),
                locals_=1,
            ),
            (-128, -1, 0, 1, 128),
        ),
        "early_return": (
            p(
                i("local.get", 0), i("i32.const", 0),
                i("i32.gt_s"), i("if"), i("i32.const", 9),
                i("return"), i("end"),
                i("i32.const", -3), i("end"),
            ),
            (-128, -1, 0, 1, 128),
        ),
        "i32_overflow": (
            p(
                i("local.get", 0), i("i32.const", 1),
                i("i32.add"), i("end"),
            ),
            (-(1 << 31), -1, 0, (1 << 31)-1),
        ),
    }


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _wasmtime_fuel_limit(wasm: bytes) -> bool:
    """Independently confirm a nonterminating real WASM loop hits engine fuel.

    Fuel exhaustion is an *execution resource limit*, not a WASM opcode trap.
    """
    settings = wasmtime.Config()
    settings.consume_fuel = True
    engine = wasmtime.Engine(settings)
    wasmtime.Module.validate(engine, wasm)
    store = wasmtime.Store(engine)
    store.set_fuel(500)
    run = wasmtime.Instance(
        store, wasmtime.Module(engine, wasm), [],
    ).exports(store)["run"]
    try:
        run(store, 0)
    except wasmtime.Trap:
        return True
    return False


def run() -> dict:
    engine = wasmtime.Engine()
    witnesses = []
    programs = []
    for name, (original, inputs) in fixtures().items():
        raw = encode_control(original)
        roundtrip = decode_control(raw)
        if roundtrip != original:
            raise AssertionError("Original canonical structured WASM not roundtrip stable")
        # This is the independent full-WASM validator: VisuaML's own
        # envelope/structure validator does NOT implement full stack typing.
        wasmtime.Module.validate(engine, raw)
        module = wasmtime.Module(engine, raw)
        store = wasmtime.Store(engine)
        callable_ = wasmtime.Instance(store, module, []).exports(store)["run"]
        actual_weights = set()
        for value in inputs:
            interpreted = execute_control(original, value, fuel=500)
            original_wasm = callable_(store, value)
            if interpreted.value != original_wasm:
                raise AssertionError(
                    f"Constructed-bit arithmetic/control diverged: {name}, {value}"
                )
            actual_weights.add(interpreted.constructed_arithmetic_weights_sha256)
            witnesses.append({
                "program": name, "input_i32": value,
                "source_wasmtime_i32": original_wasm,
                "host_control_bit_neural_i32": interpreted.value,
                "instruction_transitions": interpreted.transitions,
                "complete_host_control_trace": list(interpreted.trace),
                "passed": True,
            })
        if len(actual_weights) != 1:
            raise AssertionError("Different arithmetic weights selected by input")
        programs.append({
            "name": name,
            "canonical_wasm_sha256": _sha(raw),
            "canonical_wasm_hex": raw.hex(),
            "source_program_sha256": original.digest(),
            "operation_count": len(original.operations),
            "all_inputs_used_same_bit_weights": True,
        })
    infinite = ControlProgram(0, (
        _op("loop"), _op("br", 0), _op("end"),
        _op("i32.const", 1), _op("end"),
    ))
    never = encode_control(infinite)
    wasmtime.Module.validate(engine, never)
    try:
        execute_control(infinite, 0, fuel=50)
    except ControlFuelExhausted:
        host_fuel = True
    else:
        host_fuel = False
    wasm_fuel = _wasmtime_fuel_limit(never)
    if not host_fuel or not wasm_fuel:
        raise AssertionError("Either host or independent Wasmtime missed finite fuel")
    source_dir = Path(__file__).resolve().parents[1]
    files = (
        "visuaml/compiler_lab/wasm_control_flow_v51.py",
        "visuaml/compiler_lab/bit_neural_vm_v49.py",
        "scripts/probe_wasm_control_flow_v51.py",
    )
    return {
        "schema": SCHEMA,
        "source_commit": os.environ.get("VISUAML_SOURCE_COMMIT"),
        "source_file_sha256": {
            name: _sha((source_dir/name).read_bytes()) for name in files
        },
        "source_wasm_independently_validated": True,
        "real_wasmtime_executed": True,
        "all_finite_witnesses_passed": True,
        "program_count": len(programs),
        "programs": programs,
        "witnesses": witnesses,
        "total_control_state_transitions": sum(
            w["instruction_transitions"] for w in witnesses
        ),
        "bounded_nontermination_example_sha256": _sha(never),
        "nontermination_original_wasm_hex": never.hex(),
        "nonterminating_case_rejected_with_bounded_fuel": True,
        "host_fuel_exhaustion_observed": host_fuel,
        "independent_wasmtime_fuel_exhaustion_observed": wasm_fuel,
        "bit_neural_arithmetic_executed": True,
        "neural_pc_routing_implemented": False,
        "general_wasm_supported": False,
        "unrestricted_wasm_validation_implemented": False,
        "all_possible_input_equivalence_proved": False,
        "standard_transformer_executed": False,
        "arbitrary_decompilation_executed": False,
        "wasmtime_version": version("wasmtime"),
        "numpy_version": np.__version__,
        "limitations": (
            "Canonical one-function WASM block/loop/if/else/br/br_if/return "
            "plus reviewed i32 subset, empty block signatures, no imports, "
            "memory, calls, globals, tables, traps or additional numeric types. "
            "Host CPU dispatches control frames and PC; v49 constructed "
            "bit gates execute i32 arithmetic. Finite real Wasmtime witnesses "
            "and explicit fuel exhaustion are not formal general equivalence "
            "or full neural branch execution."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "programs": result["program_count"],
        "witnesses": len(result["witnesses"]),
        "control_state_steps": result["total_control_state_transitions"],
        "wasmtime_fuel_exhaustion_observed": result[
            "independent_wasmtime_fuel_exhaustion_observed"
        ],
        "control_routing_on_neural_weights": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
