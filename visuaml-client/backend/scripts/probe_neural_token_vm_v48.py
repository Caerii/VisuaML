"""v48 source-bound seven-program neural machine + Wasmtime differential report.

Each case executes ORIGINAL canonical WebAssembly in independent Wasmtime,
the v40 typed interpreter, and one fixed-weight NumPy/ReGLU neural-state
machine. Every instruction trace is checked against v40. Full output witnesses
and sampled complete traces are preserved. No trained weights and no general
WASM or arbitrary neural decompilation claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import wasmtime

from visuaml.compiler_lab.fixed_neural_token_vm_v48 import (
    OPCODES,
    construct_fixed_weights,
    decode_state,
    detokenize_program,
    initial_neural_state,
    neural_step,
    tokenize_program,
)
from visuaml.compiler_lab.wasm_state_vm_v40 import (
    I32Program,
    VMOperation,
    encode_program,
    execute_program,
    initial_state,
    step,
)

SCHEMA = "visuaml.multi-program-neural-matrix-vm-evidence.v48"
SAMPLE_TRACES = {-128, -1, 0, 1, 127}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _op(name: str, immediate: int | None = None) -> VMOperation:
    return VMOperation(name, immediate)


def programs() -> dict[str, I32Program]:
    return {
        "identity": I32Program(0, (
            _op("local.get", 0), _op("end"),
        )),
        "affine": I32Program(0, (
            _op("local.get", 0), _op("i32.const", 7),
            _op("i32.add"), _op("end"),
        )),
        "quadratic": I32Program(0, (
            _op("local.get", 0), _op("local.get", 0),
            _op("i32.mul"), _op("i32.const", 3),
            _op("i32.sub"), _op("end"),
        )),
        "locals_tee": I32Program(2, (
            _op("local.get", 0), _op("local.set", 1),
            _op("local.get", 1), _op("i32.const", 2),
            _op("i32.mul"), _op("local.tee", 2),
            _op("local.get", 2), _op("i32.add"), _op("end"),
        )),
        "eqz": I32Program(0, (
            _op("local.get", 0), _op("i32.eqz"), _op("end"),
        )),
        "less_than": I32Program(0, (
            _op("local.get", 0), _op("i32.const", 3),
            _op("i32.lt_s"), _op("end"),
        )),
        "greater_than_drop": I32Program(0, (
            _op("local.get", 0), _op("i32.const", 9),
            _op("drop"), _op("i32.const", 1),
            _op("i32.gt_s"), _op("end"),
        )),
    }


def run() -> dict:
    weights = construct_fixed_weights()
    fixed_sha = weights.digest()
    engine = wasmtime.Engine()
    records = []
    full_witnesses = []
    instruction_steps = 0
    for name, program in programs().items():
        original_bytes = encode_program(program)
        wasmtime.Module.validate(engine, original_bytes)
        module = wasmtime.Module(engine, original_bytes)
        store = wasmtime.Store(engine)
        instance = wasmtime.Instance(store, module, [])
        independent_wasm = instance.exports(store)["run"]
        tokens = tokenize_program(program)
        if detokenize_program(tokens) != program:
            raise AssertionError("Structured token decoder changed typed source")
        detailed = []
        for x in range(-128, 128):
            neural = initial_neural_state(program, x)
            reference = initial_state(program, x)
            trace = []
            for operation in program.operations:
                neural = neural_step(neural, tokens, weights)
                reference, transition = step(program, reference)
                decoded = decode_state(neural, program)
                if decoded != reference or transition.opcode != operation.opcode:
                    raise AssertionError(
                        f"Neural instruction divergence: {name}, x={x}, {operation.opcode}"
                    )
                if x in SAMPLE_TRACES:
                    trace.append({
                        "opcode": operation.opcode,
                        "state": asdict(decoded),
                        "reference_transition": asdict(transition),
                    })
                instruction_steps += 1
            output = reference.stack[0]
            observed_wasm = independent_wasm(store, x)
            independently_typed = execute_program(program, x)
            if (not neural.halted or observed_wasm != output
                    or independently_typed.value != output):
                raise AssertionError(f"Original WebAssembly mismatch: {name}, {x}")
            full_witnesses.append({
                "program": name, "input_i32": x,
                "neural_result_i32": output,
                "wasmtime_result_i32": observed_wasm,
                "typed_reference_i32": independently_typed.value,
                "step_count": len(program.operations),
                "passed": True,
            })
            if x in SAMPLE_TRACES:
                detailed.append({"input_i32": x, "transitions": trace})
        records.append({
            "name": name,
            "wasm_sha256": _sha(original_bytes),
            "canonical_wasm_hex": original_bytes.hex(),
            "typed_program_sha256": program.digest(),
            "structured_tokens_sha256": tokens.digest(),
            "structured_token_decoder_roundtrip": True,
            "fixed_weights_sha256": fixed_sha,
            "encoded_opcodes": [o.opcode for o in program.operations],
            "instruction_count": len(program.operations),
            "sample_full_step_traces": detailed,
        })
    if len({r["fixed_weights_sha256"] for r in records}) != 1:
        raise AssertionError("The fixed interpreter was incorrectly specialized by program")
    if len({r["structured_tokens_sha256"] for r in records}) != 7:
        raise AssertionError("Program token identities were conflated")
    backend = Path(__file__).resolve().parents[1]
    sources = (
        "visuaml/compiler_lab/fixed_neural_token_vm_v48.py",
        "visuaml/compiler_lab/wasm_state_vm_v40.py",
        "scripts/probe_neural_token_vm_v48.py",
    )
    return {
        "schema": SCHEMA,
        "source_commit": os.environ.get("VISUAML_SOURCE_COMMIT"),
        "source_file_sha256": {
            name: _sha((backend / name).read_bytes()) for name in sources
        },
        "fixed_weights_sha256": fixed_sha,
        "fixed_weights_reused_without_retraining": True,
        "number_of_programs": len(records),
        "exact_input_domain": [-128, 127],
        "all_inputs_checked_per_program": 256,
        "total_output_witnesses": len(full_witnesses),
        "total_per_instruction_comparisons": instruction_steps,
        "instruction_set": list(OPCODES),
        "programs": records,
        "witnesses": full_witnesses,
        "independent_wasmtime_executed": True,
        "independent_v40_per_instruction_state_compared": True,
        "neural_machine_architecture": (
            "Analytically constructed fixed one-hot routing matrices and "
            "ReLU/ReGLU gated state transitions with data-supplied programs"
        ),
        "standard_transformer_layer_weights_constructed": False,
        "standard_softmax_attention_executed": False,
        "general_wasm_mvp_supported": False,
        "control_flow_branches_supported": False,
        "memory_load_store_supported": False,
        "i32_modular_overflow_semantics_supported": False,
        "structured_token_disassembly_performed": True,
        "arbitrary_neural_weight_decompilation": False,
        "unique_program_recovery_from_behavior": False,
        "measured_speedup_against_wasmtime": False,
        "numpy_version": np.__version__,
        "wasmtime_version": version("wasmtime"),
        "limitations": (
            "7 reviewed source WASM functions under identical fixed routing and "
            "gated-arithmetic weights, real code as token matrices. Exact finite "
            "integer arithmetic and no modular wrap. Fixed architecture is not "
            "a stock softmax transformer, arbitrary WASM interpreter, or unique "
            "learned-weight decompiler. Source bytes and complete numerical/"
            "per-step witnesses are retained for independent inspection."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "programs": result["number_of_programs"],
        "inputs": result["total_output_witnesses"],
        "steps": result["total_per_instruction_comparisons"],
        "same_weights_sha256": result["fixed_weights_sha256"],
        "independent_wasmtime_executed": result["independent_wasmtime_executed"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
